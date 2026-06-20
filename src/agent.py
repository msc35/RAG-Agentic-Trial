"""Phase 4 — Agentic tool-calling loop.

The key difference from plain RAG:
  • RAG always retrieves.
  • An agent decides WHEN to retrieve, WHAT to search for, and WHETHER to
    search again if the first result wasn't enough.

Public API:
    agent = Agent(retriever)
    result = agent.run("What did MentorNet change about curriculum learning?")
    print(result.answer)
    print(result.tool_calls_made)

Loop mechanics (raw OpenAI tool-calling, no framework):
  1. Send the user question + tool schemas to the LLM.
  2. If the model returns tool_calls → execute each tool, append results, loop.
  3. If the model returns a plain text answer → done, return it.
  4. Hard cap at MAX_ITERATIONS to prevent runaway loops and token burn.

Tools exposed to the agent:
  • search_docs(query)     — hybrid + reranked retrieval (Phase 2).
  • list_documents()       — tells the agent which PDFs are loaded, so it can
                             tailor its search query to a specific paper.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .generate import GenerationResult, _build_context_block, _extract_sources
from .llm import complete
from .logging_conf import get_logger
from .retriever import RetrievedChunk, Retriever

_log = get_logger("agent")

MAX_ITERATIONS = 5

# --------------------------------------------------------------------------- #
# Tool schemas (OpenAI function-calling format)
# --------------------------------------------------------------------------- #

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search the ingested technical documents for chunks relevant to a query. "
                "Use this whenever you need information from the documents to answer the question. "
                "You can call it multiple times with different queries if the first result is incomplete."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused search query. Be specific — narrow queries retrieve better chunks than broad ones.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to return. Default 5. Use up to 10 for broad questions.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "List the PDF documents currently loaded in the knowledge base. "
                "Call this first if the user asks about a specific paper by name, "
                "so you can confirm it is available before searching."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

_SYSTEM_PROMPT = """\
You are a research assistant with access to a knowledge base of technical PDF documents.

To answer a question:
1. Use `list_documents` if you need to know which papers are available.
2. Use `search_docs` with a focused query to retrieve relevant passages.
3. You may search multiple times with different queries if the first search is insufficient.
4. When you have enough information, write your final answer using ONLY what the retrieved passages say.
5. Cite sources as (filename, p.N) after each claim.
6. If the documents do not contain the answer, say so explicitly — do NOT guess.
"""


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #

@dataclass
class AgentResult:
    answer: str
    sources: list[tuple[str, int]] = field(default_factory=list)
    tool_calls_made: list[str] = field(default_factory=list)   # log of what was called
    iterations: int = 0


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class Agent:
    """Runs the tool-calling loop against the ingested document collection."""

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever
        # Accumulate retrieved chunks across all search_docs calls in one turn,
        # so the final source list reflects everything the agent actually used.
        self._retrieved: list[RetrievedChunk] = []

    # ------------------------------------------------------------------ #
    # Tool implementations
    # ------------------------------------------------------------------ #

    def _search_docs(self, query: str, top_k: int = 5) -> str:
        """Execute a retrieval and return the result as a formatted string."""
        top_k = min(max(top_k, 1), 10)  # clamp to [1, 10]
        chunks = self._retriever.retrieve(query, top_k=top_k)
        self._retrieved.extend(chunks)
        if not chunks:
            return "No relevant passages found for that query."
        return _build_context_block(chunks)

    def _list_documents(self) -> str:
        """Return the unique filenames in the collection."""
        corpus = self._retriever._corpus
        names = sorted({doc["source"] for doc in corpus})
        if not names:
            return "No documents are currently loaded."
        return "Available documents:\n" + "\n".join(f"  • {n}" for n in names)

    def _dispatch_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Route a tool call to its implementation. Returns a string result."""
        t0 = time.perf_counter()
        try:
            if name == "search_docs":
                result = self._search_docs(
                    query=arguments["query"],
                    top_k=arguments.get("top_k", 5),
                )
            elif name == "list_documents":
                result = self._list_documents()
            else:
                result = f"Unknown tool: {name}"
        except Exception as exc:
            result = f"Tool error in {name}: {exc}"
            _log.warning("tool_error", tool=name, args=arguments, error=str(exc))

        _log.info(
            "tool_call",
            tool=name,
            args={k: str(v)[:80] for k, v in arguments.items()},
            result_len=len(result),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return result

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self, question: str) -> AgentResult:
        """Run the agent loop for a single user question."""
        self._retrieved = []  # reset per question
        tool_calls_log: list[str] = []
        t_start = time.perf_counter()

        _log.info("agent_start", question=question[:120])

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        for iteration in range(1, MAX_ITERATIONS + 1):
            message = complete(messages, tools=_TOOLS)

            # ---- Case 1: the model wants to call tools ----
            if message.tool_calls:
                # Append the assistant's tool-call message first (required by API).
                messages.append(message)

                for tc in message.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    tool_calls_log.append(f"{fn_name}({fn_args})")

                    result_text = self._dispatch_tool(fn_name, fn_args)

                    # Append the tool result — must use tool_call_id to match.
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_text,
                        }
                    )
                continue  # loop — let the model react to the tool results

            # ---- Case 2: the model produced a final answer ----
            answer = (message.content or "").strip()
            if not answer:
                answer = "I don't have enough information in the provided documents to answer this question."

            sources = _extract_sources(self._retrieved)
            _log.info(
                "agent_done",
                iterations=iteration,
                tool_calls=tool_calls_log,
                n_sources=len(sources),
                latency_ms=round((time.perf_counter() - t_start) * 1000, 1),
            )
            return AgentResult(
                answer=answer,
                sources=sources,
                tool_calls_made=tool_calls_log,
                iterations=iteration,
            )

        # ---- Loop cap reached without a final answer ----
        # Ask for a plain answer with what we have so far.
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the maximum number of tool calls. "
                    "Please give your best answer now based on what you have retrieved so far."
                ),
            }
        )
        message = complete(messages)
        answer = (message.content or "I was unable to find a complete answer within the allowed steps.").strip()
        sources = _extract_sources(self._retrieved)
        return AgentResult(
            answer=answer,
            sources=sources,
            tool_calls_made=tool_calls_log,
            iterations=MAX_ITERATIONS,
        )
