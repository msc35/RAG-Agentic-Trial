"""Phase 6 — Evaluation harness.

Run with:
    python -m evals.run_eval
    python -m evals.run_eval --output evals/results_custom.json

What it measures:
    1. Retrieval hit rate  — did the right source file appear in the top-5 chunks?
       Separates retrieval failures from generation failures.

    2. LLM-as-judge scores — the model rates each answer 1–5 on:
         • correctness   (does the answer contain the right information?)
         • faithfulness  (does it stick to the documents, no hallucination?)
       A separate judge call per question, returning structured JSON.

    3. "I don't know" accuracy — for unanswerable questions, did the model
       correctly decline rather than hallucinate?

Why split retrieval from generation?
    If the judge scores are low, you need to know whether to fix the chunking /
    retrieval (Phase 2) or the generation prompt (Phase 3). These two metrics
    tell you which half to fix.

Why LLM-as-judge?
    Cheaper and faster than human annotation at scale, decent correlation with
    human judgment. Weakness: the judge can be wrong, especially on nuanced
    technical content — spot-check low-scoring answers manually.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from src.generate import generate
from src.llm import complete
from src.retriever import Retriever

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
RESULTS_DIR = Path(__file__).parent

_JUDGE_SYSTEM = """\
You are an expert evaluator for a retrieval-augmented generation (RAG) system.
You will be given a question, a reference answer, and a generated answer.
Score the generated answer on two dimensions, each from 1 to 5:

correctness  — Does the generated answer convey the same key facts as the reference?
               5 = fully correct, 3 = partially correct, 1 = wrong or missing key facts.

faithfulness — Does the generated answer stay grounded in the source documents?
               5 = no hallucination, 1 = significant invented facts.

For unanswerable questions (reference says the docs don't contain the answer):
  • 5 for both dimensions if the model correctly says it doesn't know.
  • 1 for both if it fabricates an answer.

Return ONLY valid JSON in this exact format (no markdown, no extra keys):
{"correctness": <int>, "faithfulness": <int>, "reason": "<one sentence>"}
"""


@dataclass
class QuestionResult:
    id: str
    question: str
    answerable: bool
    generated_answer: str
    reference_answer: str
    sources_returned: list[str]      # filenames the pipeline cited
    expected_sources: list[str]
    retrieval_hit: bool              # True if any expected source was returned
    correctness: int = 0
    faithfulness: int = 0
    judge_reason: str = ""
    latency_ms: float = 0.0


@dataclass
class EvalSummary:
    timestamp: str
    total_questions: int
    answerable_count: int
    unanswerable_count: int
    retrieval_hit_rate: float        # over answerable questions only
    avg_correctness: float
    avg_faithfulness: float
    idk_accuracy: float              # % of unanswerable Qs correctly declined
    results: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #

def judge(question: str, reference: str, generated: str) -> tuple[int, int, str]:
    """Ask the LLM to rate the generated answer. Returns (correctness, faithfulness, reason)."""
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Reference answer: {reference}\n\n"
                f"Generated answer: {generated}"
            ),
        },
    ]
    try:
        message = complete(messages, json_mode=True, temperature=0.0)
        data = json.loads(message.content)
        return (
            int(data.get("correctness", 1)),
            int(data.get("faithfulness", 1)),
            str(data.get("reason", "")),
        )
    except Exception as exc:
        return 1, 1, f"Judge error: {exc}"


# --------------------------------------------------------------------------- #
# Retrieval hit check
# --------------------------------------------------------------------------- #

def _retrieval_hit(expected_sources: list[str], returned_sources: list[str]) -> bool:
    """True if at least one expected source appears in the returned sources."""
    if not expected_sources:
        return True  # unanswerable: hit is not applicable, don't penalise
    returned_set = set(returned_sources)
    return any(src in returned_set for src in expected_sources)


# --------------------------------------------------------------------------- #
# Unanswerable check
# --------------------------------------------------------------------------- #

_IDK_PHRASES = [
    "i don't have enough information",
    "i don't have information",
    "not in the provided",
    "not contained in",
    "cannot find",
    "no information",
    "documents do not",
    "not mentioned",
    "not discussed",
    "not covered",
    "i cannot answer",
]


def _correctly_declined(answer: str) -> bool:
    """Heuristic: did the model say it doesn't know?"""
    lower = answer.lower()
    return any(phrase in lower for phrase in _IDK_PHRASES)


# --------------------------------------------------------------------------- #
# Main eval loop
# --------------------------------------------------------------------------- #

def run_eval(output_path: Path | None = None) -> EvalSummary:
    golden = json.loads(GOLDEN_SET_PATH.read_text())

    print(f"Loading retriever ({len(golden)} questions to evaluate)...")
    retriever = Retriever()
    print(f"Corpus: {len(retriever._corpus)} chunks\n")

    results: list[QuestionResult] = []

    for i, item in enumerate(golden, start=1):
        qid = item["id"]
        question = item["question"]
        reference = item["reference_answer"]
        expected = item["expected_sources"]
        answerable = item["answerable"]

        print(f"[{i:02d}/{len(golden)}] {qid}: {question[:70]}...")

        t0 = time.perf_counter()
        chunks = retriever.retrieve(question, top_k=5)
        gen_result = generate(question, chunks)
        latency_ms = (time.perf_counter() - t0) * 1000

        returned_sources = [src for src, _ in gen_result.sources]
        hit = _retrieval_hit(expected, returned_sources)

        print(f"         retrieval hit={hit}  sources={returned_sources[:2]}")

        correctness, faithfulness, reason = judge(question, reference, gen_result.answer)
        print(f"         correctness={correctness}  faithfulness={faithfulness}  reason={reason[:80]}")
        print()

        results.append(
            QuestionResult(
                id=qid,
                question=question,
                answerable=answerable,
                generated_answer=gen_result.answer,
                reference_answer=reference,
                sources_returned=returned_sources,
                expected_sources=expected,
                retrieval_hit=hit,
                correctness=correctness,
                faithfulness=faithfulness,
                judge_reason=reason,
                latency_ms=round(latency_ms, 1),
            )
        )

    # ----------- aggregate metrics ----------- #
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]

    retrieval_hit_rate = (
        sum(r.retrieval_hit for r in answerable) / len(answerable)
        if answerable else 0.0
    )
    avg_correctness = sum(r.correctness for r in results) / len(results)
    avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
    idk_accuracy = (
        sum(1 for r in unanswerable if _correctly_declined(r.generated_answer))
        / len(unanswerable)
        if unanswerable else 0.0
    )

    summary = EvalSummary(
        timestamp=datetime.utcnow().isoformat(),
        total_questions=len(results),
        answerable_count=len(answerable),
        unanswerable_count=len(unanswerable),
        retrieval_hit_rate=round(retrieval_hit_rate, 3),
        avg_correctness=round(avg_correctness, 2),
        avg_faithfulness=round(avg_faithfulness, 2),
        idk_accuracy=round(idk_accuracy, 3),
        results=[asdict(r) for r in results],
    )

    _print_table(results, summary)

    # Save to disk
    out = output_path or (RESULTS_DIR / f"results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    out.write_text(json.dumps(asdict(summary), indent=2))
    print(f"\nResults saved → {out}")

    return summary


# --------------------------------------------------------------------------- #
# Pretty-print
# --------------------------------------------------------------------------- #

def _print_table(results: list[QuestionResult], summary: EvalSummary) -> None:
    print("\n" + "=" * 80)
    print(f"{'ID':<6} {'ANS?':<6} {'HIT':<5} {'COR':<5} {'FAI':<5} {'ms':<8} QUESTION")
    print("-" * 80)
    for r in results:
        hit_str  = "✓" if r.retrieval_hit else "✗"
        ans_str  = "Y" if r.answerable else "N"
        print(
            f"{r.id:<6} {ans_str:<6} {hit_str:<5} {r.correctness:<5} "
            f"{r.faithfulness:<5} {r.latency_ms:<8.0f} {r.question[:45]}..."
        )
    print("=" * 80)
    print(f"\nSUMMARY ({summary.total_questions} questions)")
    print(f"  Retrieval hit rate (answerable only) : {summary.retrieval_hit_rate:.0%}")
    print(f"  Avg correctness  (1–5)               : {summary.avg_correctness:.2f}")
    print(f"  Avg faithfulness (1–5)               : {summary.avg_faithfulness:.2f}")
    print(f"  I-don't-know accuracy (unanswerable) : {summary.idk_accuracy:.0%}")
    print()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG evaluation harness.")
    parser.add_argument("--output", type=Path, default=None, help="Path to save results JSON.")
    args = parser.parse_args()
    run_eval(output_path=args.output)
