"""Thin provider wrapper.

This is the ONE place model/provider choices live. The rest of the codebase
calls `complete()` and `embed()` and never imports the OpenAI SDK directly.
Swapping to Claude, adding a fallback, or changing models is a change here only.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from openai import OpenAI

from .config import settings


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    """Lazily construct a single OpenAI client.

    Lazy + cached so importing this module never fails just because the API
    key is missing (e.g. during tests that don't hit the network).
    """
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=settings.openai_api_key)


def complete(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    model: str | None = None,
):
    """Run a chat completion and return the assistant message.

    Returns the full message object (not just the string) so later phases can
    read `.content` for normal answers and `.tool_calls` for the agent loop.
    """
    kwargs: dict[str, Any] = {
        "model": model or settings.llm_model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = _client().chat.completions.create(**kwargs)
    return response.choices[0].message


def stream_complete(
    messages: list[dict[str, Any]],
    model: str | None = None,
):
    """Stream chat completion tokens as a generator of string chunks.

    Yields each delta string as it arrives so callers can forward tokens
    to the client without buffering the whole response.
    """
    stream = _client().chat.completions.create(
        model=model or settings.llm_model,
        messages=messages,
        temperature=0.0,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed a list of strings, returning one vector per input.

    Batched to stay well under request limits. Order is preserved.
    """
    if not texts:
        return []

    model = model or settings.embed_model
    vectors: list[list[float]] = []
    batch_size = 100  # comfortably under the API's per-request input cap
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = _client().embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors
