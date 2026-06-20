"""Phase 7 — Structured logging and per-request tracing.

Usage:
    from src.logging_conf import configure_logging, get_logger, set_trace_id

    configure_logging()          # call once at startup
    log = get_logger("module")
    log.info("event_name", key=value, ...)

Trace IDs:
    Each request/run gets a UUID stored in a contextvars.ContextVar.
    Every log line automatically picks it up via the add_trace_id processor,
    so you can grep a single trace_id to follow one question end-to-end.

    set_trace_id("abc123")       # set at request boundary (API middleware)
    get_trace_id()               # read from anywhere in the call stack

Output:
    JSON when JSON_LOGS=1 (production / grep-friendly).
    Human-readable coloured output otherwise (dev / terminal).
"""

from __future__ import annotations

import contextvars
import logging
import os
import sys
import uuid

import structlog

# --------------------------------------------------------------------------- #
# Trace ID — lives in a context variable so it propagates through the call
# stack without being passed explicitly to every function.
# --------------------------------------------------------------------------- #

_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default="-"
)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def set_trace_id(tid: str | None = None) -> str:
    """Set (or generate) the trace ID for the current context. Returns the id."""
    tid = tid or new_trace_id()
    _trace_id_var.set(tid)
    return tid


def get_trace_id() -> str:
    return _trace_id_var.get()


# --------------------------------------------------------------------------- #
# Custom structlog processor: inject trace_id into every log record
# --------------------------------------------------------------------------- #

def _add_trace_id(
    logger: object, method: str, event_dict: dict
) -> dict:
    event_dict["trace_id"] = get_trace_id()
    return event_dict


# --------------------------------------------------------------------------- #
# Configure
# --------------------------------------------------------------------------- #

_configured = False


def configure_logging(json_logs: bool | None = None) -> None:
    """Configure structlog. Call exactly once at application startup.

    json_logs defaults to True when the JSON_LOGS env var is set (useful for
    piping logs to a collector), False otherwise (human-readable dev output).
    """
    global _configured
    if _configured:
        return
    _configured = True

    if json_logs is None:
        json_logs = os.getenv("JSON_LOGS", "").lower() in ("1", "true", "yes")

    shared_processors: list = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_trace_id,
        structlog.processors.StackInfoRenderer(),
    ]

    if json_logs:
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also quieten noisy third-party loggers that emit via stdlib logging.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(name: str = "") -> structlog.BoundLogger:
    """Return a structlog logger bound to name. Configures with defaults if needed."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)
