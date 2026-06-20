"""Central settings, loaded from the environment.

Everything configurable lives here so the rest of the code never reads
os.environ directly. Import `settings` and use it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env (if present) into the process environment exactly once, here.
load_dotenv()

# Project paths, derived from this file's location so they work from anywhere.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
STORE_DIR = DATA_DIR / "store"


def _int_env(name: str, default: int) -> int:
    """Read an int from the env, falling back to a default if unset/blank."""
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    """Immutable bundle of runtime settings."""

    # --- Models / provider ---
    openai_api_key: str
    llm_model: str
    embed_model: str

    # --- Ingestion / chunking ---
    chunk_size_tokens: int
    chunk_overlap_tokens: int

    # --- Vector store ---
    collection_name: str
    store_dir: Path
    pdf_dir: Path


def get_settings() -> Settings:
    """Build a Settings object from current environment variables."""
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        chunk_size_tokens=_int_env("CHUNK_SIZE_TOKENS", 500),
        chunk_overlap_tokens=_int_env("CHUNK_OVERLAP_TOKENS", 50),
        collection_name=os.getenv("COLLECTION_NAME", "docs"),
        store_dir=STORE_DIR,
        pdf_dir=PDF_DIR,
    )


# A ready-to-import singleton for convenience.
settings = get_settings()
