"""
Configuration module for the RAG Pipeline.

Loads all settings from environment variables (via .env file) into a
single Settings dataclass. Any module that needs configuration imports
from here:

    from src.config import settings
    print(settings.chunk_size)  # 500

WHY a centralized config?
- Avoids hardcoded values scattered across files
- Makes it easy to change settings without editing source code
- Keeps API keys out of the codebase (loaded from .env)
"""

import os
import logging
from dataclasses import dataclass
from pathlib import Path


from dotenv import load_dotenv

# ── Load .env file ──────────────────────────────────────────────
# load_dotenv() reads the .env file in the project root and sets
# the key-value pairs as environment variables in the current process.
# If a variable is already set in the OS environment, it WON'T be
# overridden (OS env takes precedence). This is useful for deployment
# where you set env vars directly instead of using .env files.

# Find the project root (parent of 'src/')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Settings:
    """
    Central configuration container.

    All values come from environment variables with sensible defaults.
    Using a dataclass instead of a plain dict gives us:
    - Type hints (IDE autocomplete)
    - Attribute access (settings.chunk_size instead of settings["chunk_size"])
    - Immutability awareness (easier to reason about)
    """

    # --- Paths ---
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data" / "documents"

    # --- LLM ---
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # --- Qdrant ---
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "rag_documents")

    # --- Embedding ---
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    # --- Chunking ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # --- Retrieval ---
    top_k: int = int(os.getenv("TOP_K", "5"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))

    # --- Logging ---
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


# ── Create a single global instance ────────────────────────────
# This is the "singleton pattern" — every module imports the same
# 'settings' object rather than creating its own. This ensures
# consistency across the entire application.
settings = Settings()


# ── Configure logging ──────────────────────────────────────────
# Set up a project-wide logger format. Any module can now do:
#   import logging
#   logger = logging.getLogger(__name__)
#   logger.info("Processing document...")
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
