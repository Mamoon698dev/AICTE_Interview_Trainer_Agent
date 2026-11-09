"""
config.py – Centralised configuration loaded from environment variables.
All modules import from here instead of reading .env directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env (or env.example fallback) from the project root
_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env", override=False)
load_dotenv(_root / "env.example", override=False)


class Config:
    # ── IBM Watsonx ────────────────────────────────────────────
    WATSONX_API_KEY: str = os.getenv("WATSONX_API_KEY", "")
    WATSONX_PROJECT_ID: str = os.getenv("WATSONX_PROJECT_ID", "")
    WATSONX_URL: str = os.getenv(
        "WATSONX_URL", "https://us-south.ml.cloud.ibm.com"
    )

    # ── IBM Granite models ─────────────────────────────────────
    GRANITE_MODEL_ID: str = os.getenv(
        "GRANITE_MODEL_ID", "ibm/granite-4-h-small"
    )
    GRANITE_EMBEDDING_MODEL: str = os.getenv(
        "GRANITE_EMBEDDING_MODEL", "ibm/slate-30m-english-rtrvr-v2"
    )

    # ── RAG / Vector store ─────────────────────────────────────
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR", "./data/chroma_db"
    )
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

    # ── LLM generation params ──────────────────────────────────
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "1024"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))

    # ── Application ────────────────────────────────────────────
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ── Data directories ───────────────────────────────────────
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
    RESUME_DIR: Path = Path(os.getenv("RESUME_DIR", "./data/resumes"))
    KB_DIR: Path = Path(os.getenv("KB_DIR", "./data/knowledge_base"))

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create required data directories if they don't exist."""
        for d in (cls.DATA_DIR, cls.RESUME_DIR, cls.KB_DIR,
                  Path(cls.CHROMA_PERSIST_DIR)):
            Path(d).mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls) -> list[str]:
        """Return a list of missing mandatory env-var names."""
        missing = []
        if not cls.WATSONX_API_KEY or cls.WATSONX_API_KEY == "your_ibm_cloud_api_key_here":
            missing.append("WATSONX_API_KEY")
        if not cls.WATSONX_PROJECT_ID or cls.WATSONX_PROJECT_ID == "your_watsonx_project_id_here":
            missing.append("WATSONX_PROJECT_ID")
        return missing


config = Config()
