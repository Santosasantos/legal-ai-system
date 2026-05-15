"""
Central configuration — all tuneable values live here.
Switch LLM provider by changing LLM_PROVIDER in your .env file.
Supported providers: ollama | openai | gemini | anthropic
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal
import os


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────────────────────────
    APP_NAME: str = "Legal AI System — Pearson Specter Litt"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── LLM Provider (swap here) ─────────────────────────────────────────────
    LLM_PROVIDER: Literal["ollama", "openai", "gemini", "anthropic"] = "gemini"

    # ── Ollama (default / local) ─────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    OLLAMA_VISION_MODEL: str = "qwen2.5vl:3b"       # OCR / vision
    OLLAMA_TEXT_MODEL: str = "qwen2.5:3b"            # generation / extraction
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"     # embeddings

    # ── OpenAI (optional) ────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_TEXT_MODEL: str = "gpt-4o"
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"

    # ── Gemini (optional) ────────────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_TEXT_MODEL: str = "gemini-2.0-flash"   # fast + free tier, handles vision too

    # ── Anthropic (optional) ─────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_TEXT_MODEL: str = "claude-3-5-sonnet-20241022"

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    CHROMA_HOST: str = "chromadb"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION_NAME: str = "legal_documents"

    # ── Chunking ─────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512           # tokens per chunk
    CHUNK_OVERLAP: int = 64         # overlap between consecutive chunks
    MIN_CHUNK_SIZE: int = 50        # discard chunks smaller than this

    # ── Retrieval ────────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 8        # passages returned per query
    RETRIEVAL_SCORE_THRESHOLD: float = 0.35   # minimum similarity score

    # ── Paths ────────────────────────────────────────────────────────────────
    DATA_DIR: str = "/app/data"
    SAMPLE_DOCS_DIR: str = "/app/data/sample_documents"
    OUTPUTS_DIR: str = "/app/data/outputs"
    FEEDBACK_DIR: str = "/app/data/feedback"
    PREFERENCE_RULES_FILE: str = "/app/data/feedback/preference_rules.json"

    # ── Draft generation ─────────────────────────────────────────────────────
    DRAFT_TYPE: str = "case_fact_summary"   # default draft type
    MAX_DRAFT_TOKENS: int = 2048

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
