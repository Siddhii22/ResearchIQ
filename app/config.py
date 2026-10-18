"""Environment-backed application settings."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Settings loaded from environment variables and an optional .env file."""

    app_name: str = os.getenv("APP_NAME", "ResearchIQ")
    llm_api_key: str | None = os.getenv("LLM_API_KEY")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")
    llm_api_base_url: str = os.getenv("LLM_API_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    dense_top_k: int = int(os.getenv("DENSE_TOP_K", "20"))
    bm25_top_k: int = int(os.getenv("BM25_TOP_K", "20"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "20"))
    final_top_k: int = int(os.getenv("FINAL_TOP_K", "5"))
    arxiv_search_limit: int = int(os.getenv("ARXIV_SEARCH_LIMIT", "5"))
    rerank_model: str = os.getenv(
        "RERANK_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )


settings = Settings()