from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///./rag.db"

    # "fake" works offline (hashing embeddings + extractive answers): dev, tests, CI.
    # "openai" calls any OpenAI-compatible API for both embeddings and answers.
    llm_provider: Literal["fake", "openai"] = "fake"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    llm_timeout_seconds: float = 60.0

    # Ingestion
    max_upload_mb: int = 10
    chunk_size: int = 800  # characters
    chunk_overlap: int = 150  # characters
    embedding_batch_size: int = 64

    # Retrieval
    default_top_k: int = 4
    # Below this cosine similarity a chunk is considered unrelated to the question.
    min_score: float = 0.2


@lru_cache
def get_settings() -> Settings:
    return Settings()
