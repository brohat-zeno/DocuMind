"""Environment-based application configuration."""

from enum import StrEnum
from typing import Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMBackend(StrEnum):
    """Available chat backends."""

    OPENAI = "openai"
    GROQ = "groq"


class EmbeddingBackend(StrEnum):
    """Available embedding backends."""

    OPENAI = "openai"
    LOCAL = "local"


class StructuredMode(StrEnum):
    """Groq structured-output selection."""

    AUTO = "auto"
    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"


class Settings(BaseSettings):
    """Configure application dependencies and behavior."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_name: str = "DocuMind"
    llm_provider: LLMBackend = LLMBackend.OPENAI
    embedding_provider: EmbeddingBackend = EmbeddingBackend.OPENAI
    groq_api_key: SecretStr = SecretStr("")
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_chat_model: str = "openai/gpt-oss-20b"
    groq_structured_mode: StructuredMode = StructuredMode.AUTO
    groq_schema_models: tuple[str, ...] = (
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
    )
    structured_output_retries: int = Field(default=1, ge=0, le=1)
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: str = "./data/documind-embedding-cache"
    local_embedding_threads: int = Field(default=2, gt=0)
    openai_api_key: SecretStr = SecretStr("")
    openai_chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = Field(default=1536, gt=0)
    embedding_batch_size: int = Field(default=64, gt=0)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documind_documents"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "sqlite:///./data/documind.db"
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    history_turns: int = Field(default=10, gt=0)
    memory_ttl_seconds: int = Field(default=86400, gt=0)
    retrieval_top_k: int = Field(default=5, gt=0)
    retrieval_score_threshold: float = Field(default=0.58, ge=-1, le=1)
    provider_timeout_seconds: float = Field(default=60, gt=0)
    booking_timezone: str = "UTC"

    booking_confirmation_phrases: tuple[str, ...] = (
        "yes",
        "yes please",
        "yes confirm",
        "yes book it",
        "confirm",
        "confirmed",
        "please confirm",
        "go ahead",
        "yes that is correct",
    )

    @field_validator("booking_timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        """Require a recognized IANA booking timezone."""
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("Use a valid IANA booking timezone.") from exc
        return value

    @model_validator(mode="after")
    def valid_chunk_defaults(self) -> Self:
        """Reject inconsistent default chunking configuration."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        return self
