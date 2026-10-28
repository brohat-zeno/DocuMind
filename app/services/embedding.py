"""Embedding provider abstraction and OpenAI adapter."""

import math
from threading import Lock
from typing import Protocol

from openai import OpenAI
from pydantic import BaseModel

from app.core.config import Settings
from app.core.exceptions import DependencyError


class EmbeddingProvider(Protocol):
    """Embed batches of text."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input, preserving order."""
        ...


class OpenAIEmbeddingProvider:
    """Use configurable OpenAI embeddings."""

    def __init__(self, client: OpenAI, settings: Settings) -> None:
        self._client: OpenAI = client
        self._settings: Settings = settings

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Request and validate a batch of embedding vectors."""
        if not self._settings.openai_api_key.get_secret_value():
            raise DependencyError("OPENAI_API_KEY is not configured.")
        try:
            response = self._client.embeddings.create(
                model=self._settings.embedding_model,
                input=texts,
                dimensions=self._settings.embedding_dimension,
            )
            return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        except Exception as exc:
            raise DependencyError("The embedding provider is unavailable.") from exc


class LocalEmbeddingProvider:
    """Generate CPU embeddings with a lazily loaded, disk-cached FastEmbed model."""

    def __init__(self, settings: Settings) -> None:
        from fastembed import TextEmbedding

        self._settings: Settings = settings
        self._model: TextEmbedding | None = None
        self._lock: Lock = Lock()
        descriptions = [
            LocalModelDescription.model_validate(item)
            for item in TextEmbedding.list_supported_models()
        ]
        description = next(
            (item for item in descriptions if item.model == settings.local_embedding_model), None
        )
        if description is None:
            raise DependencyError("LOCAL_EMBEDDING_MODEL is not supported by FastEmbed.")
        self.dimension: int = description.dim
        if (
            "embedding_dimension" in settings.model_fields_set
            and settings.embedding_dimension != self.dimension
        ):
            raise DependencyError(
                f"EMBEDDING_DIMENSION must be {self.dimension} for "
                f"{settings.local_embedding_model}; got {settings.embedding_dimension}."
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Load once, preserve batch ordering, and validate every generated vector."""
        from fastembed import TextEmbedding

        if not texts:
            return []
        try:
            # Serialize first load and inference for the shared tokenizer/model instance.
            with self._lock:
                if self._model is None:
                    self._model = TextEmbedding(
                        model_name=self._settings.local_embedding_model,
                        cache_dir=self._settings.embedding_cache_dir,
                        threads=self._settings.local_embedding_threads,
                    )
                vectors = [
                    [float(value) for value in vector]
                    for vector in self._model.embed(
                        texts, batch_size=self._settings.embedding_batch_size
                    )
                ]
            if len(vectors) != len(texts) or any(
                len(vector) != self.dimension or not all(math.isfinite(value) for value in vector)
                for vector in vectors
            ):
                raise ValueError("Invalid local embedding result")
            return vectors
        except Exception as exc:
            raise DependencyError(
                "Local embeddings are unavailable; check model download access and cache storage."
            ) from exc


class LocalModelDescription(BaseModel):
    """Read the model name and dimension from FastEmbed's model registry."""

    model: str
    dim: int
