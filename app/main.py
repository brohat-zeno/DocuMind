"""FastAPI application factory and client lifecycle."""

from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager

from fastapi import FastAPI
from openai import OpenAI
from qdrant_client import QdrantClient
from redis import Redis

from app.api.routes.chat import router as chat_router
from app.api.routes.ingestion import router as ingestion_router
from app.core.config import EmbeddingBackend, LLMBackend, Settings
from app.core.exceptions import register_handlers
from app.core.logging import configure_logging
from app.db.session import create_database
from app.models.base import Base
from app.repositories.booking_repo import SQLBookingRepository
from app.repositories.chat_memory_repo import RedisChatMemoryRepository
from app.repositories.document_repo import SQLDocumentRepository
from app.repositories.vector_repo import QdrantVectorRepository
from app.services.booking_service import BookingService
from app.services.embedding import (
    EmbeddingProvider,
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from app.services.ingestion_service import IngestionService
from app.services.llm import GroqLLMProvider, LLMProvider, OpenAILLMProvider
from app.services.rag_service import RAGService


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construct the application with clients owned by its lifespan."""
    configuration = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        with ExitStack() as resources:
            engine, sessions = create_database(configuration.database_url)
            resources.callback(engine.dispose)
            Base.metadata.create_all(engine)
            effective = configuration
            embeddings: EmbeddingProvider
            llm: LLMProvider
            openai: OpenAI | None = None
            if (
                configuration.embedding_provider == EmbeddingBackend.OPENAI
                or configuration.llm_provider == LLMBackend.OPENAI
            ):
                openai = OpenAI(
                    api_key=configuration.openai_api_key.get_secret_value() or "not-configured",
                    timeout=configuration.provider_timeout_seconds,
                )
                resources.callback(openai.close)
            if configuration.embedding_provider == EmbeddingBackend.LOCAL:
                local = LocalEmbeddingProvider(configuration)
                effective = configuration.model_copy(
                    update={"embedding_dimension": local.dimension}
                )
                embeddings = local
            else:
                assert openai is not None
                embeddings = OpenAIEmbeddingProvider(openai, effective)
            if configuration.llm_provider == LLMBackend.GROQ:
                groq = OpenAI(
                    api_key=configuration.groq_api_key.get_secret_value() or "not-configured",
                    base_url=configuration.groq_base_url,
                    timeout=configuration.provider_timeout_seconds,
                )
                resources.callback(groq.close)
                llm = GroqLLMProvider(groq, effective)
            else:
                assert openai is not None
                llm = OpenAILLMProvider(openai, effective)
            qdrant = QdrantClient(
                url=configuration.qdrant_url,
                timeout=int(configuration.provider_timeout_seconds),
                check_compatibility=False,
            )
            resources.callback(qdrant.close)
            redis = Redis.from_url(
                configuration.redis_url,
                socket_timeout=configuration.provider_timeout_seconds,
                socket_connect_timeout=configuration.provider_timeout_seconds,
            )
            resources.callback(redis.close)
            vectors = QdrantVectorRepository(qdrant, effective)
            application.state.ingestion_service = IngestionService(
                SQLDocumentRepository(sessions),
                vectors,
                embeddings,
                effective,
            )
            memory = RedisChatMemoryRepository(redis, configuration)
            booking = BookingService(SQLBookingRepository(sessions), memory, llm, configuration)
            application.state.rag_service = RAGService(
                memory,
                vectors,
                embeddings,
                llm,
                effective,
                booking,
            )
            yield

    application = FastAPI(
        title=configuration.app_name,
        description="DocuMind: document ingestion, conversational RAG, and interview booking.",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    register_handlers(application)
    application.include_router(ingestion_router)
    application.include_router(chat_router)
    return application
