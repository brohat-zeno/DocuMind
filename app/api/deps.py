"""Typed dependency accessors."""

from typing import cast

from fastapi import Request

from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService


def get_ingestion_service(request: Request) -> IngestionService:
    """Get the lifespan-managed ingestion service."""
    return cast(IngestionService, request.app.state.ingestion_service)


def get_rag_service(request: Request) -> RAGService:
    """Get the lifespan-managed RAG service."""
    return cast(RAGService, request.app.state.rag_service)
