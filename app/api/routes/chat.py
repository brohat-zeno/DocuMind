"""Conversational API transport."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_rag_service
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    service: Annotated[RAGService, Depends(get_rag_service)],
) -> ChatResponse:
    """Answer a document question or process a booking turn."""
    return service.chat(request)
