"""Adapter contract checks without external services."""

from unittest.mock import MagicMock, create_autospec
from uuid import uuid4

from qdrant_client import QdrantClient
from redis import Redis

from app.core.config import Settings
from app.repositories.chat_memory_repo import RedisChatMemoryRepository
from app.repositories.vector_repo import QdrantVectorRepository
from app.schemas.booking import BookingState
from app.schemas.chat import ChatMessage, MessageRole
from app.schemas.ingestion import ChunkingStrategy, VectorChunk, VectorPayload


def test_qdrant_payload_search_and_document_cleanup() -> None:
    """Exercise the production adapter against Qdrant's in-process memory implementation."""
    client = QdrantClient(":memory:")
    settings = Settings(embedding_dimension=3)
    repo = QdrantVectorRepository(client, settings)
    assert repo.search([1, 0, 0], 5, 0.3) == []
    first, second = uuid4(), uuid4()
    points = [
        VectorChunk(
            id=uuid4(),
            embedding=vector,
            payload=VectorPayload(
                document_id=document_id,
                chunk_index=0,
                text="Evidence",
                filename="a.txt",
                chunking_strategy=ChunkingStrategy.FIXED_SIZE,
            ),
        )
        for document_id, vector in [(first, [1, 0, 0]), (second, [0, 1, 0])]
    ]
    repo.upsert(points)
    results = repo.search([1, 0, 0], 5, 0.5)
    assert len(results) == 1
    assert results[0].payload.document_id == first
    repo.delete_document(first)
    assert repo.search([1, 0, 0], 5, 0.5) == []
    assert repo.search([0, 1, 0], 5, 0.5)[0].payload.document_id == second
    repo.delete_document(first)
    client.close()


def test_redis_keys_expiration_and_history_bound() -> None:
    """Verify exact keys, TTLs, serialization, and transactional history trimming."""
    client = create_autospec(Redis, instance=True)
    settings = Settings(history_turns=2, memory_ttl_seconds=123)
    repo = RedisChatMemoryRepository(client, settings)
    messages = [
        ChatMessage(role=MessageRole.USER, content="Question"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Answer"),
    ]
    client.lrange.return_value = [message.model_dump_json().encode() for message in messages]
    assert repo.history("s") == messages
    client.lrange.assert_called_once_with("chat:s:history", -4, -1)
    pipe = MagicMock()
    client.pipeline.return_value.__enter__.return_value = pipe
    repo.append("s", messages)
    client.pipeline.assert_called_once_with(transaction=True)
    pipe.rpush.assert_called_once_with(
        "chat:s:history", *(message.model_dump_json() for message in messages)
    )
    pipe.ltrim.assert_called_once_with("chat:s:history", -4, -1)
    pipe.expire.assert_called_once_with("chat:s:history", 123)
    pipe.execute.assert_called_once()
    state = BookingState(name="Ada")
    repo.save_booking("s", state)
    client.set.assert_called_once_with("chat:s:booking", state.model_dump_json(), ex=123)
    client.get.return_value = state.model_dump_json().encode()
    assert repo.booking("s") == state
    client.get.return_value = None
    assert repo.booking("s") is None
    repo.clear_booking("s")
    client.delete.assert_called_once_with("chat:s:booking")


def test_tuned_threshold_preserves_evidence_and_rejects_unrelated_scores() -> None:
    """Keep measured education-level similarity while rejecting the unrelated rocket score."""
    import math

    client = QdrantClient(":memory:")
    settings = Settings(embedding_dimension=3)
    repo = QdrantVectorRepository(client, settings)
    evidence_id, unrelated_id = uuid4(), uuid4()
    repo.upsert(
        [
            VectorChunk(
                id=uuid4(),
                embedding=[score, math.sqrt(1 - score * score), 0.0],
                payload=VectorPayload(
                    document_id=document_id,
                    chunk_index=0,
                    text="Threshold fixture",
                    filename="scores.txt",
                    chunking_strategy=ChunkingStrategy.FIXED_SIZE,
                ),
            )
            for document_id, score in [(evidence_id, 0.5979), (unrelated_id, 0.576844)]
        ]
    )
    results = repo.search([1, 0, 0], 5, settings.retrieval_score_threshold)
    assert [result.payload.document_id for result in results] == [evidence_id]
    client.close()
