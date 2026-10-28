"""Provider behavior without credentials, inference downloads, or external calls."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr
from qdrant_client import QdrantClient, models

from app.core.config import EmbeddingBackend, Settings, StructuredMode
from app.core.exceptions import DependencyError, VectorStoreError
from app.repositories.vector_repo import QdrantVectorRepository
from app.schemas.booking import BookingAction, BookingExtraction
from app.schemas.chat import ChatMessage, Intent, IntentDecision, MessageRole
from app.services.embedding import LocalEmbeddingProvider, OpenAIEmbeddingProvider
from app.services.llm import GroqLLMProvider, OpenAILLMProvider


def completion(content: str) -> SimpleNamespace:
    """Build a minimal SDK response fixture."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_groq_strict_schemas() -> None:
    """Use JSON-schema parsing for both intent classification and booking extraction."""
    client = MagicMock()
    settings = Settings(groq_api_key=SecretStr("unit-test"))
    provider = GroqLLMProvider(client, settings)
    outputs = [
        IntentDecision(intent=Intent.BOOKING, changes_topic=False),
        BookingExtraction(name="Ada", email=None, date=None, time=None, action=BookingAction.NONE),
    ]
    for output in outputs:
        client.chat.completions.parse.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=output))]
        )
        assert provider.structured([], type(output)) == output
        assert client.chat.completions.parse.call_args.kwargs["response_format"] is type(output)
        assert client.chat.completions.parse.call_args.kwargs["model"] == settings.groq_chat_model


@pytest.mark.parametrize("invalid", ["not JSON", '{"intent":"invalid","changes_topic":false}'])
def test_groq_json_retries_once(invalid: str) -> None:
    """Retry invalid syntax or invalid fields, then return a validated model."""
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        completion(invalid),
        completion('{"intent":"question","changes_topic":false}'),
    ]
    settings = Settings(
        groq_api_key=SecretStr("unit-test"), groq_chat_model="llama-3.3-70b-versatile"
    )
    result = GroqLLMProvider(client, settings).structured([], IntentDecision)
    assert result.intent == Intent.QUESTION
    assert client.chat.completions.create.call_count == 2
    for call in client.chat.completions.create.call_args_list:
        assert call.kwargs["response_format"] == {"type": "json_object"}
        assert "JSON schema" in call.kwargs["messages"][0]["content"]


def test_groq_json_stops_after_one_retry() -> None:
    """Fail clearly after two invalid JSON responses without an unbounded loop."""
    client = MagicMock()
    client.chat.completions.create.return_value = completion("{}")
    settings = Settings(
        groq_api_key=SecretStr("unit-test"), groq_structured_mode=StructuredMode.JSON_OBJECT
    )
    with pytest.raises(DependencyError, match="after validation retries"):
        GroqLLMProvider(client, settings).structured([], IntentDecision)
    assert client.chat.completions.create.call_count == 2


def test_groq_json_booking_fields() -> None:
    """Validate nullable booking fields in JSON object mode."""
    client = MagicMock()
    client.chat.completions.create.return_value = completion(
        '{"name":"Ada","email":null,"date":null,"time":null,"action":"none"}'
    )
    provider = GroqLLMProvider(
        client,
        Settings(
            groq_api_key=SecretStr("unit-test"), groq_structured_mode=StructuredMode.JSON_OBJECT
        ),
    )
    assert provider.structured([], BookingExtraction).name == "Ada"


def test_groq_text_and_missing_key() -> None:
    """Generate ordinary answers and fail before a request when the key is absent."""
    client = MagicMock()
    client.chat.completions.create.return_value = completion("Grounded answer")
    messages = [ChatMessage(role=MessageRole.USER, content="Question")]
    provider = GroqLLMProvider(client, Settings(groq_api_key=SecretStr("unit-test")))
    assert provider.complete(messages) == "Grounded answer"
    client.reset_mock()
    with pytest.raises(DependencyError, match="GROQ_API_KEY"):
        GroqLLMProvider(client, Settings()).complete(messages)
    client.chat.completions.create.assert_not_called()


def test_local_model_dimension_loading_and_cache() -> None:
    """Derive dimensions from the model registry and load only on the first embed call."""
    with patch("fastembed.TextEmbedding") as model:
        model.list_supported_models.return_value = [{"model": "BAAI/bge-small-en-v1.5", "dim": 384}]
        model.return_value.embed.side_effect = [
            iter([[0.5] * 384, [0.25] * 384]),
            iter([[0.1] * 384]),
        ]
        settings = Settings(embedding_provider=EmbeddingBackend.LOCAL)
        provider = LocalEmbeddingProvider(settings)
        assert provider.dimension == 384
        model.assert_not_called()
        assert provider.embed([]) == []
        assert provider.embed(["first", "second"]) == [[0.5] * 384, [0.25] * 384]
        assert provider.embed(["third"]) == [[0.1] * 384]
        model.assert_called_once_with(
            model_name=settings.local_embedding_model,
            cache_dir=settings.embedding_cache_dir,
            threads=settings.local_embedding_threads,
        )


def test_local_dimension_configuration_mismatch() -> None:
    """Reject an explicit incompatible dimension before model inference."""
    with patch("fastembed.TextEmbedding") as model:
        model.list_supported_models.return_value = [{"model": "BAAI/bge-small-en-v1.5", "dim": 384}]
        with pytest.raises(DependencyError, match="must be 384"):
            LocalEmbeddingProvider(Settings(embedding_dimension=1536))


def test_local_invalid_vectors_and_download_errors() -> None:
    """Do not let malformed vectors or raw download errors escape the provider."""
    with patch("fastembed.TextEmbedding") as model:
        model.list_supported_models.return_value = [{"model": "BAAI/bge-small-en-v1.5", "dim": 384}]
        model.return_value.embed.return_value = iter([[0.1] * 3])
        provider = LocalEmbeddingProvider(Settings(embedding_dimension=384))
        with pytest.raises(DependencyError, match="Local embeddings"):
            provider.embed(["text"])
        model.side_effect = RuntimeError("download failed")
        with pytest.raises(DependencyError, match="cache storage"):
            LocalEmbeddingProvider(Settings(embedding_dimension=384)).embed(["text"])


def test_qdrant_rejects_existing_wrong_dimension() -> None:
    """Explain dimension conflicts on both ingestion and retrieval without changing data."""
    client = QdrantClient(":memory:")
    client.create_collection(
        "documind_documents",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    repo = QdrantVectorRepository(client, Settings(embedding_dimension=384))
    with pytest.raises(VectorStoreError, match="1536.*384"):
        repo.upsert([])
    with pytest.raises(VectorStoreError, match="1536.*384"):
        repo.search([0.1] * 384, 5, 0.3)
    assert client.count("documind_documents").count == 0
    client.close()


def test_openai_implementations_remain_available() -> None:
    """Preserve OpenAI embedding ordering and chat behavior."""
    client = MagicMock()
    settings = Settings(openai_api_key=SecretStr("unit-test"), embedding_dimension=3)
    client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[0.0, 1.0, 0.0]),
            SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0]),
        ]
    )
    assert OpenAIEmbeddingProvider(client, settings).embed(["a", "b"]) == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    client.chat.completions.create.return_value = completion("Answer")
    assert OpenAILLMProvider(client, settings).complete([]) == "Answer"
