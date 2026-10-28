"""Measure five handbook queries against real local embeddings and Qdrant."""

import json

from qdrant_client import QdrantClient, models

from app.core.config import EmbeddingBackend, Settings
from app.services.embedding import LocalEmbeddingProvider


def main() -> None:
    """Print cosine scores without exposing credentials or document text."""
    settings = Settings()
    if settings.embedding_provider != EmbeddingBackend.LOCAL:
        raise SystemExit("This calibration expects EMBEDDING_PROVIDER=local.")
    provider = LocalEmbeddingProvider(settings)
    queries = [
        (
            "Foundation and Advanced, in that order",
            "Which training plans does the DocuMind Learning Centre offer?",
        ),
        (
            "NPR 7500 per month and six instructor-led workshops",
            "What is the monthly price and workshop allowance of the Advanced plan?",
        ),
        (
            "Monday through Friday, 09:00 to 17:00 Nepal time",
            "When is the DocuMind Learning Centre support desk open?",
        ),
        ("not stated", "What scholarship policy does the handbook specify?"),
        ("not stated", "What accommodation policy does the handbook specify?"),
    ]
    vectors = provider.embed([query for _, query in queries])
    rows: list[dict[str, object]] = []
    client = QdrantClient(url=settings.qdrant_url, check_compatibility=False)
    try:
        for (expected, query), vector in zip(queries, vectors, strict=True):
            points = client.query_points(
                settings.qdrant_collection,
                query=vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename", match=models.MatchValue(value="handbook.pdf")
                        )
                    ]
                ),
                limit=100,
                with_payload=True,
            ).points
            rows.append(
                {
                    "question": query,
                    "expected": expected,
                    "chunks": [
                        {
                            "score": point.score,
                            "filename": (point.payload or {}).get("filename"),
                            "chunk_index": (point.payload or {}).get("chunk_index"),
                            "strategy": (point.payload or {}).get("chunking_strategy"),
                        }
                        for point in points
                    ],
                }
            )
    finally:
        client.close()
    print(
        json.dumps(
            {
                "fixture": "tests/fixtures/handbook.pdf",
                "model": settings.local_embedding_model,
                "dimension": provider.dimension,
                "threshold": settings.retrieval_score_threshold,
                "questions": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
