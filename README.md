# DocuMind backend

A Python 3.11+ FastAPI backend with document ingestion, conversational retrieval-augmented answers, and multi-turn interview booking. It exposes exactly:

- `POST /api/v1/documents`
- `GET /api/v1/documents/{document_id}`
- `POST /api/v1/chat`

There is no frontend, authentication, streaming, background queue, or calendar/email integration. Swagger and ReDoc are disabled.

## Architecture

```text
HTTP multipart / JSON
        |
FastAPI routes --> injected services
                      |
                      +-- IngestionService
                      |     validate -> extract -> chunk -> embed -> store
                      |     failures after writes -> vector + SQL cleanup
                      |
                      +-- RAGService
                      |     Redis history -> classify -> rewrite -> retrieve
                      |     -> build grounded prompt -> generate -> Redis history
                      |
                      +-- BookingService
                            extract fields -> validate -> ask -> confirm -> SQL
                      |
                 repository / provider Protocols
                      |
              +-------+--------+--------+--------+
              |                |        |        |
         SQLite/SQLAlchemy   Qdrant   Redis    Providers
          metadata, slots    chunks   memory   FastEmbed/OpenAI + Groq/OpenAI
```

Routes handle HTTP transport only. Services receive repositories/providers through constructors. Repository methods own data access and short transactions. Synchronous handlers run in FastAPI's worker thread pool. The app lifespan owns and closes all clients and the SQL engine.

The RAG pipeline is implemented explicitly without LangChain, LlamaIndex, or prebuilt chains. Follow-up questions are rewritten using the last configured number of exchanges. Generated answers are instructed to use only retrieved context; empty retrieval returns a deterministic insufficient-information answer.

## Setup with Docker

Requirements: Docker Engine/Compose and a Groq API key. The example configuration uses local CPU embeddings, so OpenAI billing is not required. The first local embedding request needs network access to download the model; subsequent starts reuse its cache.

```sh
cp .env.example .env
# Edit .env and set GROQ_API_KEY.
docker compose up --build
# Wait for Application startup complete; send API requests from another terminal.
```

The app listens on `http://localhost:8000` by default. If that port is occupied, set `APP_PORT=8001` in `.env` before starting and use port 8001 in the curl examples. Compose starts app, Qdrant, and Redis. Named volumes persist SQLite, vectors, Redis data, and downloaded embedding model files. Compose supplies the internal service URLs and SQLite path. Dependencies connect on use, so an operation attempted before a dependency is ready can return 503.

The service can boot without a chat API key. Requests requiring that provider then return a clear 503 configuration error. No secret is included in the source or example configuration.

## Local development

Install Python 3.11+ and [uv](https://docs.astral.sh/uv/), then:

```sh
uv sync --frozen --python 3.11
cp .env.example .env
# Set GROQ_API_KEY and URLs for running Qdrant and Redis services.
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

The supplied Compose file keeps Qdrant and Redis internal to its network. For a locally running app, point `.env` to separately reachable instances; otherwise run the app in Compose too.

## Configuration

Settings are read from environment variables and an optional `.env` file.

| Variable | Default | Meaning |
| --- | --- | --- |
| `APP_PORT` | `8000` | Compose host port; container port stays 8000 |
| `APP_NAME` | `DocuMind` | Application name |
| `LLM_PROVIDER` | `groq` in example; code fallback `openai` | `groq` or `openai` |
| `EMBEDDING_PROVIDER` | `local` in example; code fallback `openai` | `local` or `openai` |
| `GROQ_API_KEY` | empty | Required for Groq chat |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Groq compatible endpoint |
| `GROQ_CHAT_MODEL` | `openai/gpt-oss-20b` | Documented production model with strict schema support |
| `GROQ_STRUCTURED_MODE` | `auto` | `auto`, `json_schema`, or `json_object`; JSON mode validates and retries once |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed CPU model |
| `EMBEDDING_CACHE_DIR` | `./data/documind-embedding-cache` locally | Compose overrides to a persistent `/app/documind-embedding-cache` volume |
| `LOCAL_EMBEDDING_THREADS` | `2` | CPU inference threads |
| `OPENAI_API_KEY` | empty | Required only when an OpenAI provider is selected |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat/structured-output model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model only |
| `EMBEDDING_DIMENSION` | `384` in example | Must match local model; OpenAI fallback is 1536 |
| `EMBEDDING_BATCH_SIZE` | `64` | Texts per ingestion embedding request |
| `QDRANT_URL` | `http://localhost:6333` | Vector database |
| `QDRANT_COLLECTION` | `documind_documents` | Cosine-distance collection |
| `REDIS_URL` | `redis://localhost:6379/0` | Conversation store |
| `DATABASE_URL` | `sqlite:///./data/documind.db` | SQLite database |
| `MAX_UPLOAD_BYTES` | `10485760` | Maximum accepted file bytes |
| `CHUNK_SIZE` | `800` | Default character limit |
| `CHUNK_OVERLAP` | `100` | Default overlap in characters |
| `HISTORY_TURNS` | `10` | Retained user/assistant exchanges |
| `MEMORY_TTL_SECONDS` | `86400` | History and partial-booking TTL |
| `RETRIEVAL_TOP_K` | `5` | Maximum retrieved chunks |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.58` | Minimum cosine score |
| `PROVIDER_TIMEOUT_SECONDS` | `60` | Provider/network timeout |
| `BOOKING_TIMEZONE` | `Asia/Kathmandu` in `.env.example` | IANA timezone for booking; code fallback is UTC |
| `BOOKING_CONFIRMATION_PHRASES` | See `Settings` | Optional JSON array of accepted affirmative phrases |

Local dimensions are read from FastEmbed's registry; an explicitly configured dimension must match. Qdrant checks dimensions before writes and searches. Changing embedding models requires a new collection and re-ingestion, even when dimensions happen to match. Existing collections are never silently deleted. Chunk size must be positive and overlap must be nonnegative and smaller than size.

## Provider choices

The example deployment uses Groq chat and local CPU embeddings, without requiring OpenAI billing. The `EmbeddingProvider` and `LLMProvider` interfaces and API schemas remain unchanged. OpenAI implementations remain selectable:

```dotenv
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_CHAT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
QDRANT_COLLECTION=documind_documents_openai
```

Set a funded `OPENAI_API_KEY` for that configuration. For Groq/local use `.env.example`, set `GROQ_API_KEY`, and choose a separate collection if switching from existing OpenAI vectors. Restart the app after changing provider settings.

Groq's [model list](https://console.groq.com/docs/models) and [structured-output documentation](https://console.groq.com/docs/structured-outputs) were checked on 2026-10-27. The chosen default supports strict JSON-schema mode. Auto mode uses JSON object mode for other model IDs and Pydantic validates the result with one retry on invalid output. The mode can also be selected explicitly. See [Groq's compatibility guide](https://console.groq.com/docs/openai) and [FastEmbed's model dimensions](https://qdrant.tech/documentation/embeddings/fastembed/).

## API examples

### Ingest and inspect a document

```sh
printf 'Plans: Basic and Plus. Plus includes email support.\n' > /tmp/plans.txt

curl -sS -X POST http://localhost:8000/api/v1/documents \
  -F 'file=@/tmp/plans.txt;type=text/plain' \
  -F 'chunking_strategy=recursive' \
  -F 'chunk_size=800' \
  -F 'chunk_overlap=100'

# Use the document_id from the 201 response:
curl -sS http://localhost:8000/api/v1/documents/DOCUMENT_UUID

# PDF alternative:
curl -sS -X POST http://localhost:8000/api/v1/documents \
  -F 'file=@tests/fixtures/handbook.pdf;type=application/pdf' \
  -F 'chunking_strategy=fixed_size'
```

A successful ingestion returns `document_id`, `filename`, `chunking_strategy`, and `chunk_count`. Metadata lookup returns the stored `id`, filename, content type, byte size, chunk settings/count, and creation timestamp.

TXT decoding tries UTF-8 with BOM support, then Latin-1. PDF extraction uses pypdf; image-only, empty, invalid, or unreadable PDFs are rejected. Original files are not retained.

### Ask a question and follow up

```sh
curl -sS http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"plans-demo","message":"What plans are available?"}'

curl -sS http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"plans-demo","message":"What about the second one?"}'
```

Every chat response contains `session_id`, `answer`, `sources`, and `booking`. Sources identify the document UUID, filename, and chunk index. Booking is null for ordinary answers.

### Full booking conversation

Use a fresh session and an available future slot. These examples use natural phrasing; the LLM normalizes it relative to the configured booking timezone.

```sh
curl -sS http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"booking-demo","message":"I would like to book an interview."}'

curl -sS http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"booking-demo","message":"My name is Ada Lovelace and my email is ada@example.com."}'

curl -sS http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"booking-demo","message":"Tomorrow at 2 PM."}'

# After receiving the complete summary and awaiting_confirmation:
curl -sS http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"booking-demo","message":"yes"}'
```

Expect `collecting`, then `awaiting_confirmation`, then `confirmed` with a booking UUID. Each booking object contains `status`, `missing_fields`, and `booking_id`. Reply `no` at confirmation or explicitly cancel to obtain `cancelled` and clear the partial state.

Email syntax, actual calendar dates strictly after today, and `HH:MM` times are validated in Python. Changes to the summary require reconfirmation. A SQLite unique constraint protects the exact date+time slot, including competing inserts. An occupied slot returns a clear chat response asking for another time/date; it does not create a second booking.

Redis uses `chat:{session_id}:history` for the bounded message list and `chat:{session_id}:booking` for partial-booking JSON. Each write refreshes that key's TTL. Session IDs separate conversation histories; document retrieval searches the shared collection.

## Chunking rationale

- **Fixed size:** predictable character windows with exact overlap, low overhead, and reliable coverage. Boundaries can cut sentences or words.
- **Recursive:** recursively partitions long text by paragraph, sentence, then word boundaries, and merges pieces up to the character limit. Long unbroken text falls back to character splits. Exact character overlap preserves continuity and can begin the next chunk mid-word.

Both strategies preserve all input characters, enforce the same limits, handle empty/tiny inputs, and implement the `Chunker` Protocol. Defaults live in `Settings`.

## Errors and persistence

Domain failures use `{"error":{"code":"...","message":"..."}}`. Invalid HTTP fields use FastAPI's standard 422 `detail` response.

- 400: unsupported file extension/MIME type, empty content, or extraction failure.
- 413: file exceeds the upload limit.
- 422: invalid form/JSON values or chunk overlap.
- 404: missing document metadata.
- 503: an unavailable dependency or failed ingestion.

Ingestion assigns one UUID to the document, embeds in batches, stores vectors, and then commits SQL metadata. Any failure after a write attempt triggers deletion of both document vectors and SQL metadata. Both cleanup operations are attempted independently. As with any compensating operation across separate stores, a store that remains unavailable can prevent deletion; this is logged with the document UUID and returned as a cleanup-failure 503 rather than being reported as successful rollback.

## Tests and checks

```sh
uv sync --frozen --python 3.11
uv run pytest
uv run ruff check .
uv run mypy --strict app
docker compose --env-file .env.example config --quiet
```

Create `.env` before the Compose check because the app service references it. The Python tests need no key, Docker daemon, or running external services.

Integration tests use FastAPI TestClient, real isolated SQLite, and in-memory fakes for OpenAI embeddings/chat, Qdrant, and Redis. Unit tests also exercise the production Qdrant adapter with the client's in-process memory backend and verify Redis key/TTL/transaction calls with a mocked client. No tests call external APIs.

## Requirement checklist

- [x] Required Python/FastAPI/Pydantic v2 stack, tooling and dependency configuration: `pyproject.toml`, `uv.lock`.
- [x] Required directory structure and real implementations: `app/`, `tests/unit/`, `tests/integration/`.
- [x] Application factory, lifespan-owned clients and clean shutdown: `app/main.py`.
- [x] Environment-only secrets/configuration and validated defaults: `app/core/config.py`, `.env.example`, `.gitignore`, `.dockerignore`.
- [x] Logging and centralized domain-to-HTTP error handling: `app/core/logging.py`, `app/core/exceptions.py`.
- [x] Thin routes, typed dependencies and constructor injection: `app/api/deps.py`, `app/api/routes/ingestion.py`, `app/api/routes/chat.py`.
- [x] Pydantic request/response, cross-layer and LLM-output models; enums: `app/schemas/ingestion.py`, `app/schemas/chat.py`, `app/schemas/booking.py`.
- [x] Multipart validation, size limits, PDF/TXT extraction and empty-text rejection: `app/services/ingestion_service.py`, `app/services/text_extraction.py`.
- [x] Both hand-written chunkers behind a common Protocol: `app/services/chunking.py`.
- [x] Selectable local/OpenAI embeddings behind a Protocol and configurable dimensions: `app/services/embedding.py`, `app/services/ingestion_service.py`.
- [x] Qdrant cosine collection creation, payloads, vector storage/search/deletion: `app/repositories/vector_repo.py`.
- [x] SQL document metadata, 201 ingestion response and 404 lookup: `app/models/document.py`, `app/repositories/document_repo.py`, `app/api/routes/ingestion.py`.
- [x] Cleanup of vector and SQL partial writes: `app/services/ingestion_service.py`; failure-path tests in `tests/integration/test_ingestion.py`.
- [x] Typed SQLAlchemy ORM and consistent synchronous SQLite sessions: `app/models/base.py`, `app/models/document.py`, `app/models/booking.py`, `app/db/session.py`.
- [x] Selectable Groq/OpenAI generation and validated structured outputs behind a Protocol: `app/services/llm.py`.
- [x] Explicit RAG steps, intent classification, active-booking routing and query rewriting: `app/services/rag_service.py`.
- [x] Top-k/score filtering, grounded prompts, sources and empty-context behavior: `app/services/rag_service.py`, `app/repositories/vector_repo.py`.
- [x] Redis bounded history, required key names, partial-state JSON and TTLs: `app/repositories/chat_memory_repo.py`.
- [x] Booking extraction/merge, EmailStr/date/time validation and targeted prompts: `app/services/booking_service.py`, `app/schemas/booking.py`.
- [x] Separate explicit confirmation, cancellation and clearing Redis state: `app/services/booking_service.py`.
- [x] SQL booking persistence and race-safe date+time uniqueness: `app/models/booking.py`, `app/repositories/booking_repo.py`.
- [x] Unit coverage for chunk boundaries/overlap, extraction, booking validation/merging, prompts and routing: `tests/unit/`.
- [x] Integration coverage for ingestion success/errors/rollback, follow-up chat and full booking flow: `tests/integration/`.
- [x] Full type annotations and public docstrings; strict mypy/Ruff configuration without suppression directives: `app/`, `pyproject.toml`.
- [x] No prohibited libraries, prebuilt chains, frontend or extra product features: `pyproject.toml`, `app/`.
- [x] App, Qdrant and Redis container configuration and persistent storage: `Dockerfile`, `docker-compose.yml`.
- [x] Setup, architecture, API examples, chunking rationale and check commands: this README.
- [x] Ambiguous implementation choices recorded one per line: `DECISIONS.md`.

## Verification status

The deterministic suite contains 134 tests. Provider calls and Redis are replaced
with fakes or mocks; SQLite persistence and the in-process Qdrant adapter are
exercised locally. These checks do not establish live model accuracy.

The public demo and retrieval utility use only
[the synthetic handbook PDF](tests/fixtures/handbook.pdf).
See [the current handbook verification report](LIVE_VERIFICATION.md) for what was
actually rerun and its limits. Older private evidence is excluded from Git and
is not required to run this project. Other retained files in `verification/`
are historical development records, not results from the current handbook run.

After starting the stack, run:

```sh
BASE_URL=http://localhost:8000 bash verification/demo.sh
# Use the host port configured through APP_PORT if different.
docker compose exec -T documind-app python < verification/measure_retrieval.py
```

The demo uploads the handbook with both strategies and creates a synthetic
booking. Repeated runs add documents and may encounter an occupied booking slot;
select another future date with `BOOKING_DATE` if needed. Broad questions can
miss evidence at the configured threshold, and provider rate limits can cause
request failures.

## Possible future work

- Further prompt-injection hardening of retrieved document text.

## Deployment naming

The distribution is `documind-backend`; the import package remains `app`.
Compose uses project `documind`, image `documind-backend:latest`, and services/containers
`documind-app`, `documind-qdrant`, and `documind-redis`. Its network is
`documind_default`; persistent volumes are `documind_sql_data`,
`documind_vector_data`, `documind_redis_data`, and `documind_embedding_cache`.
The default collection is `documind_documents`, SQLite file is `documind.db`,
and demo sessions begin with `documind-demo-`. Host ports and API paths are unchanged.

The renamed stores start empty; existing stores are not migrated or deleted.
Private local documents and evidence are ignored by Git and excluded from Docker
builds. Public demonstration data comes from the synthetic handbook fixtures.

```sh
docker compose logs --tail=20 documind-app
docker compose exec -T documind-app python < verification/measure_retrieval.py
docker compose exec -T documind-redis redis-cli ping
```
