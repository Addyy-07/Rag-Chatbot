# Architecture Decision Records — DocChat AI

This document records the key architectural decisions made during the production
refactor of the RAG Chatbot, explaining the *why* behind each choice.

---

## ADR-001: Pydantic BaseSettings for Configuration

**Decision:** Use `pydantic-settings` `BaseSettings` instead of raw `os.getenv()` calls.

**Context:** The original codebase called `os.getenv("PINECONE_API_KEY")` in multiple
places without defaults, types, or validation. A missing key caused cryptic runtime errors
far from where the variable was read.

**Consequences:**
- ✅ All env vars are validated at startup — fail fast, fail clearly.
- ✅ Types are enforced (`float`, `int`, `str`) — no silent type coercion bugs.
- ✅ Single source of truth — add a new setting in one place.
- ✅ `lru_cache` singleton pattern ensures Settings() is constructed once.

---

## ADR-002: Service Layer Pattern

**Decision:** Split business logic into four focused services:
`EmbeddingService`, `VectorStoreService`, `IngestionService`, `ChatService`.

**Context:** The original `app.py` mixed PDF loading, embedding, Pinecone management,
LLM calls, and UI rendering in functions like `ingest_pdf()` and `get_answer()`.

**Consequences:**
- ✅ Each service is independently testable with mocks.
- ✅ Services have zero Streamlit dependency — reusable in FastAPI or CLI.
- ✅ Single Responsibility: adding a new LLM provider only changes ChatService.
- ⚠️ More files to navigate (tradeoff accepted for maintainability).

---

## ADR-003: Routes as Thin Orchestration Layer

**Decision:** Add a `routes/` layer between UI and services.

**Context:** When we eventually add a FastAPI backend, we need a seam where
"Streamlit-aware" code ends and "business logic" begins.

**Consequences:**
- ✅ Input validation (empty question check) lives in the router.
- ✅ Streamlit type conversion (`UploadedFile` → bytes → path) is isolated.
- ✅ Future REST endpoints reuse the same services by swapping routers.

---

## ADR-004: Prompts as Module-Level Constants

**Decision:** Extract all prompt templates to `backend/prompts/rag_prompt.py`.

**Context:** The original prompt was a string literal inside `get_answer()`, making
it impossible to version, test, or reuse independently.

**Consequences:**
- ✅ Prompts are visible as a product artifact, not buried in code.
- ✅ A/B testing prompts requires no service changes.
- ✅ The `QUESTION_REFORMULATION_PROMPT` is scaffolded for future use.

---

## ADR-005: Context Manager for Temp Files

**Decision:** Use a `@contextmanager` `temp_pdf_path()` instead of manual
`NamedTemporaryFile` + `os.unlink()`.

**Context:** The original code did `os.unlink(tmp_path)` at the end of
`ingest_pdf()`. If an exception occurred before that line, the temp file leaked.

**Consequences:**
- ✅ Temp files are always deleted, even on exceptions (`finally` block).
- ✅ File handling concern is isolated from ingestion logic.

---

## ADR-006: Structured Logging over print()

**Decision:** Replace all `print()` calls with Python's `logging` module.

**Context:** `print()` is not thread-safe, has no levels, and doesn't integrate
with observability tools (Datadog, AWS CloudWatch, etc.).

**Consequences:**
- ✅ Log levels (DEBUG/INFO/WARNING/ERROR) allow filtering in production.
- ✅ Timestamps and module names are included automatically.
- ✅ Noisy third-party loggers (httpx, pinecone) are suppressed.

---

## ADR-007: Multi-Stage Docker Build

**Decision:** Use a two-stage Dockerfile (builder → runtime).

**Context:** A single-stage image would include build tools (`gcc`, `build-essential`)
in the final image, adding ~200MB and increasing attack surface.

**Consequences:**
- ✅ Runtime image contains only what's needed to run the app.
- ✅ Non-root user (`appuser`) reduces container privilege.
- ✅ Health check endpoint enables orchestrator (k8s/ECS) liveness probes.

---

## Future Architecture Considerations

### REST API (FastAPI)
When adding a FastAPI backend:
1. Add `backend/api/` with FastAPI routers.
2. Reuse all existing services — they have no Streamlit dependency.
3. Move `@st.cache_resource` to FastAPI lifespan events.

### Multi-tenancy / SaaS
When adding user namespaces:
1. Pass a `namespace` parameter through routers → services → Pinecone.
2. Pinecone supports namespacing natively.
3. Settings can be extended with per-tenant overrides.

### Async Support
When scaling to concurrent users:
1. Swap `langchain_groq.ChatGroq` for its async variant.
2. Use `asyncio` in services.
3. FastAPI handles async natively.
