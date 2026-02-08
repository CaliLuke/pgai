# Architecture

## Workspace Layout

```text
pgai/
├── Cargo.toml                 # Workspace root
├── worker/                    # Vectorizer worker binary
├── extension/                 # PostgreSQL extension (pgrx)
├── text-splitter/             # Text chunking library
├── scripts/                   # Load test tooling
└── plans/                     # Coordination docs (temporary)
```

## Crate Dependencies

```text
worker ──depends──> pgai-text-splitter
worker ──depends──> async-openai, ollama-rs, sqlx, tokio, tiktoken-rs
extension ──depends──> pgrx, reqwest, serde_json
```

The worker and extension are independent — the worker talks to Postgres over the network, the extension runs inside the Postgres process.

## Data Flow

```text
┌─────────────┐    trigger    ┌──────────────┐    poll     ┌──────────┐
│ Source Table │──────────────>│ Queue Table  │<───────────│  Worker  │
│ (user data) │               │ (ai schema)  │            │ (Rust)   │
└─────────────┘               └──────────────┘            └────┬─────┘
                                                               │
                                                   chunk → format → embed
                                                               │
                                                          ┌────▼──────────┐
                                                          │ Target Table  │
                                                          │ (embeddings)  │
                                                          └───────────────┘
```

1. `ai.create_vectorizer()` sets up: target table, queue table, triggers, metadata row
2. Triggers on source table insert PKs into queue on INSERT/UPDATE/DELETE
3. Worker polls queue with `FOR UPDATE SKIP LOCKED` (concurrent-safe)
4. For each batch: fetch source rows → chunk text → format → call embedding API → write to target table
5. Delete processed PKs from queue

## Worker (`worker/`)

### Modules

| Module               | Purpose                                                                    |
| -------------------- | -------------------------------------------------------------------------- |
| `main.rs`            | CLI (clap), signal handling, telemetry init                                |
| `worker.rs`          | Main loop: poll interval, version check, dispatch vectorizers              |
| `executor.rs`        | Pipeline: fetch_work → chunk → format → embed_with_retry → write_results   |
| `embedder.rs`        | `Embedder` trait + OpenAI/Ollama implementations, batching, token counting |
| `errors.rs`          | `EmbeddingError` enum (Transient/Permanent), pattern-based classification  |
| `models.rs`          | Serde types: Vectorizer, Config, EmbeddingConfig, ChunkerConfig, etc.      |
| `worker_tracking.rs` | Background heartbeat task, atomic success/error counters                   |

### Vectorizer Fetch

Worker loads vectorizers via:

```sql
SELECT to_jsonb(v) AS vectorizer FROM ai.vectorizer v WHERE v.id = $1
```

Deserialized into `Vectorizer` struct. Unknown JSON fields (e.g. `trigger_name`, `name`) silently ignored by serde.

### Queue Processing

```sql
WITH selected_rows AS (
    SELECT {pk_cols} FROM queue_table LIMIT $batch_size FOR UPDATE SKIP LOCKED
),
locked_items AS (
    SELECT {pk_cols}, pg_try_advisory_xact_lock(...) AS locked FROM ...
),
deleted_rows AS (
    DELETE FROM queue_table USING locked_items WHERE locked = true ...
)
SELECT to_jsonb(s) FROM locked_items JOIN source_table ...
```

Advisory locks prevent duplicate processing across concurrent executors.

### Error Classification

Embedding errors classified by string pattern matching:

- **Permanent** (no retry): 401, 403, 400, "invalid api key", "model not found", "billing", "quota exceeded"
- **Transient** (retry with backoff): 429, 500-504, "timeout", "connection", "rate limit", "temporarily unavailable"
- **Default**: unknown errors treated as transient

Retry: exponential backoff, 1-10s delay, max 3 attempts.

### Embedding Providers

**OpenAI** (`async-openai`):

- tiktoken token counting per chunk
- 300K token batch limit, auto-batching
- 8191 context length truncation
- API key from env `OPENAI_API_KEY` or DB secret via `ai.reveal_secret()`
- Configurable `base_url` for compatible APIs

**Ollama** (`ollama-rs`):

- Base URL from `OLLAMA_HOST` env or config `base_url` (default: `http://localhost:11434`)
- No token counting, simple pass-through
- Configurable `max_chunks_per_batch` via `OLLAMA_MAX_CHUNKS_PER_BATCH` env

### Worker Tracking

Background tokio task sends heartbeat every `poll_interval`:

- Calls `ai._worker_heartbeat(worker_id, version, heartbeat_at)`
- Reports per-vectorizer progress via `ai._worker_progress(worker_id, vectorizer_id, successes, errors, last_error)`
- Uses atomic counters (lock-free) accumulated between heartbeats

## Extension (`extension/`)

### Rust Functions (pgrx)

| Function            | Signature                                                 |
| ------------------- | --------------------------------------------------------- |
| `openai_embed`      | `(model, input, api_key?, dimensions?) -> Vec<f32>`       |
| `ollama_embed`      | `(model, input, base_url?) -> Vec<f32>`                   |
| `create_vectorizer` | `(source_table, config) -> i32` (legacy, see SQL version) |

GUC: `ai.openai_api_key` registered at `_PG_init`.

### SQL Infrastructure (`extension/sql/setup.sql`)

Loaded at `CREATE EXTENSION` time via `extension_sql_file!` macro.

**Schema objects:**

- `ai.vectorizer` table — metadata (id, source, queue, config, trigger_name, name, disabled)
- `ai.vectorizer_errors` table — error log (id FK, message, details, recorded_at)
- `ai.vectorizer_id_seq` — ID sequence

**Config helpers** (all return JSONB):

- `ai.loading_column(column_name, retries)`
- `ai.embedding_ollama(model, dimensions, base_url, options, keep_alive)`
- `ai.embedding_openai(model, dimensions, chat_user, api_key_name, base_url)`
- `ai.chunking_none()`
- `ai.chunking_recursive_character_text_splitter(chunk_size, chunk_overlap, separators, is_separator_regex)`
- `ai.chunking_character_text_splitter(chunk_size, chunk_overlap, separator, is_separator_regex)`
- `ai.processing_default(batch_size, concurrency)`
- `ai.destination_table(target_schema, target_table, view_schema, view_name)`
- `ai.formatting_python_template(template)`
- `ai.formatting_chunk_value()`

**Internal functions:**

- `ai._vectorizer_source_pk(regclass)` — extract PK metadata
- `ai._vectorizer_create_target_table(...)` — CREATE TABLE with PK cols + chunk_seq + chunk + embedding
- `ai._vectorizer_create_view(...)` — CREATE VIEW joining source and target
- `ai._vectorizer_create_queue_table(...)` — queue table with PK cols + queued_at
- `ai._vectorizer_create_queue_failed_table(...)` — failed queue table
- `ai._vectorizer_build_trigger_definition(...)` — returns trigger function SQL
- `ai._vectorizer_create_source_trigger(...)` — creates row + truncate triggers

**Public API:**

- `ai.create_vectorizer(source, loading, embedding, chunking, formatting, processing, destination, queue_schema, queue_table, enqueue_existing, name, if_not_exists)` — full orchestrator
- `ai.drop_vectorizer(vectorizer_id, drop_all)` — cleanup
- `ai.vectorizer_queue_pending(vectorizer_id)` — queue depth

## Text Splitter (`text-splitter/`)

Pure Rust library, no async, single dependency (`regex`).

### Types

| Type                             | Description                                                             |
| -------------------------------- | ----------------------------------------------------------------------- |
| `CharacterTextSplitter`          | Split on single separator, merge into chunks. LangChain port.           |
| `RecursiveCharacterTextSplitter` | Try separators in order (`\n\n`, `\n`, ` `, ``), merge. LangChain port. |
| `SentenceChunker`                | Split into sentences, greedily pack. Chonkie-inspired.                  |
| `SemchunkSplitter`               | Fixed 8-level delimiter hierarchy, recursive. semchunk-inspired.        |

### Common Fields

All splitters share:

- `chunk_size: usize`
- `chunk_overlap: usize`
- `strip_whitespace: bool` (default true)
- `length_fn: Option<Box<dyn Fn(&str) -> usize + Send + Sync>>` (default: char count)

### SemchunkSplitter Hierarchy

```text
1. Newlines     (\n sequences, longest first)
2. Tabs         (\t sequences, longest first)
3. Sentence terminators  (". ", "? ", "! ")
4. Clause separators     ("; ", ", ", ") ", "] ")
5. Sentence interrupters (": ", "-- ", "... ")
6. Whitespace   (space sequences, longest first)
7. Word joiners ("/", "\\", "&", "-")
8. Characters   (individual chars, final fallback)
```

Non-whitespace delimiters are reattached to the preceding chunk.

### Internal Modules

- `merge.rs` — `merge_splits(splits, separator, chunk_size, chunk_overlap, strip, length_fn)` — core merging logic
- `split.rs` — `split_text_with_regex(text, separator, keep_separator)` — regex/literal splitting
- `char_len(s)` — default length function (Unicode char count)
