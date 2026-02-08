# Integrating pgai-rs into Auto-K

## Current Auto-K Setup

Auto-K currently uses the **Python pgai** package for vectorizer infrastructure:

- **Docker Compose** (`docker-compose.yml`):
  - `pgvector/pgvector:pg18` — Postgres with pgvector
  - `timescale/pgai-vectorizer-worker:latest` — Python vectorizer worker (profiles: embeddings)
  - `ollama/ollama:latest` — Embedding model server (profiles: embeddings)
  - Worker env: `PGAI_VECTORIZER_WORKER_DB_URL=postgres://admin:password@postgres:5432/auto_k`

- **Migration** (`src/migrations/postgres/versions/20260129_1500_a7f2c9e1b3d4_add_versioning_tables.py`):
  - Calls `pgai.install(sync_url)` (Python library) to create `ai` schema + functions
  - Then runs `ai.create_vectorizer(...)` SQL

- **Test file** (`src/tests/integration/versioning/test_pgai_setup.py`):
  - Also calls `pgai.install(sync_url)` in fixtures

## What to Change

### 1. Migration: Replace `pgai.install()` with `setup.sql`

The file `extension/sql/setup.sql` (870 lines) contains all the SQL that `pgai.install()` runs. It creates:

- `ai` schema, `ai.vectorizer` table, `ai.vectorizer_errors` table
- Config helpers: `ai.loading_column()`, `ai.embedding_ollama()`, `ai.embedding_openai()`, `ai.chunking_none()`, `ai.chunking_recursive_character_text_splitter()`, `ai.formatting_python_template()`, `ai.formatting_chunk_value()`, `ai.processing_default()`, `ai.destination_table()`
- Internal functions: `ai._vectorizer_source_pk()`, `ai._vectorizer_create_target_table()`, etc.
- `ai.create_vectorizer()`, `ai.drop_vectorizer()`, `ai.vectorizer_queue_pending()`

**In `_install_pgai()`**, replace:

```python
import pgai
pgai.install(sync_url)
```

With reading and executing `setup.sql`:

```python
from pathlib import Path
# Adjust path to wherever setup.sql lives relative to auto-k-server
setup_sql = Path(__file__).parent.parent.parent.parent.parent / "vendor" / "pgai" / "setup.sql"
# Or copy setup.sql into the migrations directory
op.execute(setup_sql.read_text())
```

**The `_create_vectorizer()` function stays exactly the same** — all SQL function signatures are identical.

### 2. Docker Compose: Replace Python Worker with Rust Worker

The Rust worker binary uses different env vars and CLI args:

| Python worker                   | Rust worker                               |
| ------------------------------- | ----------------------------------------- |
| `PGAI_VECTORIZER_WORKER_DB_URL` | `DB_URL` (or `--db-url`)                  |
| `--poll-interval 10s`           | `--poll-interval 10` (seconds as integer) |

**Option A — Run locally (development):**

```bash
cd /Users/luca/code/pgai
cargo run -p worker -- --db-url postgres://admin:password@localhost:8002/auto_k --poll-interval 10
```

**Option B — Docker Compose with build from source:**

```yaml
vectorizer-worker:
  build:
    context: /path/to/pgai
    dockerfile: Dockerfile.worker # needs to be created
  command: ["--poll-interval", "10"]
  environment:
    DB_URL: postgres://admin:password@postgres:5432/auto_k
    OLLAMA_HOST: http://ollama:11434
  depends_on:
    - postgres
    - ollama
  profiles:
    - embeddings
```

You'd need a `Dockerfile.worker`:

```dockerfile
FROM rust:1.83-slim AS builder
WORKDIR /app
COPY . .
RUN cargo build --release -p worker

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/worker /usr/local/bin/pgai-worker
ENTRYPOINT ["pgai-worker"]
```

**Option C — Just run the binary directly (simplest for now):**

Skip Docker for the worker entirely. Build once with `cargo build --release -p worker`, then run:

```bash
./target/release/worker --db-url postgres://admin:password@localhost:8002/auto_k --poll-interval 10
```

### 3. Test File: Same `pgai.install()` Replacement

In `test_pgai_setup.py`, replace all `pgai.install(sync_url)` calls with executing setup.sql. Or create a shared helper that both migration and tests use.

### 4. Remove Python `pgai` Dependency

Once setup.sql is used directly, the `pgai` Python package is no longer needed. Remove it from `pyproject.toml` / `requirements.txt`.

## Worker Compatibility Notes

The Rust worker is fully compatible with the SQL schema created by `setup.sql`:

- Fetches vectorizers via `SELECT to_jsonb(v) FROM ai.vectorizer v`
- Extra columns (`trigger_name`, `name`) are silently ignored by serde
- Config JSON shapes match exactly (extra `config_type` fields ignored)
- Queue table uses `FOR UPDATE SKIP LOCKED` pattern — same as Python worker
- Error recording writes to `ai.vectorizer_errors` — same table

## Embedding Provider Note

Auto-K uses `ai.embedding_ollama('embeddinggemma:300m', 768)`. The Rust worker's Ollama embedder reads `OLLAMA_HOST` env var for the base URL, or falls back to the `base_url` in the vectorizer config. Make sure either:

- `OLLAMA_HOST` is set when running the worker, OR
- The `base_url` parameter in `ai.embedding_ollama(...)` points to the correct Ollama instance

## File Locations

| File                                            | Purpose                                                       |
| ----------------------------------------------- | ------------------------------------------------------------- |
| `/Users/luca/code/pgai/extension/sql/setup.sql` | SQL infrastructure (copy or reference from autok)             |
| `/Users/luca/code/pgai/worker/`                 | Rust worker crate                                             |
| `/Users/luca/code/pgai/target/release/worker`   | Built worker binary (after `cargo build --release -p worker`) |
