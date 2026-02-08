# Setup Guide

## Prerequisites

- PostgreSQL 13–18
- [pgvector](https://github.com/pgvector/pgvector) extension installed
- Rust toolchain (stable) — for building from source
- An embedding provider: [Ollama](https://ollama.ai) (local) or an OpenAI API key

## Installing the Extension

Build and install via pgrx:

```bash
cargo pgrx install -p extension --release
```

Then in your database:

```sql
CREATE EXTENSION ai CASCADE;
```

The `CASCADE` installs the `vector` dependency automatically (declared in `extension.control`).

If you're loading `setup.sql` manually (without pgrx), execute it as a single script against your database:

```bash
psql -f extension/sql/setup.sql
```

### Loading setup.sql from Python / SQLAlchemy

The SQL contains PL/pgSQL `format()` specifiers (`%I`, `%L`, `%s`) inside function bodies. These conflict with Python DB-API `%`-style parameter markers, so SQLAlchemy's `text()` will misparse them.

Use a raw DBAPI cursor instead:

```python
# SQLAlchemy 2.x
with engine.connect() as conn:
    conn.connection.cursor().execute(open("setup.sql").read())
    conn.connection.commit()
```

This bypasses the parameter-binding layer entirely.

## Creating a Vectorizer

A vectorizer connects a source table to an embedding pipeline. At minimum you need a source table with a primary key and a text column:

```sql
CREATE TABLE public.documents (
    id serial PRIMARY KEY,
    title text NOT NULL,
    body text NOT NULL
);
```

Then create the vectorizer:

```sql
SELECT ai.create_vectorizer(
    'public.documents'::regclass,
    loading   => ai.loading_column('body'),
    embedding => ai.embedding_ollama('nomic-embed-text', 768),
    chunking  => ai.chunking_recursive_character_text_splitter(
        chunk_size => 500, chunk_overlap => 50
    )
);
```

This creates:

- `public.documents_embedding_store` — target table with vector column
- `public.documents_embedding` — view joining source and target
- `ai._vectorizer_q_{id}` — work queue
- Triggers on the source table that enqueue rows on INSERT/UPDATE/DELETE

See [sql-reference.md](sql-reference.md) for full parameter docs.

### Idempotency

The `if_not_exists` parameter controls behavior when a vectorizer with the same name already exists:

```sql
SELECT ai.create_vectorizer(
    'public.documents'::regclass,
    loading      => ai.loading_column('body'),
    embedding    => ai.embedding_ollama('nomic-embed-text', 768),
    if_not_exists => true
);
```

With `if_not_exists => true`, a duplicate name returns the existing vectorizer ID instead of raising an error.

Note: the underlying tables (embedding store, queue) use `IF NOT EXISTS`, so re-creating after a partial failure is safe. However, if you drop the `ai` schema while leaving embedding store tables in `public`, you'll need to clean those up manually before re-creating:

```sql
DROP TABLE IF EXISTS public.documents_embedding_store CASCADE;
DROP VIEW IF EXISTS public.documents_embedding;
```

## Running the Worker

```bash
# Build
cargo build --release -p worker

# Run
DB_URL=postgres://user:pass@localhost:5432/mydb \
  ./target/release/worker --poll-interval 10
```

The worker polls `ai.vectorizer` for active vectorizers, fetches queued rows, generates embeddings, and writes results to the target tables.

### Key flags

| Flag                  | Default    | Description                    |
| --------------------- | ---------- | ------------------------------ |
| `--db-url` / `DB_URL` | (required) | PostgreSQL connection string   |
| `--poll-interval`     | 60         | Seconds between poll cycles    |
| `--once`              | false      | Run one cycle and exit         |
| `--exit-on-error`     | false      | Exit on first error            |
| `--vectorizer-ids`    | (all)      | Comma-separated IDs to process |

### Embedding providers

**Ollama** (default local):

```bash
OLLAMA_HOST=http://localhost:11434 \
DB_URL=postgres://... \
  ./target/release/worker --poll-interval 10
```

**OpenAI**:

```bash
OPENAI_API_KEY=sk-... \
DB_URL=postgres://... \
  ./target/release/worker --poll-interval 10
```

See [worker-config.md](worker-config.md) for all environment variables.

## Monitoring

### Queue depth

```sql
SELECT ai.vectorizer_queue_pending(1);
```

### Worker health

If worker tracking tables are present (they're created by `setup.sql`), you can monitor worker processes and per-vectorizer progress:

```sql
-- Active workers (heartbeat within expected interval)
SELECT id, version, last_heartbeat, success_count, error_count
FROM ai.vectorizer_worker_process
WHERE last_heartbeat > now() - expected_heartbeat_interval * 2;

-- Per-vectorizer throughput
SELECT v.id, v.source_table, p.success_count, p.error_count,
       p.last_error_message, p.updated_at
FROM ai.vectorizer v
LEFT JOIN ai.vectorizer_worker_progress p ON p.vectorizer_id = v.id
ORDER BY v.id;
```

See [worker-tracking.md](worker-tracking.md) for full details.

## Troubleshooting

### Stale vectorizers

If a vectorizer's queue table was manually dropped or the vectorizer references a deleted source table, the worker will log an error and skip it. To clean up stale vectorizers:

```sql
-- Find vectorizers whose queue tables no longer exist
SELECT v.id, v.source_table, v.queue_schema, v.queue_table
FROM ai.vectorizer v
WHERE to_regclass(format('%I.%I', v.queue_schema, v.queue_table)) IS NULL;

-- Remove a stale vectorizer (drop_all=false since objects are already gone)
SELECT ai.drop_vectorizer(24, drop_all => false);
```

### Cleanup old worker rows

Worker rows accumulate over restarts. Periodically clean up:

```sql
DELETE FROM ai.vectorizer_worker_process
WHERE last_heartbeat < now() - interval '7 days';
```

### Extension rebuild / schema re-install

If you need to drop and recreate the `ai` schema, the embedding store tables (in `public`) and their data survive. Before re-running `create_vectorizer`:

1. Either drop the orphaned tables: `DROP TABLE IF EXISTS public.documents_embedding_store CASCADE;`
2. Or use `if_not_exists => true` — the function will detect the existing vectorizer name and return its ID.

### Worker can't find pgai

If the worker logs `pgai is not installed in the database`, verify:

```sql
SELECT extname, extversion FROM pg_extension WHERE extname = 'ai';
```

If the extension isn't installed, run `CREATE EXTENSION ai CASCADE;`.
