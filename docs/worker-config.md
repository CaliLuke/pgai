# Worker Configuration

## CLI Arguments

```text
worker [OPTIONS]
```

| Flag               | Env Var  | Default    | Description                    |
| ------------------ | -------- | ---------- | ------------------------------ |
| `--db-url`         | `DB_URL` | (required) | PostgreSQL connection string   |
| `--poll-interval`  | —        | 60         | Seconds between poll cycles    |
| `--once`           | —        | false      | Run one cycle and exit         |
| `--exit-on-error`  | —        | false      | Exit on first error            |
| `--vectorizer-ids` | —        | (all)      | Comma-separated IDs to process |

## Environment Variables

| Variable                      | Purpose                                                                      |
| ----------------------------- | ---------------------------------------------------------------------------- |
| `DB_URL`                      | PostgreSQL connection string                                                 |
| `OPENAI_API_KEY`              | OpenAI API key (fallback: DB secret)                                         |
| `OLLAMA_HOST`                 | Ollama base URL (fallback: config `base_url`, then `http://localhost:11434`) |
| `OLLAMA_MAX_CHUNKS_PER_BATCH` | Max chunks per Ollama batch (default from config)                            |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Enable OpenTelemetry tracing to this endpoint                                |
| `RUST_LOG`                    | Log level filter (default: `info`)                                           |

## Running

```bash
# Basic
DB_URL=postgres://user:pass@localhost:5432/mydb cargo run -p worker

# With options
cargo run -p worker -- \
    --db-url postgres://user:pass@localhost:5432/mydb \
    --poll-interval 10 \
    --vectorizer-ids 1,2

# Release build
cargo build --release -p worker
./target/release/worker --db-url postgres://... --poll-interval 10
```

## Startup Sequence

1. Load `.env` file (dotenvy)
2. Init tracing (fmt layer + optional OTLP)
3. Parse CLI args
4. Spawn signal handler (SIGINT/SIGTERM → graceful shutdown)
5. Connect to database
6. Start heartbeat background task
7. Enter poll loop:
   - Check pgai version (`pg_extension` or `ai.pgai_lib_version`)
   - Fetch vectorizer IDs from `ai.vectorizer`
   - For each non-disabled vectorizer: create Executor, run pipeline
   - Sleep `poll_interval`, repeat

## Concurrency

Each vectorizer's `processing.concurrency` (1-10) controls how many executor tasks run in parallel via `tokio::task::JoinSet`. Each executor independently polls the queue with `FOR UPDATE SKIP LOCKED`.

## Graceful Shutdown

On SIGTERM/SIGINT:

1. CancellationToken triggered
2. Current batch completes (no new batches started)
3. Final heartbeat sent
4. OTLP tracer flushed
5. Process exits cleanly
