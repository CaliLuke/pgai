# Worker Logging Improvements

## Current State

The worker uses `tracing` with `tracing_subscriber::fmt` and optional OTLP export.
Log level is controlled via `RUST_LOG` env var (default: `info`).

## Suggestions

### 1. Structured JSON output mode

Add a `--log-format` flag (`text` | `json`). When embedded in auto-k-server,
JSON output is easier to parse and route to SQLite / structured log sinks.

Currently auto-k-server's `_vectorizer_log_reader()` parses the fmt output
with a regex — fragile and loses structured fields. JSON mode would give us
`level`, `message`, `target`, `span`, `fields` cleanly.

```rust
// In init_telemetry(), switch based on flag:
if json_mode {
    let fmt_layer = tracing_subscriber::fmt::layer().json();
    // ...
} else {
    let fmt_layer = tracing_subscriber::fmt::layer();
    // ...
}
```

### 2. Per-vectorizer span context

`process_vectorizer` already has `#[tracing::instrument(fields(vectorizer_id))]`
but `Executor::do_batch` logs generic messages like "Fetched N items". Adding
the source table name to the span would make logs scannable:

```rust
#[tracing::instrument(skip(pool, cancel, tracking),
    fields(vectorizer_id = vectorizer.id, source = %vectorizer.source_table))]
```

### 3. Batch progress logging

`Executor::run` loops over batches but only logs when fetching. Adding a
running total would help monitor long backfills:

```rust
info!(
    vectorizer_id = self.vectorizer.id,
    batch_items = items.len(),
    total_processed = total_processed,
    "Batch complete"
);
```

### 4. Startup summary

Log the full configuration at startup (vectorizer count, poll interval,
embedding provider, etc.) so you can diagnose misconfiguration from logs alone:

```rust
info!(
    vectorizer_count = vectorizer_ids.len(),
    poll_interval_secs = self.poll_interval.as_secs(),
    "Starting poll cycle"
);
```

### 5. Heartbeat visibility

The heartbeat loop logs at `debug` level. Consider `info` for the first
heartbeat and `debug` for subsequent ones, so operators can confirm tracking
is working without cranking up verbosity.

### 6. Error classification in logs

When `embed_with_retry` fails, log whether the error was transient (all retries
exhausted) vs permanent (no retry attempted). This helps distinguish "Ollama is
down" from "model doesn't exist":

```rust
if e.is_transient() {
    error!("Transient embedding error after retries: {e}");
} else {
    error!("Permanent embedding error (no retry): {e}");
}
```

## Priority

Items 1 (JSON mode) and 3 (batch progress) have the highest impact for
auto-k-server integration. The rest are nice-to-haves.
