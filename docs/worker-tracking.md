# Worker Tracking

Worker tracking gives visibility into worker health and per-vectorizer throughput. The worker registers itself on startup, sends periodic heartbeats, and reports progress after each embedding batch.

## How It Works

1. **Feature detection:** On startup, the worker checks `information_schema.tables` for `ai.vectorizer_worker_process`. If the table doesn't exist, tracking is disabled entirely — the worker runs normally but without health reporting.

2. **Registration:** The worker calls `ai._worker_start(version, poll_interval)` and gets back a UUID. This creates a row in `ai.vectorizer_worker_process`.

3. **Heartbeat loop:** A background task fires every `poll_interval`, calling `ai._worker_heartbeat()` with the success/error deltas accumulated since the last heartbeat. Uses atomic counters (lock-free) that are swapped to zero each tick.

4. **Progress reporting:** After each embedding batch, the worker calls `ai._worker_progress()` for the vectorizer it just processed. This upserts a row in `ai.vectorizer_worker_progress`.

5. **Shutdown:** The worker sends one final heartbeat with any remaining counters, then exits.

If the heartbeat fails 3 consecutive times, the heartbeat loop stops (the worker keeps running, just without health reporting).

## Tables

### ai.vectorizer_worker_process

One row per worker process (alive or dead). Rows accumulate over time — cleanup is left to operators.

| Column                      | Type        | Description                                      |
| --------------------------- | ----------- | ------------------------------------------------ |
| id                          | uuid PK     | Auto-generated                                   |
| version                     | text        | Worker version string                            |
| started                     | timestamptz | When the worker registered                       |
| expected_heartbeat_interval | interval    | How often the worker promises to heartbeat       |
| last_heartbeat              | timestamptz | Last successful heartbeat time                   |
| heartbeat_count             | bigint      | Total heartbeats sent                            |
| success_count               | bigint      | Lifetime total successful embeddings             |
| error_count                 | bigint      | Lifetime total errors                            |
| last_error_at               | timestamptz | When the last error occurred (NULL if no errors) |
| last_error_message          | text        | Most recent error message (NULL if no errors)    |

Indexed on `last_heartbeat` for monitoring queries.

### ai.vectorizer_worker_progress

One row per vectorizer, shared across all workers. Tracks which worker last processed each vectorizer.

| Column                  | Type        | Description                                     |
| ----------------------- | ----------- | ----------------------------------------------- |
| vectorizer_id           | int4 PK     | FK to `ai.vectorizer(id)` ON DELETE CASCADE     |
| success_count           | bigint      | Lifetime total successful embeddings            |
| error_count             | bigint      | Lifetime total errors                           |
| last_success_at         | timestamptz | When the last success occurred                  |
| last_success_process_id | uuid        | Which worker last succeeded (no FK — see below) |
| last_error_at           | timestamptz | When the last error occurred                    |
| last_error_message      | text        | Most recent error message                       |
| last_error_process_id   | uuid        | Which worker last errored (no FK — see below)   |
| updated_at              | timestamptz | When this row was last modified                 |

The `process_id` columns intentionally have **no foreign key** to `vectorizer_worker_process`:

- The process table may be cleaned up independently of progress data.
- We don't want FK checks slowing down the hot path.
- We don't want progress inserts to fail because a process row was deleted.

## Functions

All functions use `clock_timestamp()` (not `now()`) so timestamps reflect actual wall time, not transaction start time. All are `SECURITY INVOKER` with `search_path = pg_catalog, pg_temp`.

### ai.\_worker_start(version, expected_heartbeat_interval)

Registers a worker. Returns a UUID to use in subsequent calls.

```sql
SELECT ai._worker_start('0.1.0', interval '30 seconds');
-- Returns: a0b1c2d3-...
```

### ai.\_worker_heartbeat(worker_id, successes, errors, error_message)

Periodic health signal. Increments `heartbeat_count`, accumulates success/error deltas into lifetime totals. If `error_message` is non-NULL, updates `last_error_at` and `last_error_message`; otherwise preserves the previous error.

```sql
SELECT ai._worker_heartbeat('a0b1c2d3-...', 42, 0, NULL);       -- 42 successes, no errors
SELECT ai._worker_heartbeat('a0b1c2d3-...', 10, 3, 'API 429');  -- 10 successes, 3 errors
```

### ai.\_worker_progress(worker_id, vectorizer_id, successes, error_message)

Per-vectorizer upsert. If `error_message` is NULL, it's a success (increments `success_count`, updates `last_success_*`). If non-NULL, it's an error (increments `error_count`, updates `last_error_*`).

Creates the row on first call for a vectorizer, updates on subsequent calls. Multiple workers can update the same vectorizer — counts accumulate, and the `last_*_process_id` tracks which worker was most recent.

```sql
SELECT ai._worker_progress('a0b1c2d3-...', 1, 5, NULL);              -- 5 successes for vectorizer 1
SELECT ai._worker_progress('a0b1c2d3-...', 1, 0, 'model not found'); -- error for vectorizer 1
```

## Monitoring Queries

### Find dead workers

```sql
SELECT id, version, last_heartbeat, success_count, error_count
FROM ai.vectorizer_worker_process
WHERE last_heartbeat < now() - expected_heartbeat_interval * 2;
```

### Per-vectorizer health

```sql
SELECT v.id, v.source_table, p.success_count, p.error_count,
       p.last_error_message, p.updated_at
FROM ai.vectorizer v
LEFT JOIN ai.vectorizer_worker_progress p ON p.vectorizer_id = v.id
ORDER BY v.id;
```

### Cleanup old worker rows

```sql
DELETE FROM ai.vectorizer_worker_process
WHERE last_heartbeat < now() - interval '7 days';
```

## Design Decisions

- **Progress is per-vectorizer, not per-worker-per-vectorizer.** Gives one view of each vectorizer's health regardless of which worker processed it.
- **`bigint` for all counts.** `int` overflows at ~2.1B — a worker doing 100 embeds/sec hits that in ~248 days.
- **`expected_heartbeat_interval` stored in DB.** Monitoring queries can detect staleness without knowing worker config.
- **No built-in stale worker cleanup.** Each restart creates a new row. Operators clean up with a simple DELETE query.
- **`drop_vectorizer` cascades to progress.** The FK on `vectorizer_worker_progress.vectorizer_id` handles this automatically.
