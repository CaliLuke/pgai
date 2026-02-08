# Worker Tracking: Design Doc

## Problem

The Rust worker calls three SQL functions for lifecycle tracking and heartbeat:

- `ai._worker_start(text) → uuid` — register on connect
- `ai._worker_heartbeat(uuid, bigint, bigint, text)` — periodic health signal
- `ai._worker_progress(uuid, int4, int4, text)` — per-vectorizer counts

These don't exist in `setup.sql`. The worker falls back gracefully (disables heartbeat), but that means:

- No way to detect a silently dead worker
- No visibility into per-vectorizer throughput or errors
- No worker process inventory for multi-worker deployments

## What the Rust Worker Does (`worker_tracking.rs`)

1. **Feature detection:** Checks `information_schema.tables` for `ai.vectorizer_worker_process`. If missing, heartbeat is disabled entirely.
2. **Start:** Calls `ai._worker_start(version_string)`, gets back a `uuid` worker ID.
3. **Heartbeat loop:** Every `poll_interval`, calls `ai._worker_heartbeat(worker_id, successes_delta, errors_delta, last_error)`. Uses atomic counters swapped to zero each tick.
4. **Progress:** After each batch, calls `ai._worker_progress(worker_id, vectorizer_id, count, error_msg)`. `error_msg` is null on success.
5. **Shutdown:** Sends one final heartbeat with remaining counters, then exits.

Counter semantics: `successes`/`errors` in heartbeat are **deltas** accumulated since last heartbeat. The DB function should add them to running totals.

## Proposed Schema

### Tables

```sql
-- One row per worker process (alive or dead)
CREATE TABLE ai.vectorizer_worker_process (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    version       text,
    started_at    timestamptz NOT NULL DEFAULT now(),
    heartbeat_at  timestamptz NOT NULL DEFAULT now(),
    successes     bigint NOT NULL DEFAULT 0,   -- lifetime total
    errors        bigint NOT NULL DEFAULT 0,   -- lifetime total
    last_error    text
);

-- Per-vectorizer progress within a worker
CREATE TABLE ai.vectorizer_worker_progress (
    worker_id      uuid NOT NULL REFERENCES ai.vectorizer_worker_process(id) ON DELETE CASCADE,
    vectorizer_id  int4 NOT NULL REFERENCES ai.vectorizer(id) ON DELETE CASCADE,
    successes      bigint NOT NULL DEFAULT 0,
    errors         bigint NOT NULL DEFAULT 0,
    last_error     text,
    updated_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (worker_id, vectorizer_id)
);
```

### Functions

```sql
-- Register worker, return UUID
ai._worker_start(version text) → uuid
  INSERT INTO ai.vectorizer_worker_process (version) VALUES ($1) RETURNING id;

-- Heartbeat: bump timestamp, accumulate deltas
ai._worker_heartbeat(worker_id uuid, successes bigint, errors bigint, last_error text) → void
  UPDATE ... SET heartbeat_at = now(),
    successes = successes + $2, errors = errors + $3,
    last_error = COALESCE($4, last_error);

-- Per-vectorizer upsert
ai._worker_progress(worker_id uuid, vectorizer_id int4, count int4, error_msg text) → void
  INSERT ... ON CONFLICT DO UPDATE (accumulate counts);
```

## Open Questions

1. **Stale worker cleanup** — Should there be a `ai._worker_reap(interval)` that deletes rows where `heartbeat_at < now() - interval`? Or leave that to the operator?

2. **Table in `ai` schema or `public`?** — The worker feature-detects by checking `ai.vectorizer_worker_process`. Keeping it in `ai` is consistent and gets cleaned up with `DROP SCHEMA ai CASCADE`. But that also means tracking is lost on schema reinstall.

3. **Index on `heartbeat_at`?** — Useful for monitoring queries like "workers not seen in 5 minutes". Low write volume, probably worth it.

4. **Should `_worker_start` clean up the previous row for the same process?** — Currently each restart creates a new row. Over time this accumulates. Options: (a) let it grow, query only recent rows, (b) add a TTL reap function, (c) make the worker pass a stable identifier.

5. **Integration with `drop_vectorizer`** — When a vectorizer is dropped, CASCADE handles the progress FK. But should we also log it somewhere?
