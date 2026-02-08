# pgai-rs Integration Feedback

Issues found while integrating `extension/sql/setup.sql` and the Rust worker into Auto-K (replacing both the `pgai` Python package and the `timescale/pgai-vectorizer-worker` Docker image).

## Verified Working

- Rust worker binary connects, discovers vectorizers, fetches queue items, generates embeddings via Ollama, writes to embedding store table
- Processed 227 embeddings across 183 artifact versions with `--once` mode
- `embeddinggemma:300m` (768 dims) works correctly
- Graceful shutdown on SIGTERM/SIGINT works

---

## Bugs

### 1. Missing table alias in `enqueue_existing` (FIXED)

**File:** `extension/sql/setup.sql`, line ~760 (in `create_vectorizer` function)

The `SELECT` in the `enqueue_existing` block uses `x.%I` column references but doesn't alias the source table as `x`:

```sql
-- Before (broken):
INSERT INTO %I.%I (%s)
SELECT %s FROM %I.%I

-- After (fixed):
INSERT INTO %I.%I (%s)
SELECT %s FROM %I.%I x
```

The `format('x.%I', x.attname)` on line ~767 generates column references like `x.id`, but the FROM clause had no `x` alias, causing `missing FROM-clause entry for table "x"`.

**Status:** Already fixed in this repo.

### 2. Missing `ai._worker_start()` function in setup.sql

**Observed:** Worker logs `WARN worker_tracking: Failed to register worker, heartbeat disabled: function ai._worker_start(text) does not exist`

The Rust worker calls `ai._worker_start(text)` for heartbeat registration, but `setup.sql` doesn't define this function. The worker continues without heartbeat (non-fatal), but it means no worker tracking in the `ai` schema.

**Fix:** Add `ai._worker_start()`, `ai._worker_stop()`, and `ai._worker_heartbeat()` functions to setup.sql, or document that worker tracking requires the full pgrx extension.

---

## Improvements

### 3. `create_vectorizer` should be idempotent

`create_vectorizer` raises `DuplicateObject` if the embedding store table already exists. This makes it impossible to re-run after a partial failure or schema re-installation (e.g., dropping and recreating `ai` schema while `public.*_embedding_store` tables survive).

**Suggestion:** Use `CREATE TABLE IF NOT EXISTS` for the embedding store and queue tables, or add an early check that skips creation if the vectorizer's target table already exists.

### 4. `formatting: Unknown` in vectorizer config deserialization

**Observed:** Worker logs show `formatting: Unknown` for vectorizer configs created with the default formatting. The Rust `Config` struct deserializes the formatting field as `Unknown` rather than a recognized variant.

This doesn't cause failures (the worker still processes items correctly), but it means any formatting-dependent logic is silently skipped.

### 5. SQLAlchemy compatibility note for setup.sql

The SQL contains PostgreSQL `format()` specifiers (`%I`, `%L`, `%s`) inside PL/pgSQL function bodies. These conflict with Python DB-API `%`-style parameter markers. When executing via SQLAlchemy's `text()`, the `%` characters are misinterpreted as bind parameters.

**Workaround:** Execute via raw DBAPI cursor: `connection.connection.cursor().execute(sql)`

**Suggestion:** Add a note to README/docs that setup.sql must be executed via raw cursor when using SQLAlchemy or any DB-API library that uses `%` parameter style.

---

## Nice-to-haves

### 6. Vectorizer 24 (stale) causes unnecessary work

The worker found two vectorizers (24 and 29). Vectorizer 24 is from an old config and has no queue table. Would be nice if `setup.sql` included a helper to list/clean stale vectorizers, or if the worker skipped vectorizers with missing queue tables more gracefully.

### 7. Chunking config mismatch

Vectorizer 29 was created with `chunking => ai.chunking_none()` but the worker sees `RecursiveCharacterTextSplitter { chunk_size: 2000, chunk_overlap: 200, ... }`. This suggests the vectorizer config was modified or the default chunking behavior differs between `create_vectorizer` and what gets stored. Not causing issues (embeddings generate fine) but the mismatch is confusing.
