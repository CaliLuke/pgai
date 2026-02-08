# Plan: Realistic End-to-End Integration Test

## Context

The current test suite has a gap: the worker integration tests manually create tables, queues,
and destinations — they never use the real `ai.create_vectorizer()` SQL function. The pg_regress
tests verify SQL infrastructure but never run the worker. Nothing tests the full pipeline:
**realistic content → extension creates vectorizer → triggers populate queue → worker embeds → results in target table**.

This test will prove the system works end-to-end using the real extension SQL, realistic content,
and the mock embedding server.

## Discovery Notes

### Worker::new signature change

`Worker::new` now takes 6 arguments (added `CancellationToken` as 6th param). All existing
integration tests are broken — they pass only 5 args. Need to add
`tokio_util::sync::CancellationToken::new()` as the 6th argument to every `Worker::new` call.

### Container image

The plain `postgres` testcontainer image from `testcontainers_modules::postgres::Postgres`
doesn't have pgvector. For the full pipeline test, use `GenericImage` from `testcontainers`:

```rust
use testcontainers::GenericImage;
use testcontainers::core::WaitFor;

let image = GenericImage::new("pgvector/pgvector", "pg17-v0.8.0")
    .with_exposed_port(5432.tcp())
    .with_wait_for(WaitFor::message_on_stderr("database system is ready to accept connections"))
    .with_env_var("POSTGRES_PASSWORD", "postgres")
    .with_env_var("POSTGRES_USER", "postgres")
    .with_env_var("POSTGRES_DB", "postgres");
```

### Loading setup.sql

The extension SQL (`extension/sql/setup.sql`) is pure SQL that creates schema, tables, and
functions. It can be loaded via `include_str!` and split on `$$ LANGUAGE` boundaries or
executed statement by statement. Need to handle multi-statement execution (sqlx doesn't
support multi-statement by default). Could use raw `EXECUTE` or split smartly.

### pgai_lib_version table

The worker checks for `ai.pgai_lib_version` table to verify pgai is installed. The real
`setup.sql` doesn't create this table (it's part of the Python extension, not the Rust one).
The test helper `setup_ai_schema` creates it manually. We need to do the same for the full
pipeline test.

### Vectorizer table schema differences

The real `setup.sql` creates `ai.vectorizer` with column types `name` (not TEXT). It also
includes columns the manual setup doesn't: `trigger_name`, `queue_failed_table`, `name`,
`disabled`. The worker's `Vectorizer` model expects `queue_failed_table` as `Option<String>`.

## Approach

### Part 1: Fix existing integration tests

Add `CancellationToken::new()` as the 6th argument to all `Worker::new` calls in
`worker/tests/integration.rs`.

### Part 2: New pg_regress SQL test

Add a pg_regress test with realistic content that exercises chunking, triggers, and view
correctness.

### Part 3: New worker integration test with real extension SQL

Add a test that uses the pgvector container image, loads setup.sql, calls
`ai.create_vectorizer()`, and runs the worker against it.

## Files to modify

### 1. `worker/tests/integration.rs` (MODIFY)

- Fix all `Worker::new` calls to include `CancellationToken::new()` as 6th arg
- Add `test_full_pipeline_with_extension_sql` test

### 2. `worker/tests/common/mod.rs` (MODIFY)

Add helpers:

- `load_extension_sql(pool)` — loads setup.sql + creates pgai_lib_version table
- `start_pgvector_postgres()` — starts pgvector/pgvector:pg17 container

### 3. `extension/tests/pg_regress/sql/realistic_content.sql` (NEW)

A pg_regress test with realistic data:

- Create a `blog_posts` table with `id`, `title`, `content`, `author`, `published_at`
- Insert 5 posts with realistic multi-paragraph text
- Call `ai.create_vectorizer()` with `chunking_recursive_character_text_splitter`
- Verify: queue has 5 rows, target table exists with correct columns, view joins work
- INSERT a new post → verify queue grows to 6
- UPDATE content → verify re-queued (count goes to 7)
- DELETE a post → verify trigger fires cleanly
- Verify the view has the expected column set
- Clean up with `drop_vectorizer`

### 4. `extension/tests/pg_regress/expected/realistic_content.out` (NEW)

Expected output for the above.

## Verification

```bash
# pg_regress test
cargo pgrx start pg18 -p extension
~/.pgrx/18.1/pgrx-install/bin/psql -h localhost -p 28818 -d postgres \
  -c "DROP DATABASE IF EXISTS extension_regress;"
cargo pgrx regress -p extension pg18

# Worker integration test (compile check)
cargo check --test integration -p worker

# Worker integration test (run, requires Podman)
DOCKER_HOST=unix:///var/folders/ly/gzmh62m90k162x5tz_30m6fm0000gn/T/podman/podman-machine-default-api.sock \
TESTCONTAINERS_RYUK_DISABLED=true \
cargo test --test integration -p worker test_full_pipeline -- --nocapture
```
