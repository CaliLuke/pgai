# Vectorizer Backport Plan — Minimum Viable for autok

## Goal

Enable autok's automatic embedding pipeline using our Rust pgai extension + worker. autok calls:

```sql
SELECT ai.create_vectorizer(
    'artifact_versions'::regclass,
    loading    => ai.loading_column('embedding_text'),
    embedding  => ai.embedding_ollama('embeddinggemma:300m', 768, base_url => 'http://localhost:11434'),
    chunking   => ai.chunking_none(),
    processing => ai.processing_default(batch_size => 5)
);
```

This plan covers **only** what's needed to make that call work end-to-end: the extension installs the SQL infrastructure, triggers enqueue rows, and our existing Rust worker processes them.

## Architecture Overview

```text
┌─────────────────┐    INSERT/UPDATE    ┌──────────────────┐
│  artifact_       │ ──── trigger ────> │  ai._vectorizer  │
│  versions        │                    │  _q_{id}         │
└─────────────────┘                    └────────┬─────────┘
                                                │
                                     poll (FOR UPDATE SKIP LOCKED)
                                                │
                                       ┌────────▼─────────┐
                                       │  Rust worker      │
                                       │  (already exists) │
                                       └────────┬─────────┘
                                                │
                                          embed via Ollama
                                                │
                                       ┌────────▼──────────────┐
                                       │  artifact_versions_   │
                                       │  embedding_store      │
                                       └───────────────────────┘
```

## What Already Works

- **Worker queue fetch** (`executor.rs` `fetch_work`) — `FOR UPDATE SKIP LOCKED` with advisory locks
- **Worker write** (`executor.rs` `write_to_table`) — inserts into destination table with PK + chunk_seq + chunk + embedding
- **Ollama embedder** (`embedder.rs`) — calls `/api/embed` endpoint
- **ChunkerConfig::None** — passes text through unchanged
- **Error handling** — transient/permanent classification with retry

## What Needs to Be Built

Everything below is **pure SQL** installed by the pgrx extension at `CREATE EXTENSION` time. No new Rust code in the extension crate — we replace `extension/src/lib.rs` with only `_PG_init` + embedding helper functions, and add a `setup.sql` that installs the full SQL infrastructure.

---

## Step 1: Schema and Metadata Tables

**File: `extension/sql/setup.sql`** (replaces current)

Create the `ai` schema and the vectorizer registry:

```sql
CREATE SCHEMA IF NOT EXISTS ai;

CREATE SEQUENCE IF NOT EXISTS ai.vectorizer_id_seq;

CREATE TABLE IF NOT EXISTS ai.vectorizer (
    id          int4 NOT NULL PRIMARY KEY DEFAULT nextval('ai.vectorizer_id_seq'),
    source_schema  name NOT NULL,
    source_table   name NOT NULL,
    source_pk      jsonb NOT NULL,
    trigger_name   name NOT NULL,
    queue_schema   name,
    queue_table    name,
    queue_failed_table name,
    config         jsonb NOT NULL,
    name           text UNIQUE,
    disabled       bool NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS ai.vectorizer_errors (
    id       int4 NOT NULL REFERENCES ai.vectorizer(id) ON DELETE CASCADE,
    message  text,
    details  jsonb,
    recorded timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_vectorizer_errors_id_recorded
    ON ai.vectorizer_errors(id, recorded);
```

**Changes from current**: Adds `trigger_name`, `queue_failed_table`, `name`, `disabled` columns. Removes `target_schema`/`target_table`/`view_schema`/`view_name` from the table (these live in `config->destination` instead). Adds `vectorizer_errors` table.

---

## Step 2: Config Helper Functions

Pure SQL functions that return JSONB config blobs. These are what autok calls (`ai.loading_column(...)`, etc.).

**File: `extension/sql/config_helpers.sql`**

### 2a. `ai.loading_column(column_name, retries)`

```sql
CREATE OR REPLACE FUNCTION ai.loading_column(
    column_name name,
    retries int4 DEFAULT 6
) RETURNS jsonb AS $$
    SELECT json_build_object(
        'implementation', 'column',
        'config_type', 'loading',
        'column_name', column_name,
        'retries', retries
    )::jsonb
$$ LANGUAGE sql IMMUTABLE;
```

### 2b. `ai.embedding_ollama(model, dimensions, base_url, options, keep_alive)`

```sql
CREATE OR REPLACE FUNCTION ai.embedding_ollama(
    model text,
    dimensions int4,
    base_url text DEFAULT NULL,
    options jsonb DEFAULT NULL,
    keep_alive text DEFAULT NULL
) RETURNS jsonb AS $$
    SELECT json_strip_nulls(json_build_object(
        'implementation', 'ollama',
        'config_type', 'embedding',
        'model', model,
        'dimensions', dimensions,
        'base_url', base_url,
        'options', options,
        'keep_alive', keep_alive
    ))::jsonb
$$ LANGUAGE sql IMMUTABLE;
```

### 2c. `ai.embedding_openai(model, dimensions, api_key_name, base_url)`

```sql
CREATE OR REPLACE FUNCTION ai.embedding_openai(
    model text,
    dimensions int4,
    chat_user text DEFAULT NULL,
    api_key_name text DEFAULT 'OPENAI_API_KEY',
    base_url text DEFAULT NULL
) RETURNS jsonb AS $$
    SELECT json_strip_nulls(json_build_object(
        'implementation', 'openai',
        'config_type', 'embedding',
        'model', model,
        'dimensions', dimensions,
        'user', chat_user,
        'api_key_name', api_key_name,
        'base_url', base_url
    ))::jsonb
$$ LANGUAGE sql IMMUTABLE;
```

### 2d. `ai.chunking_none()`

```sql
CREATE OR REPLACE FUNCTION ai.chunking_none() RETURNS jsonb AS $$
    SELECT json_build_object(
        'implementation', 'none',
        'config_type', 'chunking'
    )::jsonb
$$ LANGUAGE sql IMMUTABLE;
```

### 2e. `ai.chunking_recursive_character_text_splitter(...)`

```sql
CREATE OR REPLACE FUNCTION ai.chunking_recursive_character_text_splitter(
    chunk_size int4 DEFAULT 800,
    chunk_overlap int4 DEFAULT 400,
    separators text[] DEFAULT ARRAY[E'\n\n', E'\n', '.', '?', '!', ' ', ''],
    is_separator_regex bool DEFAULT false
) RETURNS jsonb AS $$
    SELECT json_strip_nulls(json_build_object(
        'implementation', 'recursive_character_text_splitter',
        'config_type', 'chunking',
        'chunk_size', chunk_size,
        'chunk_overlap', chunk_overlap,
        'separators', separators,
        'is_separator_regex', is_separator_regex
    ))::jsonb
$$ LANGUAGE sql IMMUTABLE;
```

### 2f. `ai.processing_default(batch_size, concurrency)`

```sql
CREATE OR REPLACE FUNCTION ai.processing_default(
    batch_size int4 DEFAULT NULL,
    concurrency int4 DEFAULT NULL
) RETURNS jsonb AS $$
    SELECT json_strip_nulls(json_build_object(
        'implementation', 'default',
        'config_type', 'processing',
        'batch_size', batch_size,
        'concurrency', concurrency
    ))::jsonb
$$ LANGUAGE sql IMMUTABLE;
```

### 2g. `ai.destination_table(...)` (with defaults)

```sql
CREATE OR REPLACE FUNCTION ai.destination_table(
    target_schema name DEFAULT NULL,
    target_table name DEFAULT NULL,
    view_schema name DEFAULT NULL,
    view_name name DEFAULT NULL
) RETURNS jsonb AS $$
    SELECT json_strip_nulls(json_build_object(
        'implementation', 'table',
        'config_type', 'destination',
        'target_schema', target_schema,
        'target_table', target_table,
        'view_schema', view_schema,
        'view_name', view_name
    ))::jsonb
$$ LANGUAGE sql IMMUTABLE;
```

### 2h. `ai.formatting_python_template(template)`

```sql
CREATE OR REPLACE FUNCTION ai.formatting_python_template(
    template text DEFAULT '$chunk'
) RETURNS jsonb AS $$
    SELECT json_build_object(
        'implementation', 'python_template',
        'config_type', 'formatting',
        'template', template
    )::jsonb
$$ LANGUAGE sql IMMUTABLE;
```

---

## Step 3: Internal Helper Functions

**File: `extension/sql/vectorizer_internals.sql`**

### 3a. `ai._vectorizer_source_pk(source regclass) -> jsonb`

Extracts the primary key column metadata from a source table. Returns a JSON array like:

```json
[{ "attnum": 1, "pknum": 1, "attname": "id", "typname": "integer" }]
```

Port directly from the reference's `_vectorizer_source_pk`.

### 3b. `ai._vectorizer_create_target_table(...)`

Creates the embedding store table:

```sql
CREATE TABLE {target_schema}.{target_table} (
    embedding_uuid uuid NOT NULL PRIMARY KEY DEFAULT gen_random_uuid(),
    {pk_columns with types, NOT NULL},
    chunk_seq  int NOT NULL,
    chunk      text NOT NULL,
    embedding  vector({dimensions}) NOT NULL,
    UNIQUE ({pk_columns}, chunk_seq)
);
```

Requires `pgvector` extension. We add `CREATE EXTENSION IF NOT EXISTS vector` at the top of `setup.sql`.

### 3c. `ai._vectorizer_create_view(...)`

Creates a join view between source and target:

```sql
CREATE VIEW {view_schema}.{view_name} AS
SELECT
    t.embedding_uuid, t.chunk_seq, t.chunk, t.embedding,
    s.*
FROM {source_schema}.{source_table} s
INNER JOIN {target_schema}.{target_table} t
    ON {join_on_pk_columns};
```

### 3d. `ai._vectorizer_create_queue_table(queue_schema, queue_table, source_pk)`

Creates the work queue table with PK columns + `queued_at` timestamp:

```sql
CREATE TABLE {queue_schema}.{queue_table} (
    {pk_columns with types},
    queued_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON {queue_schema}.{queue_table} ({pk_columns});
```

### 3e. `ai._vectorizer_create_queue_failed_table(queue_schema, table_name, source_pk)`

Same structure as queue table plus `failure_step text`:

```sql
CREATE TABLE {queue_schema}.{table_name} (
    {pk_columns with types},
    created_at timestamptz NOT NULL DEFAULT now(),
    failure_step text NOT NULL DEFAULT ''
);
```

### 3f. `ai._vectorizer_build_trigger_definition(...) -> text`

Generates the PL/pgSQL trigger function body. The trigger handles:

- **INSERT**: Queue the new row's PK
- **UPDATE**: If PK changed, delete old embeddings + queue new PK. If relevant (non-embedding) columns changed, queue the PK. If only embedding column changed, skip.
- **DELETE**: Delete matching rows from the destination table
- **TRUNCATE** (statement-level): Truncate both destination and queue tables

Template for table destination:

```sql
BEGIN
    IF (TG_LEVEL = 'ROW') THEN
        IF (TG_OP = 'DELETE') THEN
            DELETE FROM {target_schema}.{target_table}
                WHERE {pk_col} = OLD.{pk_col} [AND ...];
        ELSIF (TG_OP = 'UPDATE') THEN
            IF {any PK column changed: OLD.pk != NEW.pk} THEN
                DELETE FROM {target_schema}.{target_table}
                    WHERE {pk_col} = OLD.{pk_col};
                INSERT INTO {queue_schema}.{queue_table} ({pk_cols})
                    VALUES (NEW.{pk_cols});
            ELSIF {any relevant column changed} THEN
                INSERT INTO {queue_schema}.{queue_table} ({pk_cols})
                    VALUES (NEW.{pk_cols});
            END IF;
            RETURN NEW;
        ELSE -- INSERT
            INSERT INTO {queue_schema}.{queue_table} ({pk_cols})
                VALUES (NEW.{pk_cols});
            RETURN NEW;
        END IF;
    ELSIF (TG_LEVEL = 'STATEMENT') THEN
        IF (TG_OP = 'TRUNCATE') THEN
            TRUNCATE TABLE {target_schema}.{target_table};
            TRUNCATE TABLE {queue_schema}.{queue_table};
        END IF;
        RETURN NULL;
    END IF;
    RETURN NULL;
END;
```

The "relevant columns changed" check excludes the embedding column itself to avoid re-embedding loops.

### 3g. `ai._vectorizer_create_source_trigger(...)`

Creates the trigger function and attaches two triggers to the source table:

1. **Row trigger**: `AFTER INSERT OR UPDATE OR DELETE FOR EACH ROW`
2. **Statement trigger**: `AFTER TRUNCATE FOR EACH STATEMENT`

Both call the same trigger function (generated by 3f). The function is `SECURITY DEFINER` so it can write to the `ai.*` queue tables regardless of the caller's permissions.

---

## Step 4: `ai.create_vectorizer()` Orchestrator

**File: `extension/sql/create_vectorizer.sql`**

The main entry point. Signature matches the reference:

```sql
CREATE OR REPLACE FUNCTION ai.create_vectorizer(
    source           regclass,
    loading          jsonb DEFAULT NULL,
    embedding        jsonb DEFAULT NULL,
    chunking         jsonb DEFAULT ai.chunking_recursive_character_text_splitter(),
    formatting       jsonb DEFAULT ai.formatting_python_template(),
    processing       jsonb DEFAULT ai.processing_default(),
    destination      jsonb DEFAULT ai.destination_table(),
    queue_schema     name DEFAULT NULL,
    queue_table      name DEFAULT NULL,
    enqueue_existing bool DEFAULT true,
    name             text DEFAULT NULL,
    if_not_exists    bool DEFAULT false
) RETURNS int4
```

Algorithm:

1. **Validate inputs** — `embedding` and `loading` are required
2. **Resolve source** — extract `source_schema`, `source_table` from `regclass`
3. **Get source PK** — call `ai._vectorizer_source_pk(source)`; error if no PK
4. **Allocate ID** — `nextval('ai.vectorizer_id_seq')`
5. **Resolve defaults**:
   - `queue_schema` defaults to `'ai'`
   - `queue_table` defaults to `'_vectorizer_q_{id}'`
   - `queue_failed_table` = `'_vectorizer_q_failed_{id}'`
   - `trigger_name` = `'_vectorizer_src_trg_{id}'`
   - Resolve destination defaults (target_table defaults to `{source_table}_embedding_store`, view_name defaults to `{source_table}_embedding`)
   - `name` defaults to `{target_schema}_{target_table}`
6. **Check name uniqueness** — if `if_not_exists` and name taken, return existing ID
7. **Create destination** — call `_vectorizer_create_target_table` + `_vectorizer_create_view`
8. **Create queue table** — call `_vectorizer_create_queue_table`
9. **Create queue failed table** — call `_vectorizer_create_queue_failed_table`
10. **Create source trigger** — call `_vectorizer_create_source_trigger`
11. **Insert vectorizer record** — into `ai.vectorizer` with full config JSONB
12. **Enqueue existing rows** — if `enqueue_existing = true`, `INSERT INTO queue SELECT pk FROM source`
13. **Return** vectorizer ID

### Simplifications vs. reference

We skip these features (not needed for autok):

- `grant_to` role permission management — not needed for single-user dev setup
- `scheduling` (TimescaleDB background jobs) — autok runs worker as sidecar
- `indexing` (auto HNSW/DiskANN index creation) — can be added manually
- `parsing` (document parsing) — autok uses pre-built `embedding_text`
- Config validation functions — we trust the config helpers produce valid JSONB
- `destination_column` support — autok uses table destination

---

## Step 5: `ai.drop_vectorizer()` Cleanup

**File: `extension/sql/drop_vectorizer.sql`**

```sql
CREATE OR REPLACE FUNCTION ai.drop_vectorizer(
    vectorizer_id int4,
    drop_all bool DEFAULT true
) RETURNS void
```

1. Look up vectorizer record
2. Drop source trigger + trigger function
3. If `drop_all`: drop queue table, queue failed table, destination table, view
4. Delete from `ai.vectorizer`

---

## Step 6: `ai.vectorizer_queue_pending()` Status Check

```sql
CREATE OR REPLACE FUNCTION ai.vectorizer_queue_pending(
    vectorizer_id int4
) RETURNS int8
```

Returns `SELECT count(*) FROM {queue_schema}.{queue_table}`. Useful for monitoring.

---

## Step 7: Keep `extension/src/lib.rs` As-Is

No changes to the Rust code. The existing `openai_embed()`, `ollama_embed()`, and `create_vectorizer()` Rust functions remain available. The new SQL `ai.create_vectorizer()` is a separate PL/pgSQL function with a different signature (takes `regclass` + config helper JSONB args), so there's no conflict. Users get both:

- **Rust `create_vectorizer(source_table text, config jsonb)`** — low-level, manual config
- **SQL `ai.create_vectorizer(source regclass, ...)`** — high-level, with config helpers and full infrastructure setup (triggers, queue, destination table)

The `setup.sql` is loaded by pgrx at `CREATE EXTENSION` time via the `extension.control` file.

---

## Step 8: Wire up SQL Loading in pgrx

**File: `extension/extension.control`**

Ensure the control file points to setup SQL. pgrx handles this, but we need to make sure `@extschema@` references resolve. The SQL files from steps 1-6 get concatenated into a single install script.

**Approach**: Use a single `extension/sql/setup.sql` file that contains all the SQL in order:

1. Schema + tables (step 1)
2. Config helpers (step 2)
3. Internal functions (step 3)
4. `create_vectorizer` (step 4)
5. `drop_vectorizer` (step 5)
6. `vectorizer_queue_pending` (step 6)

pgrx will load `setup.sql` when the extension is created.

---

## Step 9: Worker Compatibility Check

Verify our worker's `fetch_work` query and `write_to_table` match the schema we create.

### Queue table schema match

Our worker (`executor.rs:309-382`) expects:

- Queue table with PK columns only (it selects PK columns, joins to source)
- Uses `FOR UPDATE SKIP LOCKED` + `pg_try_advisory_xact_lock`

Our new queue table has PK columns + `queued_at`. The worker selects only `{pk_list}` columns from the queue, so the extra `queued_at` column is harmless. **Compatible.**

### Destination table schema match

Our worker (`executor.rs:216-269`) writes:

- `DELETE FROM target WHERE pk = item.pk` (per item)
- `INSERT INTO target (pk_cols, chunk_seq, chunk, embedding) VALUES (...)`

Our new target table has: `embedding_uuid` (auto-generated), PK columns, `chunk_seq`, `chunk`, `embedding`. The INSERT needs to specify only the non-default columns. **The worker currently lists columns explicitly as `(pk_cols, chunk_seq, chunk, embedding)` — this matches.** The `embedding_uuid` has a DEFAULT, so it auto-populates. **Compatible.**

### Vectorizer config match

Our worker (`models.rs`) deserializes `config` from `ai.vectorizer`. The config JSONB we store includes `loading`, `embedding`, `chunking`, `formatting`, `processing`, `destination`. The worker reads `config.embedding`, `config.chunking`, `config.formatting`, `config.loading`, `config.destination`, `config.processing`. **Compatible.**

### One fix needed

The worker's `fetch_work` query references `self.vectorizer.queue_failed_table` but our current `Vectorizer` model has this as `Option<String>`. The new schema always creates a failed table, so this will be `Some(...)`. **No change needed.**

---

## Step 10: Update `Vectorizer` Model (worker)

The worker's `Vectorizer` struct (`models.rs:17-32`) needs two new fields to match the new schema:

```rust
pub trigger_name: String,
pub name: Option<String>,
```

These are just for completeness — the worker doesn't use them directly, but they come back from the `ai.vectorizer` query. Add `#[serde(default)]` if we want backwards compatibility. Since the worker `SELECT`s specific columns (not `SELECT *`), this may not even be needed. **Check the worker's vectorizer query.**

---

## File Summary

| File                          | Action    | Content                                                                     |
| ----------------------------- | --------- | --------------------------------------------------------------------------- |
| `extension/sql/setup.sql`     | REWRITE   | Full SQL: schema, tables, config helpers, internals, create/drop_vectorizer |
| `extension/src/lib.rs`        | NO CHANGE | Existing Rust functions kept as-is                                          |
| `extension/extension.control` | VERIFY    | Ensure SQL loading works with pgrx                                          |
| `worker/src/models.rs`        | MINOR     | Add `trigger_name`, `name` fields if needed                                 |

---

## Testing Plan

### Unit tests (no DB)

- Config helper functions return correct JSONB structure (test in SQL or via pgrx `pg_test`)
- Worker deserializes vectorizer config from the new schema

### Integration test (with DB)

1. `CREATE EXTENSION pgai` — installs schema + functions
2. `CREATE EXTENSION vector` — pgvector prerequisite
3. `CREATE TABLE test_source (id serial PRIMARY KEY, content text)`
4. `INSERT INTO test_source (content) VALUES ('hello world')`
5. Call `ai.create_vectorizer(...)` with Ollama config
6. Verify: `ai.vectorizer` has a row
7. Verify: queue table exists and has 1 row (from `enqueue_existing`)
8. Verify: destination table exists with correct schema
9. Verify: view exists
10. Verify: insert another row into `test_source` — queue gets a new entry (trigger works)
11. Verify: update content — queue gets a new entry
12. Verify: delete row — destination row deleted
13. Run worker against the DB — embeddings appear in destination table

### Smoke test against autok

1. Run autok migration with our extension installed instead of Python pgai
2. Start worker pointed at the same DB
3. Create an artifact version
4. Verify embedding appears in `artifact_versions_embedding_store`
5. Verify duplicate detection query works

---

## Implementation Order

1. Write `setup.sql` with all SQL (steps 1-6) — this is the bulk of the work
2. Simplify `lib.rs` (step 7)
3. Verify worker compatibility (step 9)
4. Test with `cargo pgrx test` (pg_test)
5. Integration test with real DB + worker
6. Test against autok

---

## Out of Scope

- TimescaleDB scheduling
- DiskANN/HNSW auto-indexing
- Document loading (URI/S3)
- Document parsing (PyMuPDF)
- VoyageAI/LiteLLM embedding providers
- `grant_to` permission management
- Worker tracking tables (`vectorizer_worker_process`, `vectorizer_worker_progress`)
- `pgai.install()` Python library compatibility
- Column destination mode

These can all be added incrementally later without breaking the core pipeline.
