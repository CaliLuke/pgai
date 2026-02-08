# SQL Reference

All functions live in the `ai` schema, created by `setup.sql`.

## create_vectorizer

```sql
ai.create_vectorizer(
    source           regclass,
    loading          jsonb DEFAULT NULL,
    embedding        jsonb DEFAULT NULL,
    chunking         jsonb DEFAULT ai.chunking_recursive_character_text_splitter(),
    formatting       jsonb DEFAULT ai.formatting_python_template(),
    processing       jsonb DEFAULT ai.processing_default(),
    destination      jsonb DEFAULT ai.destination_table(),
    queue_schema     name DEFAULT NULL,        -- defaults to 'ai'
    queue_table      name DEFAULT NULL,        -- defaults to '_vectorizer_q_{id}'
    enqueue_existing bool DEFAULT true,
    name             text DEFAULT NULL,        -- defaults to '{schema}_{target_table}'
    if_not_exists    bool DEFAULT false
) RETURNS int4
```

Creates:

- Target table: `{source_table}_embedding_store` (PK cols, chunk_seq, chunk, embedding vector)
- View: `{source_table}_embedding` (joins source and target)
- Queue table: `ai._vectorizer_q_{id}` (PK cols, queued_at)
- Failed queue table: `ai._vectorizer_q_failed_{id}`
- Row-level trigger (INSERT/UPDATE/DELETE) + statement-level trigger (TRUNCATE)
- Metadata row in `ai.vectorizer`
- If `enqueue_existing = true`, copies all existing PKs into queue

Returns the vectorizer ID.

### Example

```sql
SELECT ai.create_vectorizer(
    'artifact_versions'::regclass,
    loading   => ai.loading_column('embedding_text'),
    embedding => ai.embedding_ollama('embeddinggemma:300m', 768, base_url => 'http://localhost:11434'),
    chunking  => ai.chunking_none(),
    processing => ai.processing_default(batch_size => 5)
);
```

## drop_vectorizer

```sql
ai.drop_vectorizer(vectorizer_id int4, drop_all bool DEFAULT true)
```

Drops triggers, trigger function. If `drop_all = true`, also drops view, target table, queue tables. Deletes metadata row.

## vectorizer_queue_pending

```sql
ai.vectorizer_queue_pending(vectorizer_id int4) RETURNS int8
```

Returns count of pending items in the queue table.

## Config Helpers

### Loading

```sql
ai.loading_column(column_name name, retries int4 DEFAULT 6) RETURNS jsonb
-- {"implementation": "column", "column_name": "...", "retries": 6}
```

### Embedding

```sql
ai.embedding_ollama(
    model text, dimensions int4,
    base_url text DEFAULT NULL,
    options jsonb DEFAULT NULL,
    keep_alive text DEFAULT NULL
) RETURNS jsonb

ai.embedding_openai(
    model text, dimensions int4,
    chat_user text DEFAULT NULL,
    api_key_name text DEFAULT 'OPENAI_API_KEY',
    base_url text DEFAULT NULL
) RETURNS jsonb
```

### Chunking

```sql
ai.chunking_none() RETURNS jsonb
-- {"implementation": "none"}

ai.chunking_recursive_character_text_splitter(
    chunk_size int4 DEFAULT 800,
    chunk_overlap int4 DEFAULT 400,
    separators text[] DEFAULT ARRAY[E'\n\n', E'\n', '.', '?', '!', ' ', ''],
    is_separator_regex bool DEFAULT false
) RETURNS jsonb

ai.chunking_character_text_splitter(
    chunk_size int4 DEFAULT 800,
    chunk_overlap int4 DEFAULT 400,
    separator text DEFAULT E'\n\n',
    is_separator_regex bool DEFAULT false
) RETURNS jsonb
```

### Formatting

```sql
ai.formatting_python_template(template text DEFAULT '$chunk') RETURNS jsonb
-- Template supports $chunk and $column_name substitution

ai.formatting_chunk_value() RETURNS jsonb
-- Pass chunk text as-is
```

### Processing

```sql
ai.processing_default(
    batch_size int4 DEFAULT NULL,
    concurrency int4 DEFAULT NULL
) RETURNS jsonb
```

### Destination

```sql
ai.destination_table(
    target_schema name DEFAULT NULL,    -- defaults to source schema
    target_table name DEFAULT NULL,     -- defaults to {source}_embedding_store
    view_schema name DEFAULT NULL,      -- defaults to source schema
    view_name name DEFAULT NULL         -- defaults to {source}_embedding
) RETURNS jsonb
```

## Worker Tracking Functions

Internal functions used by the worker for health reporting. See [worker-tracking.md](worker-tracking.md) for full details.

```sql
ai._worker_start(version text, expected_heartbeat_interval interval) RETURNS uuid
```

Registers a worker process. Returns a UUID for subsequent calls.

```sql
ai._worker_heartbeat(
    worker_id uuid,
    num_successes_since_last_heartbeat bigint,
    num_errors_since_last_heartbeat bigint,
    error_message text
) RETURNS void
```

Periodic health signal. Accumulates success/error deltas into lifetime totals.

```sql
ai._worker_progress(
    worker_id uuid,
    worker_vectorizer_id int4,
    num_successes bigint,
    error_message text
) RETURNS void
```

Per-vectorizer upsert. `error_message = NULL` is the success path, non-NULL is the error path.

## Tables

### ai.vectorizer

| Column             | Type        | Notes                                                                                |
| ------------------ | ----------- | ------------------------------------------------------------------------------------ |
| id                 | int4 PK     | From `ai.vectorizer_id_seq`                                                          |
| source_schema      | name        |                                                                                      |
| source_table       | name        |                                                                                      |
| source_pk          | jsonb       | Array of `{attnum, pknum, attname, typname}`                                         |
| trigger_name       | name        |                                                                                      |
| queue_schema       | name        |                                                                                      |
| queue_table        | name        |                                                                                      |
| queue_failed_table | name        |                                                                                      |
| config             | jsonb       | Full config blob (loading, embedding, chunking, formatting, processing, destination) |
| name               | text UNIQUE | Human-friendly identifier                                                            |
| disabled           | bool        | Default false                                                                        |

### ai.vectorizer_errors

| Column      | Type        | Notes                                       |
| ----------- | ----------- | ------------------------------------------- |
| id          | int4 FK     | References vectorizer(id) ON DELETE CASCADE |
| message     | text        |                                             |
| details     | jsonb       |                                             |
| recorded_at | timestamptz | Default now()                               |

### ai.vectorizer_worker_process

| Column                      | Type        | Notes                     |
| --------------------------- | ----------- | ------------------------- |
| id                          | uuid PK     | Auto-generated            |
| version                     | text        | Worker version string     |
| started                     | timestamptz | Default now()             |
| expected_heartbeat_interval | interval    | Worker's poll interval    |
| last_heartbeat              | timestamptz | Default now(), indexed    |
| heartbeat_count             | bigint      | Lifetime total heartbeats |
| success_count               | bigint      | Lifetime total successes  |
| error_count                 | bigint      | Lifetime total errors     |
| last_error_at               | timestamptz | NULL until first error    |
| last_error_message          | text        | NULL until first error    |

### ai.vectorizer_worker_progress

| Column                  | Type        | Notes                                  |
| ----------------------- | ----------- | -------------------------------------- |
| vectorizer_id           | int4 PK     | FK to vectorizer(id) ON DELETE CASCADE |
| success_count           | bigint      | Lifetime total successes               |
| error_count             | bigint      | Lifetime total errors                  |
| last_success_at         | timestamptz | NULL until first success               |
| last_success_process_id | uuid        | No FK to worker_process (intentional)  |
| last_error_at           | timestamptz | NULL until first error                 |
| last_error_message      | text        | NULL until first error                 |
| last_error_process_id   | uuid        | No FK to worker_process (intentional)  |
| updated_at              | timestamptz | Default now(), set on every upsert     |

### Generated queue table (`ai._vectorizer_q_{id}`)

| Column                   | Type                      |
| ------------------------ | ------------------------- |
| (PK columns from source) | (same types)              |
| queued_at                | timestamptz DEFAULT now() |

### Generated target table (`{source}_embedding_store`)

| Column                   | Type                        |
| ------------------------ | --------------------------- |
| embedding_uuid           | uuid PK (gen_random_uuid()) |
| (PK columns from source) | (same types)                |
| chunk_seq                | int                         |
| chunk                    | text                        |
| embedding                | vector(dimensions)          |
| UNIQUE                   | (PK cols, chunk_seq)        |
