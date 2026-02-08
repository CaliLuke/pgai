# pgai

A Rust toolkit that turns PostgreSQL into a retrieval engine for RAG applications. Automatically creates and synchronizes vector embeddings from PostgreSQL data.

## Components

- **`worker/`** — Async vectorizer worker. Polls the database for pending work, chunks text, calls embedding APIs, writes results back.
- **`extension/`** — PostgreSQL extension (pgrx). Provides the `ai` schema, `create_vectorizer()` SQL API, config helpers, triggers, and queue infrastructure.
- **`text-splitter/`** — Text chunking library. Character splitter, recursive splitter, sentence chunker, and semantic recursive splitter (semchunk).

## How it works

1. Install the extension — creates the `ai` schema with vectorizer infrastructure.
2. Define a vectorizer via `ai.create_vectorizer()` — specifying source table/column, embedding model, and chunking strategy.
3. The worker polls for pending work, fetches rows, chunks, embeds, and writes results back.
4. Triggers keep embeddings in sync as data changes.

```sql
SELECT ai.create_vectorizer(
    'my_table'::regclass,
    loading   => ai.loading_column('content'),
    embedding => ai.embedding_ollama('embeddinggemma:300m', 768),
    chunking  => ai.chunking_none()
);
```

## Building

```bash
cargo build -p worker          # worker binary
cargo check -p extension       # extension (full build needs cargo pgrx)
cargo build -p pgai-text-splitter  # text splitter library
```

## Running the worker

```bash
DB_URL=postgres://user:pass@localhost/mydb cargo run -p worker
```

Options:

- `--poll-interval 60` — seconds between polls (default: 60)
- `--once` — run one cycle and exit
- `--vectorizer-ids 1,2,3` — only process specific vectorizers
- `--exit-on-error` — exit on first error instead of continuing

## Embedding providers

- **OpenAI** — tiktoken token counting, 300K token batch limit, 8191 context truncation, configurable dimensions and base_url
- **Ollama** — local models via ollama-rs, configurable base URL via `OLLAMA_HOST` env var

## Chunking strategies

- **None** — pass text through as-is (for short text / pre-chunked content)
- **Character text splitter** — split on a single separator
- **Recursive character text splitter** — split on a hierarchy of separators
- **Sentence chunker** — greedily pack whole sentences, configurable delimiters
- **Semchunk** — semantic recursive splitter with fixed 8-level delimiter hierarchy

## SQL API

Config helpers (return JSONB, compose into `create_vectorizer`):

| Function                                             | Purpose                           |
| ---------------------------------------------------- | --------------------------------- |
| `ai.loading_column(column_name)`                     | Which column to embed             |
| `ai.embedding_ollama(model, dimensions, ...)`        | Ollama embedding config           |
| `ai.embedding_openai(model, dimensions, ...)`        | OpenAI embedding config           |
| `ai.chunking_none()`                                 | No chunking                       |
| `ai.chunking_recursive_character_text_splitter(...)` | Recursive splitter                |
| `ai.chunking_character_text_splitter(...)`           | Character splitter                |
| `ai.formatting_python_template(template)`            | `$chunk` + `$column` substitution |
| `ai.formatting_chunk_value()`                        | Pass chunk as-is                  |
| `ai.processing_default(batch_size, concurrency)`     | Worker tuning                     |
| `ai.destination_table(...)`                          | Target table config               |

Management:

| Function                          | Purpose                               |
| --------------------------------- | ------------------------------------- |
| `ai.create_vectorizer(...)`       | Create vectorizer + triggers + tables |
| `ai.drop_vectorizer(id)`          | Remove vectorizer and all artifacts   |
| `ai.vectorizer_queue_pending(id)` | Check queue depth                     |

## Testing

```bash
cargo test -p pgai-text-splitter   # 60 tests
cargo test -p worker --lib         # 37 tests
cargo check -p extension           # type-check (linking needs pgrx)
```
