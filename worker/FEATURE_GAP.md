# Rust Worker vs Python Vectorizer — Feature Gap

## Summary Table

| Category            | Python                                | Rust                           | Status                                    |
| ------------------- | ------------------------------------- | ------------------------------ | ----------------------------------------- |
| Embedding Providers | 4 (OpenAI, Ollama, VoyageAI, LiteLLM) | 2 (OpenAI, Ollama)             | VoyageAI, LiteLLM missing                 |
| Chunking            | 3 (None, Character, Recursive)        | 3 (None, Character, Recursive) | Parity (minor impl differences)           |
| Loading             | 2 (Column, URI)                       | 1 (Column)                     | URI/S3 loading missing                    |
| Destination         | 2 (Table, Column)                     | 2 (Table, Column)              | Parity                                    |
| Parsing             | 4 (None, Auto, PyMuPDF, Docling)      | 0                              | Entirely missing                          |
| Formatting          | 2 (ChunkValue, PythonTemplate)        | 2 (ChunkValue, PythonTemplate) | Parity                                    |
| Error Handling      | Retries, exception taxonomy           | EmbeddingError + backon retry  | Parity (embedding); loading retry missing |
| Worker Tracking     | Heartbeat system                      | Heartbeat + progress reporting | Parity                                    |
| Concurrency         | 1-10 workers                          | 1-10 (JoinSet)                 | Parity                                    |
| Graceful Shutdown   | Yes                                   | Yes (CancellationToken)        | Parity                                    |
| Observability       | Datadog (ddtrace)                     | OpenTelemetry (OTLP), RUST_LOG | Different stack, production-ready         |

---

## 1. Embedding Providers

**Implemented:** OpenAI (full: tiktoken, 300K batch limit, 8191 truncation, dimensions, base_url), Ollama (basic: model, base_url)

**Missing:**

- **VoyageAI** — Token-aware batching with native tokenizer, input_type (document/query), output_dimension, output_dtype (float/int8/binary), model-specific token limits (320K-1M)
- **LiteLLM** — Universal adapter for 10+ providers (Cohere, Azure, Bedrock, Gemini, Huggingface, Mistral, Vertex AI). Provider-specific batch/token limits and tokenizers
- **Ollama gaps** — No `options`/`keep_alive` support, no auto-pull, no context length detection from model metadata, no dynamic max_chunks_per_batch env var

## 2. Loading

**Implemented:** Column loading (reads text from a DB column by name)

**Missing:**

- **URI loading** — Loads documents from S3/local URIs via `smart_open`. Supports `aws_role_arn` for STS cross-account access, retries (default 6), external ID
- **Binary/document content** — Column loading in Python also handles bytes to `LoadedDocument` with file type detection

## 3. Document Parsing

Entirely missing. Python supports:

- **PyMuPDF** — PDF, XPS, EPUB, MOBI, FB2, CBZ, SVG to Markdown
- **Docling** — PDF, images (OCR), DOCX, PPTX, HTML to Markdown. Configurable OCR, model caching
- **Auto** — Routes epub to PyMuPDF, others to Docling

## 4. Formatting

**Implemented:** ChunkValue (returns chunk as-is), PythonTemplate (`$chunk` + `$column` substitution from row fields)

~~**Missing:**~~

- ~~**PythonTemplate** — `string.Template` substitution with `$chunk` + all row fields~~ **Done** — `FormatterConfig::format()` with `$variable` substitution

## 5. Chunking (minor gaps)

Feature parity on strategies (None, Character, Recursive). Differences:

- Rust RecursiveCharacterTextSplitter uses `text-splitter` crate, doesn't support custom `separators` list
- Rust CharacterTextSplitter doesn't support `is_separator_regex`
- Different implementations may produce slightly different chunk boundaries

## 6. Error Handling and Retry

**Implemented:** `EmbeddingError` enum (Transient/Permanent) with pattern-based classification, `backon` exponential retry (1s–10s, 3 attempts) for transient embedding errors, error recording to `ai.vectorizer_errors`, `anyhow::Result` propagation, exit_on_error mode

**Missing:**

- Retry logic for loading (Python default: 6 retries)
- ~~Retry logic for embedding API calls (Python OpenAI: max_retries=3)~~ **Done** — `backon` with exponential backoff, transient/permanent classification
- ~~Exception taxonomy (`EmbeddingProviderError`, `LoadingError`, `BatchingError`)~~ **Partial** — `EmbeddingError` implemented; `LoadingError`/`BatchingError` not yet needed
- Failed queue table support
- Traceback/stack trace logging

## 7. Worker Orchestration

**Implemented:** Poll interval, once mode, vectorizer ID filtering, disabled vectorizer skipping, basic version check

**Missing:**

- ~~**Worker tracking/heartbeat** — Start tracking on connect, periodic heartbeats, force last heartbeat on shutdown~~ **Done** — `WorkerTracking` with feature-detection, `ai._worker_start()`, background heartbeat via `ai._worker_heartbeat()`, per-vectorizer progress via `ai._worker_progress()`, final heartbeat on shutdown
- ~~**Graceful shutdown** — `shutdown_requested` event handling~~ **Done** — CancellationToken threaded through Worker → Executor, signal handler for SIGTERM/SIGINT, clean exit between batches
- ~~**Concurrency** — Config exists (1-10) but executor is always single-threaded~~ **Done** — JoinSet spawns N executors per vectorizer
- ~~**DB secret retrieval** — Python can call `ai.reveal_secret()` for API keys~~ **Done** — `resolve_api_key()` tries env var first, falls back to `ai.reveal_secret()` via DB
- **Feature flags** — Detection from DB tables
- **Config migration** — Version upgrade system for vectorizer configs
- **Vectorizer ID randomization** — Fairness when processing multiple vectorizers
- **Connection retry logic**
- **Version upgrade warnings**
- ~~**Datadog tracing** (`ddtrace` integration)~~ **Replaced** — OpenTelemetry with OTLP export (opt-in via `OTEL_EXPORTER_OTLP_ENDPOINT`), `RUST_LOG` env filter, `tracing::instrument` on key functions

## 8. Configuration

**Implemented:** version, embedding, chunking, formatting, loading, destination, processing (batch_size, concurrency)

**Missing:**

- `original_version` tracking
- `log_level` per vectorizer (CRITICAL/ERROR/WARN/INFO/DEBUG)
- Indexing strategies (default, hnsw, diskann, none)
- Scheduling strategies (default, none, timescaledb)
- Config migration system
- Feature flags system

---

## Priority for Production Parity

### P1 — Critical

1. ~~VoyageAI provider~~ Deprioritized — not needed, use OpenAI-compatible base_url
1. ~~LiteLLM provider~~ Deprioritized — not needed, use OpenAI-compatible base_url
1. ~~Retry logic (loading + embedding)~~ Embedding retry done; loading retry still missing
1. ~~Worker tracking/heartbeat~~ Done
1. ~~Graceful shutdown~~ Done

### P2 — Important

1. URI/S3 loading
1. Document parsing (PyMuPDF, Docling)
1. ~~PythonTemplate formatting~~ Done
1. ~~Concurrency implementation~~ Done
1. ~~DB secret retrieval (`ai.reveal_secret()`)~~ Done

### P3 — Nice to have

1. Advanced Ollama features (options, keep_alive, auto-pull)
1. Config migration system
1. Feature flags
1. Custom separators for RecursiveCharacterTextSplitter
1. Indexing/scheduling strategies
1. ~~Datadog tracing~~ Replaced with OpenTelemetry
