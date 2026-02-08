# PostgreSQL Extension — Production Readiness Analysis

## Current State Overview

The `/extension` directory is a **new Rust-based PostgreSQL extension** built with pgrx. It represents an early-stage Rust port of the original Python-based pgai extension. It compiles cleanly but is functionally incomplete.

## File Inventory

```text
extension/
├── Cargo.toml                    # Rust package config (pgrx 0.16.1)
├── extension.control             # pgrx control file
├── .cargo/config.toml
├── src/
│   ├── lib.rs                    # Main implementation (~200 lines)
│   └── bin/pgrx_embed.rs         # pgrx embedding helper
├── sql/
│   └── setup.sql                 # Database schema setup (14 lines)
└── tests/
    └── pg_regress/
        ├── sql/setup.sql
        └── expected/setup.out
```

## What Exists (Functional Code)

### SQL Functions

1. **`openai_embed()`** — Calls OpenAI `/v1/embeddings` API, supports API key from GUC `ai.openai_api_key` or function argument, optional `dimensions` parameter, returns `Vec<f32>`
2. **`ollama_embed()`** — Calls Ollama `/api/embeddings`, configurable base URL (default `http://localhost:11434`), returns `Vec<f32>`
3. **`create_vectorizer()`** — Queries PK info, creates queue table via `CREATE TABLE ... LIKE ... INCLUDING ALL`, inserts config into `ai.vectorizer` metadata table

### Database Schema (setup.sql)

- Creates `ai` schema
- Defines `ai.vectorizer` metadata table (`id`, `source_schema`, `source_table`, `queue_schema`, `queue_table`, `config`, `source_pk`)

### Configuration

- Registers `ai.openai_api_key` GUC string setting at `_PG_init`

## What's Missing

### Queue & Trigger Pipeline

Vectorizers can be created but never executed. There is no trigger creation, no queue handling logic, and no integration with the worker crate's executor. The original Python extension has extensive trigger and queue management — all absent here.

### Provider Coverage

Only OpenAI and Ollama are implemented. Missing providers:

- Anthropic
- Cohere
- LiteLLM
- VoyageAI

No provider abstraction or trait-based architecture exists.

### Error Handling

- Extensive use of `.unwrap_or_else()` with `error!()` macro
- Unchecked `.unwrap()` on JSON object mutation (`body.as_object_mut().unwrap()`)
- No retry logic or transient vs. permanent error classification (unlike the worker crate)
- HTTP errors logged but no detailed error codes or recovery strategies

### Configuration Validation

- Config accepted as raw `JsonB` but never validated or parsed into typed structs
- No chunking logic despite config having a `chunking` field
- No formatting or loading/destination pipeline

### SQL Function Coverage

Only 3 functions exist vs. 20+ in the original Python extension. Missing:

- Semantic catalog functions (text-to-SQL, search, describe, render)
- Chunking functions
- Additional provider-specific functions
- Utility functions

### Testing

- `#[pg_test]` implementation is broken (won't compile)
- No integration tests for API calls
- No tests for error handling or edge cases
- No test data or fixtures
- No mocking for external APIs

### Documentation

- No README or module-level docs
- Minimal inline comments
- No architecture documentation

## Code Quality Issues

### Safety

| Location         | Issue                                                                 |
| ---------------- | --------------------------------------------------------------------- |
| `lib.rs:53`      | `body.as_object_mut().unwrap()` — panics if body is not a JSON object |
| `lib.rs:137-159` | Multiple `.unwrap()` calls in critical database operations            |
| `lib.rs:69,108`  | No validation of JSON structure before parsing API responses          |

### Error Messages

- Generic messages ("Failed to create HTTP client") without context
- No structured error types (contrast with `worker/src/errors.rs` which has `EmbeddingError` with transient/permanent classification)

## Production Readiness Scorecard

| Dimension             | Status  | Notes                                             |
| --------------------- | ------- | ------------------------------------------------- |
| Compilation           | Pass    | Compiles with zero errors, PG13-18 support        |
| Core Features         | Partial | Two embedding providers, basic metadata table     |
| Error Handling        | Weak    | Generic errors, no retry strategy, unsafe unwraps |
| Testing               | Broken  | `pg_test` macro fails, no integration tests       |
| Documentation         | Missing | No docs, minimal comments                         |
| Feature Parity        | Low     | ~10% of Python extension functionality            |
| API Completeness      | Low     | 3 functions vs. 20+ in original                   |
| Production Deployment | No      | Incomplete, untested, insufficient error handling |

## Risks if Deployed

1. **Silent panics** on malformed JSON config
2. **No queue processing** — vectorizers can be created but never executed
3. **No retry logic** — transient API failures will lose work
4. **Incomplete API** — users expecting semantic catalog, text-to-SQL, etc. will get errors
5. **No observability** — worker tracking and progress reporting missing from the extension side

## Comparison with Worker Crate

The worker crate (`worker/`) is significantly more mature:

- Full executor pipeline (fetch → chunk → format → embed → write)
- Transient/permanent error classification with retry logic
- Multiple embedding providers (OpenAI, Ollama) with trait-based abstraction
- Worker tracking and heartbeat reporting
- Comprehensive unit and integration tests
- Proper serde models for all configuration variants

## Recommendations

1. **Fix test infrastructure** — Make `#[pg_test]` macro work, add integration test framework
2. **Add safety checks** — Remove unsafe unwraps, validate JSON before use
3. **Implement core pipeline** — Add trigger setup, queue processing, full executor integration
4. **Add more providers** — Anthropic, Cohere, LiteLLM, VoyageAI
5. **Error handling** — Implement transient vs. permanent classification with retries
6. **Configuration validation** — Parse `JsonB` config into typed structs (reuse `worker/src/models.rs`)
7. **Testing** — Integration tests with mock/real APIs, edge cases, pg_regress tests
8. **Documentation** — README, architecture overview, function-level docs

## Verdict

**Status: Scaffolding / Proof of Concept — NOT production ready.**

The extension is at ~5-10% feature completeness compared to the original Python extension. The worker crate is the functional component; the extension needs significant development before it can replace the Python version.
