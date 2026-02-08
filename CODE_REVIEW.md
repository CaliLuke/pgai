# Code Review: pgai

This document outlines the findings of a thorough code review of the `pgai` project, covering security, architecture, performance, and robustness.

## 1. Security Analysis

### High Severity: Undefined Secret Retrieval Function
*   **Location:** `worker/src/embedder.rs` (`resolve_api_key`)
*   **Issue:** The worker calls `ai.reveal_secret($1)` in the database if an environment variable is missing. This function is not defined in `setup.sql`.
*   **Risk:** The worker will crash/fail if it attempts to fall back to database secrets.
*   **Recommendation:** Define a standard, secure secret management schema or integrate with `pg_vault`.

### Medium Severity: PII Leakage to External APIs
*   **Location:** `worker/src/models.rs` (`FormatterConfig::format`)
*   **Issue:** The `PythonTemplate` formatter allows arbitrary columns from the source table to be injected into the text chunk sent to embedding providers (OpenAI/Ollama).
*   **Risk:** Sensitive data (emails, PII, internal IDs) can be accidentally included in embeddings, violating privacy policies and exposing data to third-party providers.
*   **Recommendation:** Implement a "Safe Columns" allowlist in the vectorizer configuration or add a basic redaction step before sending data to external APIs.

---

## 2. Architecture & Performance

### Starvation in Vectorizer Processing
*   **Current State:** `Worker::run_once` processes vectorizers sequentially in a loop.
*   **Impact:** A vectorizer with a massive backfill will block all other vectorizers from running, even if they have `concurrency` set.
*   **Recommendation:** Parallelize the vectorizer loop. Spawn a `tokio` task for each vectorizer ID so they can poll independently.

---

## 3. Robustness & Edge Cases

### Ollama Chunk Limits
*   **Issue:** Unlike the OpenAI implementation, the `OllamaEmbedder` performs no token counting or truncation.
*   **Impact:** If a row contains text exceeding the model's context window, Ollama might return truncated embeddings, error out, or hang.
*   **Recommendation:** Implement a basic character or token-based truncation for Ollama to ensure API stability.

### Synchronous Extension Functions
*   **Issue:** `openai_embed` and `ollama_embed` in `extension/src/lib.rs` are blocking HTTP calls.
*   **Impact:** If called within a large transaction or trigger, they will stall the Postgres backend, potentially causing lock contention and timeout issues.
*   **Recommendation:** Add a warning in the documentation that these are for "out-of-band" use (e.g., manual testing) and not for production triggers.

---

## Resolved Issues

### [FIXED] Manual SQL Escaping & Potential Injection
*   **Location:** `worker/src/executor.rs`
*   **Fix:** Replaced manual string interpolation and escaping with `sqlx` query binding (`$1`, `$2`) for all data values.

### [FIXED] Missing Search Path on SECURITY DEFINER Functions
*   **Location:** `extension/sql/setup.sql` (`ai._vectorizer_create_source_trigger`)
*   **Fix:** Added `SET search_path = pg_catalog, pg_temp` to all `SECURITY DEFINER` trigger functions to prevent search path hijacking.

### [FIXED] Inefficient Database Writes
*   **Location:** `worker/src/executor.rs`
*   **Fix:** Implemented batched `DELETE` statements (using `WHERE (...) IN (...)`) and batched `INSERT` statements to reduce database round-trips.

### [FIXED] Fragile Error Classification
*   **Location:** `worker/src/errors.rs`
*   **Fix:** Replaced string-pattern matching with robust HTTP status code and structured error code classification for OpenAI and generic embedding errors.