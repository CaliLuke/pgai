# Code Review: pgai

This document outlines the findings of a thorough code review of the `pgai` project, covering security, architecture, performance, and robustness.

## 1. Security Analysis

### High Severity: Manual SQL Escaping & Potential Injection
*   **Location:** `worker/src/executor.rs` (`build_join_predicates_val`, `write_to_table`)
*   **Issue:** The worker manually escapes single quotes with `.replace("'", "''")` and interpolates identifiers (schema/table names) using `format!`.
*   **Risk:** Manual escaping is prone to bypasses (e.g., handling non-string PK types). Interpolating identifiers from database metadata is dangerous if the `ai.vectorizer` table is compromised or misconfigured.
*   **Recommendation:** 
    *   Use `sqlx` query binding (`$1`, `$2`) for **all** data values, including Primary Key values in `DELETE` and `UPDATE` statements.
    *   For identifiers (table/schema names), use the Postgres `quote_ident()` equivalent or strictly validate them against a whitelist.

### High Severity: Missing Search Path on SECURITY DEFINER Functions
*   **Location:** `extension/sql/setup.sql` (`ai._vectorizer_create_source_trigger`)
*   **Issue:** Trigger functions use `SECURITY DEFINER` but do not set an explicit `search_path`.
*   **Risk:** A malicious user could create objects in a schema they control and manipulate the `search_path` to hijack the execution of these privileged functions.
*   **Recommendation:** Add `SET search_path = pg_catalog, pg_temp` to all `SECURITY DEFINER` functions.

### Medium Severity: PII Leakage to External APIs
*   **Location:** `worker/src/models.rs` (`FormatterConfig::format`)
*   **Issue:** The `PythonTemplate` formatter allows arbitrary columns from the source table to be injected into the text chunk sent to embedding providers (OpenAI/Ollama).
*   **Risk:** Sensitive data (emails, PII, internal IDs) can be accidentally included in embeddings, violating privacy policies and exposing data to third-party providers.
*   **Recommendation:** Implement a "Safe Columns" allowlist in the vectorizer configuration or add a basic redaction step before sending data to external APIs.

### Medium Severity: Undefined Secret Retrieval Function
*   **Location:** `worker/src/embedder.rs` (`resolve_api_key`)
*   **Issue:** The worker calls `ai.reveal_secret($1)` in the database if an environment variable is missing. This function is not defined in `setup.sql`.
*   **Risk:** The worker will crash/fail if it attempts to fall back to database secrets.
*   **Recommendation:** Define a standard, secure secret management schema or integrate with `pg_vault`.

---

## 2. Architecture & Performance

### Inefficient Database Writes
*   **Current State:** `executor.rs` performs a `DELETE` followed by multiple individual `INSERT` statements for every item processed.
*   **Impact:** Massive overhead for large tables. Processing 10,000 rows with 5 chunks each results in 60,000 round-trips to the database.
*   **Recommendation:** 
    *   Use **multi-row inserts** (e.g., `INSERT INTO ... VALUES (...), (...), (...)`).
    *   Batch `DELETE` statements by PK using `WHERE (id1, id2) IN (...)`.

### Starvation in Vectorizer Processing
*   **Current State:** `Worker::run_once` processes vectorizers sequentially in a loop.
*   **Impact:** A vectorizer with a massive backfill will block all other vectorizers from running, even if they have `concurrency` set.
*   **Recommendation:** Parallelize the vectorizer loop. Spawn a `tokio` task for each vectorizer ID so they can poll independently.

### Fragile Error Classification
*   **Current State:** `worker/src/errors.rs` classifies errors as Transient or Permanent using string pattern matching (e.g., looking for "401").
*   **Impact:** API error messages are not stable. A change in the provider's wording could cause the worker to retry a permanent failure indefinitely or fail on a temporary one.
*   **Recommendation:** Capture and use the **HTTP Status Code** from `async-openai` or `reqwest` for classification (4xx vs 5xx).

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

## Summary of Recommended Fixes
1.  **Refactor SQL Generation:** Replace `format!` with proper `sqlx` bindings in the worker.
2.  **Harden PL/pgSQL:** Set `search_path` on all `SECURITY DEFINER` functions.
3.  **Optimize Worker I/O:** Batch inserts and deletes in `executor.rs`.
4.  **Parallelize Worker:** Process multiple vectorizers concurrently in `worker.rs`.
5.  **Robust Error Handling:** Switch from string-matching to status-code-based error classification.
