# Rust Test Port Plan

Systematic port of Python vectorizer tests to the Rust worker crate. Each item describes the test, what it validates, and how to implement it in Rust.

**Infra used by Python tests**: testcontainers (Postgres), VCR cassettes (HTTP replay), mitmproxy (reverse proxy), OllamaContainer, LocalStack (S3). The Rust integration test already uses testcontainers + an in-process Axum mock server, which is a good foundation.

---

## Phase 1: Test Infrastructure

These lay the groundwork for all subsequent tests.

- [ ] **1.1 — Extract reusable test helpers from integration.rs**
  - **What**: The current integration test has all setup inline (schema creation SQL, mock server, worker instantiation). Refactor into reusable helpers: `setup_test_db()`, `create_mock_embedding_server()`, `insert_vectorizer_config()`, `setup_source_table()`.
  - **Why**: Every subsequent test needs these. Without this, each test will duplicate 80+ lines of setup.
  - **How**: Create `worker/tests/common/mod.rs` with helper functions. The mock server should accept a configurable response handler so tests can return different embeddings, errors, or delays.

- [ ] **1.2 — Configurable mock embedding server**
  - **What**: Extend the mock Axum server to support: (a) configurable response payloads, (b) request recording (capture what was sent), (c) error responses (401, 429, 500), (d) request counting, (e) latency injection.
  - **Why**: Python tests validate error handling, batching behavior, and retry logic by controlling what the API returns. We need the same capability.
  - **How**: The mock server handler should read from a shared `Arc<Mutex<MockConfig>>` that tests can configure. Record requests in a `Vec<serde_json::Value>` for later assertion.

- [ ] **1.3 — Source table setup helper**
  - **What**: Helper that creates a `blog` table with `(id SERIAL PRIMARY KEY, id2 INT DEFAULT 1, content TEXT)` and populates it with N rows (`post_1`, `post_2`, ...). Matching Python's `setup_source_table()`.
  - **Why**: Nearly every CLI test uses this exact table structure.
  - **How**: A function `setup_source_table(pool, num_rows) -> Result<()>` that runs the CREATE TABLE + INSERT.

- [ ] **1.4 — Vectorizer config builder**
  - **What**: A builder/helper that constructs the JSON vectorizer config and inserts it into `ai.vectorizer` + creates the queue and destination tables. Matching Python's `configure_vectorizer()`.
  - **Why**: Each test configures different embedding providers, chunking strategies, batch sizes, etc. A builder avoids repeating the massive JSON blob.
  - **How**: Struct with builder pattern: `VectorizerConfigBuilder::new().embedding_openai("text-embedding-ada-002", 1536).chunking_character_text_splitter(800, 400).batch_size(2).build_and_insert(pool)`.

---

## Phase 2: Worker Lifecycle Tests

These test the worker's control flow, not the embedding pipeline.

- [ ] **2.1 — Worker exits cleanly when no vectorizers exist**
  - **What**: Start worker with `once=true` against a DB with pgai installed but no vectorizers configured. Worker should exit with code 0 and log "no vectorizers found".
  - **Python source**: `test_worker_no_tasks`
  - **How**: Setup DB with schema only (no vectorizer rows). Run `Worker::new(..., once=true).run()`. Assert Ok result. Optionally capture tracing output to verify log message.

- [ ] **2.2 — Worker errors when pgai is not installed**
  - **What**: Start worker against a bare Postgres (no `ai` schema). Worker should return an error indicating pgai is not installed.
  - **Python source**: `test_vectorizer_exits_with_error_when_no_ai_extension`
  - **How**: Start testcontainer, do NOT create `ai` schema. Run worker with `exit_on_error=true`. Assert error result.

- [ ] **2.3 — Worker errors when requested vectorizer IDs don't exist**
  - **What**: Start worker requesting specific vectorizer IDs that don't exist. Should error with a message like "invalid vectorizers, wanted: [0], got: []".
  - **Python source**: `test_vectorizer_exits_with_error_when_vectorizers_specified_but_missing`
  - **How**: Setup DB with schema but no vectorizer rows. Run worker with `vectorizer_ids=vec![0]` and `exit_on_error=true`. Assert error.

- [ ] **2.4 — `exit_on_error=false` suppresses errors**
  - **What**: Same scenarios as 2.2 and 2.3 but with `exit_on_error=false`. Worker should return Ok instead of Err.
  - **Python source**: `test_vectorizer_does_not_exit_with_error_when_no_ai_extension`, `test_vectorizer_does_not_exit_with_error_when_vectorizers_specified_but_missing`
  - **How**: Same setup as 2.2/2.3 but `exit_on_error=false`. Assert Ok result.

- [ ] **2.5 — Worker `once=true` processes and exits**
  - **What**: Verify that with `once=true`, the worker processes all pending items and then exits (doesn't loop).
  - **Python source**: `test_vectorizer_run_once_with_shutdown`
  - **How**: Setup 2 rows, run worker with `once=true`. Assert all embeddings created and worker returned.

- [ ] **2.6 — Disabled vectorizer is skipped**
  - **What**: Create a vectorizer, disable it via the DB flag, run worker. No embeddings should be created, queue should remain full.
  - **Python source**: `test_disabled_vectorizer_is_skipped`
  - **How**: Setup source table + vectorizer, then `UPDATE ai.vectorizer SET config = jsonb_set(config, '{scheduling,implementation}', '"disabled"')` or set the disabled flag. Run worker. Assert 0 embeddings, pending items unchanged.

---

## Phase 3: Chunking Strategy Tests

These validate text splitting logic. Many can be pure unit tests.

- [ ] **3.1 — No chunking preserves full document**
  - **What**: With `chunking: none`, each source row becomes exactly one chunk containing the full text. `chunk_seq` should be 0.
  - **Python source**: `test_chunking_none`
  - **How**: Unit test: call chunking with `None` config. Assert 1 chunk per input, content unchanged, seq=0. Integration test: run full pipeline, verify destination table has 1 row per source row with full content.

- [ ] **3.2 — Character text splitter**
  - **What**: With character text splitter (chunk_size, chunk_overlap, separator), text is split at separator boundaries respecting size limits. Already partially tested in Python via VCR cassettes.
  - **Python source**: Used across many tests as default chunking
  - **How**: Unit test with known inputs. E.g. chunk_size=100, overlap=20, verify chunk count and boundaries.

- [ ] **3.3 — Recursive character text splitter**
  - **What**: With recursive splitter, text is split using a hierarchy of separators (`\n\n`, `\n`, ` `). Should respect natural document boundaries. Chunks should have sequential `chunk_seq` values (0, 1, 2, ...).
  - **Python source**: `test_recursive_character_splitting`
  - **How**: Unit test with structured markdown content (headings + paragraphs). Assert: multiple chunks created, chunks respect paragraph boundaries, chunk_seq is sequential. Integration test: setup source table with long markdown, run worker, verify chunks in destination.

- [ ] **3.4 — Chunk size limits are respected**
  - **What**: Every chunk should be <= chunk_size (accounting for overlap).
  - **Python source**: `test_perform_chunking` (existing Rust unit test)
  - **How**: Expand existing unit test with more inputs: very long single word, text with no separators, text exactly at chunk_size boundary.

- [ ] **3.5 — Batch indices calculation**
  - **What**: Given a list of token counts, batch_size, and optional token_limit, compute correct batch boundaries.
  - **Python source**: `test_batch_indices` (7 parametrized cases)
  - **How**: Port as unit test. Cases: empty input, single batch, multiple batches, token limit splitting, error when chunk exceeds max_tokens.

---

## Phase 4: OpenAI Embedder Tests

- [ ] **4.1 — Basic OpenAI embedding end-to-end (single item, batch=1)**
  - **What**: 1 source row, batch_size=1, concurrency=1. Mock server returns embedding. Verify 1 embedding stored correctly.
  - **Python source**: `test_process_vectorizer` parametrize (1,1,1)
  - **How**: Already covered by existing integration test, but refactor to use new helpers.

- [ ] **4.2 — Multiple items with batching (4 items, batch=2, concurrency=2)**
  - **What**: 4 source rows, batch_size=2, concurrency=2. Verify all 4 embeddings stored. Verify mock server received 2 requests of 2 items each.
  - **Python source**: `test_process_vectorizer` parametrize (4,2,2)
  - **How**: Setup 4 rows, configure batch_size=2 and concurrency=2 in vectorizer config. Use request-recording mock server. Assert 4 embeddings in destination, 2 API calls made.

- [ ] **4.3 — Already-embedded rows are skipped**
  - **What**: Insert an embedding for row 1 before running the worker. Worker should not re-embed row 1, only embed remaining rows.
  - **Python source**: `test_process_vectorizer` (all variants seed an embedding for item 1 before worker runs)
  - **How**: Insert source rows, manually insert 1 embedding into destination table, run worker. Assert total embeddings = num_items, mock server only received (num_items - 1) items.

- [ ] **4.4 — Custom base_url / proxy support**
  - **What**: OpenAI embedder with a custom `base_url` in config sends requests to that URL instead of the default.
  - **Python source**: `test_process_vectorizer` parametrize with openai_proxy_url=8000
  - **How**: Start mock server on a specific port, set `base_url` in vectorizer config to that port. Verify requests arrive at mock server.

- [ ] **4.5 — Missing API key error**
  - **What**: When OPENAI_API_KEY env var is not set and not available from DB, worker should fail with an appropriate error.
  - **Python source**: `test_vectorizer_without_secrets_fails`
  - **How**: Unset OPENAI_API_KEY env var, run worker. Assert error contains "api key" or similar message.

- [ ] **4.6 — Invalid API key (401 response)**
  - **What**: When the embedding API returns 401 Unauthorized, worker should record error in `ai.vectorizer_errors` table with provider and error details.
  - **Python source**: `test_invalid_api_key_error`
  - **How**: Configure mock server to return 401 with error body. Run worker. Assert error recorded in DB with correct provider name and error reason.

- [ ] **4.7 — Invalid model configuration**
  - **What**: When embedding model config has invalid parameters (e.g. wrong dimensions for a model), worker should fail with a validation error.
  - **Python source**: `test_invalid_function_arguments`
  - **How**: Configure vectorizer with mismatched dimensions. Run worker. Assert error in `ai.vectorizer_errors`.

- [ ] **4.8 — Document exceeds model context length**
  - **What**: When a chunk is too long for the model's context window, it should still be processed (truncated or handled gracefully) rather than crashing the entire batch.
  - **Python source**: `test_document_exceeds_model_context_length`
  - **How**: Insert a very long document (e.g. 15,000 chars). Run worker. Assert embedding is created (model handles truncation) or error is recorded per-document without failing the batch.

---

## Phase 5: Ollama Embedder Tests

- [ ] **5.1 — Basic Ollama embedding end-to-end**
  - **What**: 1 source row embedded via Ollama. Verify embedding stored correctly.
  - **Python source**: `test_ollama_vectorizer` parametrize (1,1,1)
  - **How**: Use mock server mimicking Ollama's `/api/embed` endpoint (different from OpenAI format). Configure vectorizer with ollama embedding type. Assert 1 embedding stored.

- [ ] **5.2 — Multiple items with batching via Ollama**
  - **What**: 4 source rows, batch_size=2, concurrency=2 via Ollama.
  - **Python source**: `test_ollama_vectorizer` parametrize (4,2,2)
  - **How**: Same as 5.1 but with 4 rows. Assert all embeddings created.

- [ ] **5.3 — Ollama with custom base_url**
  - **What**: Ollama embedder respects `base_url` config field.
  - **How**: Set `base_url` in config to mock server address. Verify requests arrive there.

---

## Phase 6: Edge Cases and Error Handling

- [ ] **6.1 — Empty/whitespace content skips embedding**
  - **What**: Source rows with empty string, whitespace-only, or `\t\n  \t` content should not trigger any API calls. Queue should be drained (pending_items=0) but no embeddings created.
  - **Python source**: `test_empty_string_content_skips_embedding`
  - **How**: Insert 3 rows with empty/whitespace content. Run worker. Assert 0 embeddings, 0 pending items, 0 API calls to mock server.

- [ ] **6.2 — NULL content handling**
  - **What**: Source rows with NULL content should not crash the worker. No embeddings created for NULL rows.
  - **Python source**: `test_vectorization_successful_with_null_contents`
  - **How**: Create source table allowing NULLs, insert rows with NULL content. Run worker. Assert 0 embeddings, no errors.

- [ ] **6.3 — Source table has "locked" column (regression)**
  - **What**: A source table with a column named `locked` should not interfere with the worker's queue locking mechanism.
  - **Python source**: `test_regression_source_table_has_locked_column`
  - **How**: Create table with id, content, locked columns. Run worker. Assert embedding created successfully.

- [ ] **6.4 — Additional columns on target table**
  - **What**: If the destination/target table has extra columns beyond the expected ones, the worker should still insert embeddings successfully.
  - **Python source**: `test_additional_columns_are_added_to_target_table`
  - **How**: Create destination table, add an extra column. Run worker. Assert embeddings created.

- [ ] **6.5 — Same-table destination (embedding column on source table)**
  - **What**: When configured with `destination_column` instead of `destination_table`, embeddings are written to a column on the source table itself.
  - **Python source**: `test_same_table_vectorizer`
  - **How**: Configure vectorizer with column destination. Run worker. Assert source table rows now have non-NULL embedding values.

- [ ] **6.6 — Complex primary keys**
  - **What**: Source tables with multi-column or non-integer primary keys (text arrays, timestamps, etc.) should work correctly.
  - **Python source**: `test_vectorizer_weird_pk`
  - **How**: Create table with complex PK (e.g. `(text[], varchar, timestamptz)`). Insert rows. Run worker. Assert correct number of embeddings.

---

## Phase 7: Formatting Tests

- [ ] **7.1 — Default chunk formatting**
  - **What**: With default formatting (`$chunk`), the embedded text should be exactly the chunk content.
  - **How**: Run worker, capture what was sent to mock server. Assert the request body contains the raw chunk text.

- [ ] **7.2 — Template formatting with title prepended**
  - **What**: With formatting like `$title: $chunk`, the embedded text should include the title prefix.
  - **Python source**: `test_vectorizer_internal` validates chunk formatting
  - **How**: Configure vectorizer with a formatting template that references multiple columns. Verify mock server received correctly formatted text.

---

## Phase 8: Worker Tracking / Observability

- [ ] **8.1 — Worker process tracking**
  - **What**: After a run, the worker should insert/update a row in `ai.vectorizer_worker_process` with: started timestamp, last_heartbeat, heartbeat_count, error_count, success_count.
  - **Python source**: `test_process_vectorizer` checks worker_process table
  - **How**: Run worker, query `ai.vectorizer_worker_process`. Assert all fields populated correctly.

- [ ] **8.2 — Worker progress tracking**
  - **What**: After a run, `ai.vectorizer_worker_progress` should reflect: last_success_at, success_count, and error fields should be NULL for successful runs.
  - **Python source**: `test_process_vectorizer` checks worker_progress table
  - **How**: Run worker, query progress table. Assert success fields populated, error fields NULL.

- [ ] **8.3 — Error tracking on failure**
  - **What**: When an embedding API call fails, error should be recorded in `ai.vectorizer_errors` with vectorizer_id, message, and JSON details (provider, error_reason).
  - **Python source**: `test_invalid_api_key_error`, `test_vectorizer_without_secrets_fails`
  - **How**: Configure mock to return errors. Run worker. Query errors table. Assert correct error structure.

---

## Phase 9: Compatibility and Upgrade Tests

- [ ] **9.1 — Works with older pgai schema versions**
  - **What**: Worker should handle vectorizer configs written by older versions of pgai (e.g., missing new fields, old field names).
  - **Python source**: `test_080_vectorizer_definition`
  - **How**: Insert a vectorizer config in the old format (v0.8.0 style). Run worker. Assert it processes correctly or gives a clear error.

---

## Priority Order

For maximum value with minimum effort:

1. **Phase 1** (infra) — Required for everything else
2. **Phase 2** (lifecycle) — Low-hanging fruit, tests worker control flow
3. **Phase 4.1-4.3** (basic OpenAI) — Core happy path
4. **Phase 3.1, 3.3** (chunking none + recursive) — Already have unit test foundation
5. **Phase 6.1-6.2** (empty/null content) — Important edge cases
6. **Phase 4.5-4.6** (API errors) — Error handling validation
7. **Phase 5** (Ollama) — Second embedder
8. **Phase 6.3-6.6** (remaining edge cases)
9. **Phase 7-8** (formatting, tracking)
10. **Phase 9** (compatibility)
