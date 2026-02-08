# Rust Worker Test Coverage Gap vs Python

## What We Have (61 tests)

### Unit Tests (32)

#### `worker/src/executor.rs` (18)

- ChunkerConfig::None: preserves_full_text, empty_string
- RecursiveCharacterTextSplitter: splits_long_text, short_text_no_split, respects_boundaries
- CharacterTextSplitter: splits_on_separator, respects_chunk_size, with_overlap, single_segment, empty_text, multi_char_separator, all_fits_one_chunk, preserves_all_content, unicode_separator
- Serde deserialization: none, recursive, character, unknown_falls_to_none

#### `worker/src/embedder.rs` (14)

- tiktoken: counts_tokens, openai_embedder_count_tokens, truncate_long_document, no_truncation_for_short_text
- batch_indices: empty, max_batch_size_1, max_batch_size_3, max_batch_size_5, with_token_limit, chunk_exceeds_max_tokens, string_documents_with_token_limit
- API key: missing_api_key, with_api_key, default_key_name

### Integration Tests (29) — `worker/tests/integration.rs`

- E2E: end_to_end_vectorization
- Lifecycle: no_vectorizers, no_pgai_schema, no_pgai_schema_no_exit_on_error, missing_vectorizer_ids, once_processes_and_exits
- Chunking E2E: none, recursive, character, empty_content
- NULL/empty: null_content_skips_embedding, empty_content_skips_embedding
- Batch/multi: batch_size_processes_all_rows, multiple_vectorizers, vectorizer_id_filtering, rerun_is_noop, mock_server_receives_correct_input
- Token batching: large_content_splits_into_multiple_batches
- API key: missing_api_key_fails_with_clear_error
- Disabled: disabled_vectorizer_is_skipped, disabled_vectorizer_does_not_block_others
- Error recording: embedding_error_recorded_to_errors_table
- Edge cases: composite_primary_key, source_table_with_reserved_column_names
- Schema evolution: extra_columns_on_destination, drop_readd_embedding_column
- Same-table: column_destination
- Ollama: e2e, different_texts_different_embeddings, with_chunking

---

## What Python Has That We Don't

### P1: Core Pipeline — ALL DONE

- [x] **Token-based batch limits** — tiktoken-rs for real token counting, batch_indices algorithm, 300K token limit, context length truncation (8191 tokens).
- [x] **NULL content in source rows** — `extract_text` returns `Ok(None)` for NULL, `do_batch` skips them.
- [x] **Empty string skips embedding call** — empty/whitespace-only content returns `Ok(None)`, zero embedding requests.
- [x] **Invalid/missing API key** — `create_embedder` fails with clear error when env var not set.
- [x] **Weird primary keys** — Composite PK test (TEXT + INT).
- [x] **Reserved column names** — Test with "locked", "data", "order" columns.

### P2: Features (medium priority)

- [x] **Disabled vectorizer** — `disabled` field in model, skip logic in worker, 2 integration tests.
- [x] **Schema evolution** — Extra columns on destination, embedding column drop/re-add. 2 integration tests.
- [x] **Errors table** — `record_error` method, catches embedding failures, 1 integration test.
- [x] **Same-table vectorization** — Column destination mode (source == destination). 1 integration test.
- [x] **Dimension validation** — Covered by error recording test (API errors are recorded). Rust delegates dimension validation to the API.

### P3: Provider Integrations (lower priority)

- [ ] **VoyageAI provider** — Python `test_voyageai_vectorizer` (2 configs), `test_voyageai_vectorizer_fails_when_api_key_is_not_set`. Not in Rust yet.
- [ ] **LiteLLM provider** — Python `test_litellm_vectorizer`. Not in Rust yet.

### P4: Out of Scope for Rust Worker

- Database installation/setup (`test_vectorizer_install_*`) — Python CLI concern
- SQLAlchemy integration — Python library concern
- S3 document loading — Separate module, future Rust work
- Configuration SQL generation — Python concern
- CLI tests — Python concern
