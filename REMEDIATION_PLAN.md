# Remediation Plan: High Priority Findings

This plan addresses the two **High Severity** security findings and the most impactful **performance/reliability** issues from `CODE_REVIEW.md`.

---

## 1. Eliminate SQL injection in `executor.rs`

**Finding:** `build_join_predicates_val` uses `.replace("'", "''")` and interpolates the result directly into DELETE/UPDATE queries. This is used in `write_to_table` (line 268) and `write_to_column` (line 309).

**Root cause:** The function builds a WHERE clause as a raw string because PK values come from `serde_json::Value` with unknown arity — you can't statically bind N parameters when N varies per vectorizer.

### SQL injection fix approach

Replace `build_join_predicates_val` with a method that returns _both_ a parameterized WHERE clause and the bind values, then apply them via sqlx's `query` binding.

### SQL injection fix changes

**`worker/src/executor.rs`**

1. Delete `build_join_predicates_val` entirely.
1. Add a new helper that returns a parameterized clause:

```rust
/// Returns (where_clause, bind_values) with placeholders starting at $offset.
/// Example: ("\"id\" = $3 AND \"tenant\" = $4", [Value::Number(1), Value::String("acme")])
fn build_pk_predicates(&self, item: &serde_json::Value, offset: usize)
    -> (String, Vec<&serde_json::Value>)
{
    let mut parts = Vec::new();
    let mut values = Vec::new();
    for (i, pk) in self.vectorizer.source_pk.iter().enumerate() {
        parts.push(format!("\"{}\" = ${}", pk.attname, offset + i));
        values.push(item.get(&pk.attname).unwrap());
    }
    (parts.join(" AND "), values)
}
```

1. Refactor `write_to_table` DELETE to use bind params:

```rust
// Before (vulnerable):
let delete_query = format!(
    "DELETE FROM \"{}\".\"{}\" WHERE {}",
    schema, table, self.build_join_predicates_val(item)
);
sqlx::query(&delete_query).execute(&self.pool).await?;

// After (safe):
let (where_clause, pk_vals) = self.build_pk_predicates(item, 1);
let delete_query = format!(
    "DELETE FROM \"{}\".\"{}\" WHERE {}",
    schema, table, where_clause
);
let mut q = sqlx::query(&delete_query);
for val in &pk_vals {
    q = bind_json_value(q, val);
}
q.execute(&self.pool).await?;
```

1. Refactor `write_to_column` UPDATE similarly — the embedding is `$1`, PK params start at `$2`.
1. Add a small `bind_json_value` helper to centralize the `Number/String/_` match arm that already exists in the INSERT loop (lines 279-283), avoiding duplication.

### SQL injection fix testing

- Existing unit tests still pass (they don't hit the SQL path).
- Integration tests cover the full write path — run them to confirm no regressions.
- Add a unit test with a PK value containing `'; DROP TABLE --` to verify it's properly bound (will need a mock or test DB).

---

## 2. Set `search_path` on SECURITY DEFINER trigger functions

**Finding:** `_vectorizer_create_source_trigger` (setup.sql line 541-545) creates trigger functions with `SECURITY DEFINER` but no explicit `search_path`, enabling search_path hijacking.

### Search path fix approach

Append `SET search_path = pg_catalog, pg_temp` to the trigger function definition.

### Search path fix changes

**`extension/sql/setup.sql`** — in `_vectorizer_create_source_trigger`, change:

```sql
-- Before:
LANGUAGE plpgsql VOLATILE SECURITY DEFINER

-- After:
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
```

However, the trigger body itself references user tables (`%I.%I` for queue, target, source). These references are already fully qualified via `format('%I.%I', ...)` in `_vectorizer_build_trigger_definition`, so restricting `search_path` to `pg_catalog, pg_temp` is safe — no unqualified table references exist in the generated trigger body.

### Search path fix testing

- Create a vectorizer, inspect the generated trigger function's `proconfig` to confirm `search_path` is set.
- Verify INSERT/UPDATE/DELETE on the source table still correctly populates the queue.

---

## 3. Robust error classification in `errors.rs`

**Finding:** `EmbeddingError::classify` matches substrings like `"401"`, `"429"` in `err.to_string()`. This is fragile — a chunk of text containing "401 Main Street" would be misclassified as permanent.

### Error classification approach

Introduce a structured error type that captures the HTTP status code when available, and fall back to string matching only as a last resort.

### Error classification changes

**`worker/src/errors.rs`**

1. Add a new constructor that takes an optional status code:

```rust
impl EmbeddingError {
    /// Classify using HTTP status code when available.
    pub fn from_status(status: u16, err: anyhow::Error) -> Self {
        match status {
            401 | 403 | 400 | 404 | 422 => Self::Permanent(err),
            429 | 500 | 502 | 503 | 504 => Self::Transient(err),
            _ if status >= 400 && status < 500 => Self::Permanent(err),
            _ => Self::Transient(err),
        }
    }
}
```

1. Keep `classify` as a fallback for errors without status codes (e.g., DNS failures, connection resets), but make the string patterns more specific to reduce false positives (e.g., `"http 401"` instead of `"401"`).

**`worker/src/embedder.rs`**

1. In `OpenAIEmbedder::embed`, extract the status code from `async_openai`'s error type before converting to `anyhow::Error`. The `async_openai::error::OpenAIError` enum has an `ApiError` variant with a status field — match on it:

```rust
.map_err(|e| match &e {
    async_openai::error::OpenAIError::ApiError(api_err) => {
        // api_err.message contains details, but we use the HTTP status
        let status = api_err.status_code();
        EmbeddingError::from_status(status, e.into())
    }
    _ => EmbeddingError::classify(e.into()),
})
```

1. For `OllamaEmbedder`, the `ollama-rs` error type is less structured — keep `classify` as the fallback there.

### Error classification testing

- Update existing tests to use `from_status` where applicable.
- Add tests: `from_status(401, ..)` is Permanent, `from_status(503, ..)` is Transient.
- Add a regression test: an error message containing "401" in the body text but with a 200 status should NOT be classified as Permanent.

---

## 4. Batch database writes in `executor.rs`

**Finding:** `write_to_table` performs one DELETE + N INSERTs per source row. For 100 rows with 5 chunks each, that's 600 round-trips.

### Batch writes approach

Batch DELETEs and INSERTs per `do_batch` call rather than per item.

### Batch writes changes

**`worker/src/executor.rs`**

1. **Batch DELETE**: Collect all PK values, then issue a single DELETE:

```rust
// Single DELETE for all items in the batch
// DELETE FROM "schema"."table" WHERE ("id") IN (($1), ($2), ...)
let pk_count = self.vectorizer.source_pk.len();
let mut placeholders = Vec::new();
let mut all_pk_vals = Vec::new();
for (i, item) in items.iter().enumerate() {
    let offset = i * pk_count + 1;
    let group: Vec<String> = (0..pk_count)
        .map(|j| format!("${}", offset + j))
        .collect();
    placeholders.push(format!("({})", group.join(", ")));
    for pk in &self.vectorizer.source_pk {
        all_pk_vals.push(item.get(&pk.attname).unwrap());
    }
}
let delete_sql = format!(
    "DELETE FROM \"{}\".\"{}\" WHERE ({}) IN ({})",
    schema, table, pk_list, placeholders.join(", ")
);
```

1. **Batch INSERT**: Build a multi-row INSERT with all chunks:

```rust
// INSERT INTO "schema"."table" (pk_cols, chunk_seq, chunk, embedding)
// VALUES ($1,$2,$3,$4), ($5,$6,$7,$8), ...
```

Postgres has a bind parameter limit of ~65535. With 4 columns per row (1 PK + seq + chunk + embedding), that's ~16K rows per statement — well above typical batch sizes. Add a safety check to split into sub-batches if needed.

1. Wrap the batch DELETE + INSERT in a transaction (the current code doesn't use an explicit transaction — items could end up in an inconsistent state if the process crashes mid-write).

### Batch writes testing

- Integration tests cover the write path end-to-end.
- Add a unit test that verifies the generated SQL has the expected number of parameter placeholders for a given batch size.

---

## Execution Order

| Priority | Item                              | Effort | Risk                                               |
| -------- | --------------------------------- | ------ | -------------------------------------------------- |
| 1        | SQL injection fix (#1)            | Medium | Low — swap interpolation for binds, same logic     |
| 2        | SECURITY DEFINER search_path (#2) | Small  | Very low — one-line addition                       |
| 3        | Error classification (#3)         | Medium | Low — additive, keeps fallback                     |
| 4        | Batch writes (#4)                 | Large  | Medium — changes write path, needs careful testing |

Items 1 and 2 are security fixes and should be done first. Item 3 improves operational reliability. Item 4 is a performance optimization that can follow once the security issues are resolved.
