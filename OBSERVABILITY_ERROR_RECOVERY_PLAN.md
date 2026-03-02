# Observability, Error Recovery, and Graceful Failure Plan

This document tracks remediation work for issues identified in the code review.
Each item is a checklist task with a problem description and concrete fixes.

- [ ] **Propagate concurrent executor failures instead of swallowing them**
  - **Issue:** In concurrent vectorizer execution, executor errors and panics are logged but not returned, so `run_once()` can report success even when a vectorizer failed. This undermines `exit_on_error` and operational correctness.
  - **Remediation:**
    - Update `process_vectorizer()` to accumulate the first executor failure and return `Err(...)` after joining tasks.
    - Treat `JoinError` panics as failures and convert them to `anyhow::Error` with vectorizer context.
    - Add integration tests for `concurrency > 1` that verify failure propagation with `exit_on_error=true`.
    - Keep non-failing executors cancellable once one fails (abort remaining tasks when policy requires fail-fast).

- [ ] **Record and propagate vectorizer task panics as first-class errors**
  - **Issue:** Panics in vectorizer tasks are only logged; they are not consistently counted in worker tracking and may not fail the loop.
  - **Remediation:**
    - In `run_once()`, on `JoinError`, call `save_vectorizer_error(...)` with a panic-specific message.
    - Set `first_error` on panic so top-level worker behavior matches policy (`exit_on_error`).
    - Include structured panic metadata in logs (`vectorizer_id`, panic kind, cancelled vs panic).
    - Add a regression test that forces a panic path and asserts error recording + loop result.

- [ ] **Remove panic-prone runtime/config paths and return actionable errors**
  - **Issue:** Several `unwrap`/`expect` calls in runtime/configuration paths can terminate the worker instead of returning recoverable errors with context.
  - **Remediation:**
    - Replace `expect`/`unwrap` in non-test code with `Result` propagation and `context(...)`.
    - Harden Ollama URL parsing and host extraction with explicit validation errors.
    - Replace unsafe PK extraction `unwrap()` paths with checked errors including vectorizer id + PK name.
    - OTLP initialization should degrade gracefully: log warning and continue without tracing exporter.
    - Add unit tests for invalid URL/config/input cases to ensure graceful failure.

- [ ] **Preserve database-secret resolution errors instead of collapsing to “not found”**
  - **Issue:** `resolve_api_key()` currently drops DB query errors and reports a generic missing secret, masking root cause (permissions/connectivity/query issues).
  - **Remediation:**
    - Return DB errors with context (secret name, source = `ai.reveal_secret`) instead of `.ok().flatten()`.
    - Keep “not found” only for true empty results.
    - Add structured log fields for secret source (`env` vs `db`) and failure reason category.
    - Add integration tests covering DB error path vs missing-key path.

- [ ] **Improve heartbeat failure handling so observability loss is visible and actionable**
  - **Issue:** Heartbeat feature detection and loop failures can silently disable tracking; the worker may continue while telemetry is effectively blind.
  - **Remediation:**
    - Do not silently coerce startup table-check query failures to “table missing”; log and classify them.
    - Emit a one-time high-severity log/metric when heartbeat loop stops after consecutive failures.
    - Expose heartbeat health state to main worker loop (shared flag/channel) so it can optionally fail or alert.
    - Add tests for heartbeat DB outage scenarios and verify counter restoration + visibility.

- [ ] **Standardize structured error telemetry for retries and provider failures**
  - **Issue:** Retry and failure logs are mostly free-form strings, limiting aggregation, alerting, and root-cause analysis.
  - **Remediation:**
    - Add structured fields to retry/failure logs: `vectorizer_id`, `provider`, `model`, `error_class` (`transient|permanent`), `attempt`, `max_attempts`, `status_code`, `error_code`.
    - Emit a final summary event per failed batch with attempts exhausted and elapsed duration.
    - Ensure OpenTelemetry spans include these attributes where tracing is enabled.
    - Add a logging-focused test (or snapshot assertion) for key failure paths.

- [ ] **Close coverage gaps for concurrent failure semantics and graceful degradation**
  - **Issue:** Tests cover throughput and mixed success scenarios, but not strict failure semantics under concurrency and observability degradation paths.
  - **Remediation:**
    - Add integration tests for:
      - concurrent executor failure with `exit_on_error=true` should return `Err`
      - concurrent executor failure with `exit_on_error=false` should continue and record error
      - join panic path should be recorded and policy-respected
      - heartbeat failure stop should be visible in logs/state
    - Add unit tests for error classification edge cases and panic-proof paths.
    - Gate merges on these scenarios to prevent regressions.

