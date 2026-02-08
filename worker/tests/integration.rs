mod common;

use common::*;
use sqlx::{Pool, Postgres, Row};
use std::time::Duration;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres as PostgresImage;
use tokio_util::sync::CancellationToken;
use worker::Worker;

async fn start_postgres() -> (testcontainers::ContainerAsync<PostgresImage>, Pool<Postgres>) {
    let node = PostgresImage::default().start().await.unwrap();
    let port = node.get_host_port_ipv4(5432).await.unwrap();
    let db_url = format!("postgres://postgres:postgres@localhost:{}/postgres", port);
    let pool = Pool::<Postgres>::connect(&db_url).await.unwrap();
    (node, pool)
}

fn db_url_from_port(port: u16) -> String {
    format!("postgres://postgres:postgres@localhost:{}/postgres", port)
}

// ============================================================
// Step 3: Refactored existing end-to-end test using helpers
// ============================================================

#[tokio::test]
async fn test_end_to_end_vectorization() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;
    setup_source_table(&pool, "notes", 2).await;
    setup_queue_table(&pool, "ai", "notes_queue", 2).await;
    setup_destination_table(&pool, "public", "notes_embeddings").await;

    let (mock_url, _log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("notes")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.notes_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 2);

    let first_embedding: Vec<f32> =
        sqlx::query_scalar("SELECT embedding FROM public.notes_embeddings WHERE id = 1")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(first_embedding.len(), 3);
}

// ============================================================
// Step 4: Worker lifecycle integration tests
// ============================================================

#[tokio::test]
async fn test_worker_no_vectorizers() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;
    // No vectorizer rows inserted

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();

    let result = worker.run().await;
    assert!(result.is_ok(), "Worker should succeed with no vectorizers");
}

#[tokio::test]
async fn test_worker_no_pgai_schema() {
    let (node, _pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();
    // No ai schema created — bare postgres

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true, // exit_on_error = true
        CancellationToken::new(),
    )
    .await
    .unwrap();

    let result = worker.run().await;
    assert!(result.is_err(), "Worker should error when pgai not installed");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("not installed") || err_msg.contains("does not exist") || err_msg.contains("ai"),
        "Error should mention pgai not installed, got: {err_msg}"
    );
}

#[tokio::test]
async fn test_worker_no_pgai_schema_no_exit_on_error() {
    let (node, _pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();
    // No ai schema created

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        false, // exit_on_error = false
        CancellationToken::new(),
    )
    .await
    .unwrap();

    let result = worker.run().await;
    assert!(
        result.is_ok(),
        "Worker should not error when exit_on_error=false, got: {:?}",
        result
    );
}

#[tokio::test]
async fn test_worker_missing_vectorizer_ids() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;
    // Request vectorizer_id=999 which doesn't exist

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![999],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();

    let result = worker.run().await;
    assert!(
        result.is_ok(),
        "Worker should succeed (warn) for missing vectorizer IDs"
    );
}

#[tokio::test]
async fn test_worker_once_processes_and_exits() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;
    setup_source_table(&pool, "posts", 2).await;
    setup_queue_table(&pool, "ai", "posts_queue", 2).await;
    setup_destination_table(&pool, "public", "posts_embeddings").await;

    let (mock_url, _log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("posts")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true, // once = true
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();

    // Worker should process and return
    worker.run().await.unwrap();

    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.posts_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 2, "Expected 2 embeddings");

    // Queue should be drained
    let queue_count: i64 = sqlx::query_scalar("SELECT count(*) FROM ai.posts_queue")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(queue_count, 0, "Queue should be empty after processing");
}

// ============================================================
// Step 6: Chunking integration tests
// ============================================================

#[tokio::test]
async fn test_chunking_none_end_to_end() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;
    setup_source_table(&pool, "articles", 2).await;
    setup_queue_table(&pool, "ai", "articles_queue", 2).await;
    setup_destination_table(&pool, "public", "articles_embeddings").await;

    let (mock_url, _log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("articles")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    // Should have exactly 2 rows, each with chunk_seq=0 and full content
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.articles_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 2);

    let rows = sqlx::query("SELECT id, chunk_seq, chunk FROM public.articles_embeddings ORDER BY id")
        .fetch_all(&pool)
        .await
        .unwrap();

    for row in &rows {
        let chunk_seq: i32 = row.get("chunk_seq");
        assert_eq!(chunk_seq, 0, "chunk_seq should be 0 for chunking=none");

        let chunk: &str = row.get("chunk");
        assert!(!chunk.is_empty(), "chunk should contain full content");
    }
}

#[tokio::test]
async fn test_recursive_chunking_end_to_end() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;

    // Create source table with one row of long content
    let long_content = "This is a paragraph about Rust programming. ".repeat(20);
    setup_source_table_with_content(&pool, "docs", &[&long_content]).await;
    setup_queue_table(&pool, "ai", "docs_queue", 1).await;
    setup_destination_table(&pool, "public", "docs_embeddings").await;

    let (mock_url, _log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("docs")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_recursive(100, 0)
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    let rows = sqlx::query("SELECT chunk_seq, chunk FROM public.docs_embeddings ORDER BY chunk_seq")
        .fetch_all(&pool)
        .await
        .unwrap();

    assert!(
        rows.len() > 1,
        "Expected multiple chunks for long content, got {}",
        rows.len()
    );

    // Verify sequential chunk_seq values
    for (i, row) in rows.iter().enumerate() {
        let chunk_seq: i32 = row.get("chunk_seq");
        assert_eq!(chunk_seq, i as i32, "chunk_seq should be sequential");

        let chunk: &str = row.get("chunk");
        assert!(
            chunk.chars().count() <= 100,
            "Chunk {} exceeded 100 chars: {} chars",
            i,
            chunk.chars().count()
        );
    }
}

#[tokio::test]
async fn test_empty_content_skips_embedding() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;
    setup_source_table_with_content(&pool, "empty_docs", &["", "   ", ""]).await;
    setup_queue_table(&pool, "ai", "empty_docs_queue", 3).await;
    setup_destination_table(&pool, "public", "empty_docs_embeddings").await;

    let (mock_url, request_log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("empty_docs")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();

    let result = worker.run().await;
    assert!(result.is_ok(), "Worker should handle empty content: {:?}", result);

    // No embeddings should be created for empty/whitespace-only content
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.empty_docs_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 0, "Empty content should produce zero embeddings");

    // Queue should be drained (items processed even if skipped)
    let queue_count: i64 = sqlx::query_scalar("SELECT count(*) FROM ai.empty_docs_queue")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(queue_count, 0, "Queue should be empty after processing");

    // No requests should have been sent to the embedding server
    let requests = request_log.lock().unwrap();
    assert_eq!(requests.len(), 0, "No embedding requests for empty content");
}

#[tokio::test]
async fn test_null_content_skips_embedding() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;

    // Create source table with mix of NULL and real content
    sqlx::query("CREATE TABLE public.null_docs (id SERIAL PRIMARY KEY, content TEXT)")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO public.null_docs (content) VALUES (NULL)")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO public.null_docs (content) VALUES ('real content here')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO public.null_docs (content) VALUES (NULL)")
        .execute(&pool)
        .await
        .unwrap();

    setup_queue_table(&pool, "ai", "null_docs_queue", 3).await;
    setup_destination_table(&pool, "public", "null_docs_embeddings").await;

    let (mock_url, request_log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("null_docs")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();

    worker.run().await.unwrap();

    // Only 1 embedding for the real content row (id=2)
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.null_docs_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 1, "Only non-NULL content should produce embeddings");

    // The embedded row should be the one with real content
    let chunk: String = sqlx::query_scalar("SELECT chunk FROM public.null_docs_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(chunk, "real content here");

    // Queue should be fully drained
    let queue_count: i64 = sqlx::query_scalar("SELECT count(*) FROM ai.null_docs_queue")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(queue_count, 0, "Queue should be empty");

    // Only 1 embedding request (for the non-NULL row)
    let requests = request_log.lock().unwrap();
    assert_eq!(requests.len(), 1, "Should have exactly 1 embedding request");
}

// ============================================================
// Token-based batching tests
// ============================================================

#[tokio::test]
async fn test_large_content_splits_into_multiple_batches() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;

    // Create many rows with moderate content — enough to exceed 300K tokens total
    // Each "word " is ~1 token, so 1000 words ≈ 1000 tokens per row.
    // 400 rows × 1000 tokens = 400K tokens → should need at least 2 batches.
    let content = "word ".repeat(1000);
    let rows: Vec<&str> = (0..400).map(|_| content.as_str()).collect();
    setup_source_table_with_content(&pool, "big_batch", &rows).await;
    setup_queue_table(&pool, "ai", "big_batch_queue", 400).await;
    setup_destination_table(&pool, "public", "big_batch_embeddings").await;

    let (mock_url, request_log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("big_batch")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.big_batch_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 400, "All 400 rows should be embedded");

    // Should have made multiple embedding requests due to token limits
    let requests = request_log.lock().unwrap();
    assert!(
        requests.len() >= 2,
        "Expected at least 2 embedding requests for 400K+ tokens, got {}",
        requests.len()
    );
}

// ============================================================
// API key validation tests
// ============================================================

#[tokio::test]
async fn test_missing_api_key_fails_with_clear_error() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;
    setup_source_table(&pool, "nokey", 1).await;
    setup_queue_table(&pool, "ai", "nokey_queue", 1).await;
    setup_destination_table(&pool, "public", "nokey_embeddings").await;

    let (mock_url, _log) = start_mock_embedding_server().await;

    // Use a unique key name that no other test touches, avoiding env var races
    let unique_key = "NOKEY_TEST_MISSING_KEY_12345";
    std::env::remove_var(unique_key);

    let mut builder = VectorizerConfigBuilder::new("nokey")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none();
    builder.set_api_key_name(unique_key);
    builder.insert(&pool).await;

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true, // exit_on_error
        CancellationToken::new(),
    )
    .await
    .unwrap();

    let result = worker.run().await;
    assert!(result.is_err(), "Should fail when API key not set");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains(unique_key) && err.contains("not found"),
        "Error should mention the missing key name, got: {err}",
    );
}

// ============================================================
// Error recording tests
// ============================================================

#[tokio::test]
async fn test_embedding_error_recorded_to_errors_table() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;
    setup_source_table(&pool, "err_test", 2).await;
    setup_queue_table(&pool, "ai", "err_test_queue", 2).await;
    setup_destination_table(&pool, "public", "err_test_embeddings").await;

    let (mock_url, _log) = start_failing_mock_embedding_server().await;

    let vid = VectorizerConfigBuilder::new("err_test")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        false, // exit_on_error = false so we don't crash
        CancellationToken::new(),
    )
    .await
    .unwrap();

    // Worker should continue despite the error (exit_on_error=false)
    worker.run().await.unwrap();

    // Error should be recorded in vectorizer_errors
    let error_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM ai.vectorizer_errors WHERE id = $1")
            .bind(vid)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert!(
        error_count > 0,
        "Should have recorded at least one error, got {}",
        error_count
    );

    // Verify error content
    let error_row = sqlx::query(
        "SELECT message, details FROM ai.vectorizer_errors WHERE id = $1 ORDER BY recorded LIMIT 1",
    )
    .bind(vid)
    .fetch_one(&pool)
    .await
    .unwrap();

    let message: &str = error_row.get("message");
    assert_eq!(message, "embedding provider failed");

    let details: serde_json::Value = error_row.get("details");
    assert!(
        details.get("error_reason").is_some(),
        "Details should contain error_reason, got: {:?}",
        details
    );
}

// ============================================================
// Multi-column / weird primary key tests
// ============================================================

#[tokio::test]
async fn test_composite_primary_key() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;

    // Create source table with composite PK (category TEXT, seq INT)
    sqlx::query(
        "CREATE TABLE public.composite_pk (
            category TEXT NOT NULL,
            seq INT NOT NULL,
            content TEXT,
            PRIMARY KEY (category, seq)
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query("INSERT INTO public.composite_pk VALUES ('a', 1, 'first item')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO public.composite_pk VALUES ('a', 2, 'second item')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO public.composite_pk VALUES ('b', 1, 'third item')")
        .execute(&pool)
        .await
        .unwrap();

    // Queue table must have matching PK columns
    sqlx::query("CREATE TABLE ai.composite_pk_queue (category TEXT, seq INT)")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO ai.composite_pk_queue VALUES ('a', 1), ('a', 2), ('b', 1)")
        .execute(&pool)
        .await
        .unwrap();

    // Destination table also needs the PK columns
    sqlx::query(
        "CREATE TABLE public.composite_pk_embeddings (
            category TEXT, seq INT, chunk_seq INT, chunk TEXT, embedding FLOAT4[]
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    let (mock_url, _log) = start_mock_embedding_server().await;

    let pk_info = serde_json::json!([
        {"attname": "category", "pknum": 1, "attnum": 1},
        {"attname": "seq", "pknum": 2, "attnum": 2}
    ]);

    VectorizerConfigBuilder::new("composite_pk")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .source_pk(pk_info)
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    let count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM public.composite_pk_embeddings")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count, 3, "All 3 rows with composite PK should be embedded");

    // Verify each row has correct PK values preserved
    let rows = sqlx::query(
        "SELECT category, seq, chunk FROM public.composite_pk_embeddings ORDER BY category, seq",
    )
    .fetch_all(&pool)
    .await
    .unwrap();

    let categories: Vec<&str> = rows.iter().map(|r| r.get("category")).collect();
    let seqs: Vec<i32> = rows.iter().map(|r| r.get("seq")).collect();
    assert_eq!(categories, vec!["a", "a", "b"]);
    assert_eq!(seqs, vec![1, 2, 1]);

    // Queue should be drained
    let queue_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM ai.composite_pk_queue")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(queue_count, 0);
}

// ============================================================
// Disabled vectorizer tests
// ============================================================

#[tokio::test]
async fn test_disabled_vectorizer_is_skipped() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;
    setup_source_table(&pool, "skip_me", 2).await;
    setup_queue_table(&pool, "ai", "skip_me_queue", 2).await;
    setup_destination_table(&pool, "public", "skip_me_embeddings").await;

    let (mock_url, request_log) = start_mock_embedding_server().await;

    let vid = VectorizerConfigBuilder::new("skip_me")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    // Disable the vectorizer
    sqlx::query("UPDATE ai.vectorizer SET disabled = true WHERE id = $1")
        .bind(vid)
        .execute(&pool)
        .await
        .unwrap();

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    // No embeddings should be created
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.skip_me_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 0, "Disabled vectorizer should produce zero embeddings");

    // Queue should NOT be drained (vectorizer was skipped entirely)
    let queue_count: i64 = sqlx::query_scalar("SELECT count(*) FROM ai.skip_me_queue")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(queue_count, 2, "Queue should remain untouched for disabled vectorizer");

    // No embedding requests
    let requests = request_log.lock().unwrap();
    assert_eq!(requests.len(), 0, "No requests for disabled vectorizer");
}

#[tokio::test]
async fn test_disabled_vectorizer_does_not_block_others() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;

    // Disabled vectorizer
    setup_source_table(&pool, "disabled_v", 2).await;
    setup_queue_table(&pool, "ai", "disabled_v_queue", 2).await;
    setup_destination_table(&pool, "public", "disabled_v_embeddings").await;

    // Enabled vectorizer
    setup_source_table(&pool, "enabled_v", 3).await;
    setup_queue_table(&pool, "ai", "enabled_v_queue", 3).await;
    setup_destination_table(&pool, "public", "enabled_v_embeddings").await;

    let (mock_url, _log) = start_mock_embedding_server().await;

    let disabled_id = VectorizerConfigBuilder::new("disabled_v")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    VectorizerConfigBuilder::new("enabled_v")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    sqlx::query("UPDATE ai.vectorizer SET disabled = true WHERE id = $1")
        .bind(disabled_id)
        .execute(&pool)
        .await
        .unwrap();

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    let disabled_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM public.disabled_v_embeddings")
            .fetch_one(&pool)
            .await
            .unwrap();
    let enabled_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM public.enabled_v_embeddings")
            .fetch_one(&pool)
            .await
            .unwrap();

    assert_eq!(disabled_count, 0, "Disabled vectorizer should have 0 embeddings");
    assert_eq!(enabled_count, 3, "Enabled vectorizer should have 3 embeddings");
}

// ============================================================
// Reserved column names / edge cases
// ============================================================

#[tokio::test]
async fn test_source_table_with_reserved_column_names() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;

    // Create a source table with columns named "locked", "data", "order" — words used
    // internally by the worker's CTE (locked_items.locked, to_jsonb data, ORDER BY)
    sqlx::query(
        r#"CREATE TABLE public.reserved_cols (
            id SERIAL PRIMARY KEY,
            content TEXT,
            "locked" BOOLEAN DEFAULT false,
            "data" TEXT DEFAULT 'extra',
            "order" INT DEFAULT 0
        )"#,
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query("INSERT INTO public.reserved_cols (content) VALUES ('test row one')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO public.reserved_cols (content) VALUES ('test row two')")
        .execute(&pool)
        .await
        .unwrap();

    setup_queue_table(&pool, "ai", "reserved_cols_queue", 2).await;
    setup_destination_table(&pool, "public", "reserved_cols_embeddings").await;

    let (mock_url, _log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("reserved_cols")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    let count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM public.reserved_cols_embeddings")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count, 2, "Should embed despite reserved column names in source table");
}

// ============================================================
// Additional integration tests
// ============================================================

#[tokio::test]
async fn test_character_chunking_end_to_end() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;

    let content = "line one\nline two\nline three\nline four\nline five";
    setup_source_table_with_content(&pool, "lines", &[content]).await;
    setup_queue_table(&pool, "ai", "lines_queue", 1).await;
    setup_destination_table(&pool, "public", "lines_embeddings").await;

    let (mock_url, _log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("lines")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_character(20, 0, "\n")
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    let rows = sqlx::query("SELECT chunk_seq, chunk FROM public.lines_embeddings ORDER BY chunk_seq")
        .fetch_all(&pool)
        .await
        .unwrap();

    assert!(rows.len() > 1, "Expected multiple chunks, got {}", rows.len());

    for (i, row) in rows.iter().enumerate() {
        let chunk_seq: i32 = row.get("chunk_seq");
        assert_eq!(chunk_seq, i as i32);

        let chunk: &str = row.get("chunk");
        assert!(
            chunk.chars().count() <= 20,
            "Chunk {} exceeded 20 chars: {:?}",
            i,
            chunk
        );
    }
}

#[tokio::test]
async fn test_batch_size_processes_all_rows() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;
    // 10 rows but batch_size=3 — worker should loop multiple batches
    setup_source_table(&pool, "items", 10).await;
    setup_queue_table(&pool, "ai", "items_queue", 10).await;
    setup_destination_table(&pool, "public", "items_embeddings").await;

    let (mock_url, request_log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("items")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .batch_size(3)
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.items_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 10, "All 10 rows should be embedded");

    // Queue should be fully drained
    let queue_count: i64 = sqlx::query_scalar("SELECT count(*) FROM ai.items_queue")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(queue_count, 0);

    // Should have made multiple embedding requests (batch_size=3, 10 items)
    let requests = request_log.lock().unwrap();
    assert!(
        requests.len() >= 3,
        "Expected at least 3 embedding requests for 10 items with batch_size=3, got {}",
        requests.len()
    );
}

#[tokio::test]
async fn test_multiple_vectorizers_in_one_run() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;

    // Vectorizer 1: table "alpha"
    setup_source_table(&pool, "alpha", 2).await;
    setup_queue_table(&pool, "ai", "alpha_queue", 2).await;
    setup_destination_table(&pool, "public", "alpha_embeddings").await;

    // Vectorizer 2: table "beta"
    setup_source_table(&pool, "beta", 3).await;
    setup_queue_table(&pool, "ai", "beta_queue", 3).await;
    setup_destination_table(&pool, "public", "beta_embeddings").await;

    let (mock_url, _log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("alpha")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    VectorizerConfigBuilder::new("beta")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    let alpha_count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.alpha_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    let beta_count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.beta_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();

    assert_eq!(alpha_count, 2, "Alpha should have 2 embeddings");
    assert_eq!(beta_count, 3, "Beta should have 3 embeddings");
}

#[tokio::test]
async fn test_specific_vectorizer_id_filtering() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;

    setup_source_table(&pool, "first", 2).await;
    setup_queue_table(&pool, "ai", "first_queue", 2).await;
    setup_destination_table(&pool, "public", "first_embeddings").await;

    setup_source_table(&pool, "second", 2).await;
    setup_queue_table(&pool, "ai", "second_queue", 2).await;
    setup_destination_table(&pool, "public", "second_embeddings").await;

    let (mock_url, _log) = start_mock_embedding_server().await;

    let id1 = VectorizerConfigBuilder::new("first")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    let _id2 = VectorizerConfigBuilder::new("second")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    // Only run vectorizer id1
    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![id1],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    let first_count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.first_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    let second_count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.second_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();

    assert_eq!(first_count, 2, "First vectorizer should have processed");
    assert_eq!(second_count, 0, "Second vectorizer should NOT have processed");
}

#[tokio::test]
async fn test_rerun_is_noop() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;
    setup_source_table(&pool, "rerun", 2).await;
    setup_queue_table(&pool, "ai", "rerun_queue", 2).await;
    setup_destination_table(&pool, "public", "rerun_embeddings").await;

    let (mock_url, request_log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("rerun")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    // First run
    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    let count_after_first: i64 = sqlx::query_scalar("SELECT count(*) FROM public.rerun_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count_after_first, 2);

    let requests_after_first = request_log.lock().unwrap().len();

    // Second run — queue is empty, should be a no-op
    let worker2 = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker2.run().await.unwrap();

    let count_after_second: i64 = sqlx::query_scalar("SELECT count(*) FROM public.rerun_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count_after_second, 2, "Count should not change on re-run");

    let requests_after_second = request_log.lock().unwrap().len();
    assert_eq!(
        requests_after_first, requests_after_second,
        "No new embedding requests on re-run"
    );
}

#[tokio::test]
async fn test_mock_server_receives_correct_input() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;
    setup_source_table_with_content(&pool, "verify", &["hello world", "rust is great"]).await;
    setup_queue_table(&pool, "ai", "verify_queue", 2).await;
    setup_destination_table(&pool, "public", "verify_embeddings").await;

    let (mock_url, request_log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("verify")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    let requests = request_log.lock().unwrap();
    assert!(!requests.is_empty(), "Should have made at least one request");

    // Verify the model name was sent correctly
    let first_req = &requests[0];
    assert_eq!(
        first_req.get("model").and_then(|v| v.as_str()),
        Some("text-embedding-3-small"),
        "Model should be text-embedding-3-small"
    );

    // Verify input texts were sent
    let input = first_req.get("input").unwrap();
    let input_arr = input.as_array().unwrap();
    let texts: Vec<&str> = input_arr.iter().filter_map(|v| v.as_str()).collect();
    assert!(
        texts.contains(&"hello world") && texts.contains(&"rust is great"),
        "Input should contain source content, got: {:?}",
        texts
    );
}

// ============================================================
// Ollama embedding tests (require local Ollama with embeddinggemma:300m)
// Run with: OLLAMA_HOST=http://localhost:11434 cargo test --test integration -- ollama
// ============================================================

fn ollama_available() -> bool {
    std::env::var("OLLAMA_HOST").is_ok()
        || std::net::TcpStream::connect("127.0.0.1:11434").is_ok()
}

fn ollama_url() -> String {
    std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string())
}

// ============================================================
// Schema evolution tests
// ============================================================

#[tokio::test]
async fn test_schema_evolution_extra_columns_on_destination() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;
    setup_source_table(&pool, "evo1", 2).await;
    setup_queue_table(&pool, "ai", "evo1_queue", 2).await;
    setup_destination_table(&pool, "public", "evo1_embeddings").await;

    // Add an extra column to the destination table before running
    sqlx::query("ALTER TABLE public.evo1_embeddings ADD COLUMN extra_data TEXT DEFAULT 'hello'")
        .execute(&pool)
        .await
        .unwrap();

    let (mock_url, _log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("evo1")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.evo1_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 2, "Should embed despite extra column on destination");

    // Extra column should have its default value
    let extras: Vec<String> = sqlx::query_scalar("SELECT extra_data FROM public.evo1_embeddings")
        .fetch_all(&pool)
        .await
        .unwrap();
    for extra in &extras {
        assert_eq!(extra, "hello", "Extra column should have default value");
    }

    // Embeddings should be valid
    let emb: Vec<f32> =
        sqlx::query_scalar("SELECT embedding FROM public.evo1_embeddings WHERE id = 1")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(emb.len(), 3);
}

#[tokio::test]
async fn test_schema_evolution_drop_readd_embedding_column() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;
    setup_source_table(&pool, "evo2", 2).await;
    setup_queue_table(&pool, "ai", "evo2_queue", 2).await;
    setup_destination_table(&pool, "public", "evo2_embeddings").await;

    let (mock_url, _log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("evo2")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    // First run — embeds 2 rows
    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.evo2_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 2);

    // Schema evolution: drop embedding column, add extra column, re-add embedding
    sqlx::query("ALTER TABLE public.evo2_embeddings DROP COLUMN embedding")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("ALTER TABLE public.evo2_embeddings ADD COLUMN extra TEXT DEFAULT 'added'")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("ALTER TABLE public.evo2_embeddings ADD COLUMN embedding FLOAT4[]")
        .execute(&pool)
        .await
        .unwrap();

    // Clear destination and re-queue for a second run
    sqlx::query("DELETE FROM public.evo2_embeddings")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO ai.evo2_queue (id) VALUES (1), (2)")
        .execute(&pool)
        .await
        .unwrap();

    // Second run — embeddings go into the re-added column
    let worker2 = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker2.run().await.unwrap();

    let count2: i64 = sqlx::query_scalar("SELECT count(*) FROM public.evo2_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count2, 2, "Should re-embed after column drop/re-add");

    // Embedding column should have valid data
    let emb: Vec<f32> =
        sqlx::query_scalar("SELECT embedding FROM public.evo2_embeddings WHERE id = 1")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(emb.len(), 3);

    // Extra column should have its default
    let extra: String =
        sqlx::query_scalar("SELECT extra FROM public.evo2_embeddings WHERE id = 1")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(extra, "added");
}

// ============================================================
// Same-table (column destination) vectorization tests
// ============================================================

#[tokio::test]
async fn test_same_table_column_destination() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;

    // Source table with an embedding column (same-table vectorization)
    sqlx::query(
        "CREATE TABLE public.col_dest (id SERIAL PRIMARY KEY, content TEXT, embedding FLOAT4[])",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query("INSERT INTO public.col_dest (content) VALUES ('first entry')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO public.col_dest (content) VALUES ('second entry')")
        .execute(&pool)
        .await
        .unwrap();

    setup_queue_table(&pool, "ai", "col_dest_queue", 2).await;

    let (mock_url, _log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("col_dest")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .destination_column("embedding")
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    // Both rows should now have non-NULL embeddings
    let null_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM public.col_dest WHERE embedding IS NULL")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(null_count, 0, "All rows should have embeddings");

    // Verify embeddings are valid float arrays of correct dimension
    let rows = sqlx::query("SELECT id, embedding FROM public.col_dest ORDER BY id")
        .fetch_all(&pool)
        .await
        .unwrap();
    assert_eq!(rows.len(), 2);

    for row in &rows {
        let emb: Vec<f32> = row.get("embedding");
        assert_eq!(emb.len(), 3, "Embedding should have 3 dimensions");
        let sum: f32 = emb.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0, "Embedding should not be all zeros");
    }

    // Queue should be drained
    let queue_count: i64 = sqlx::query_scalar("SELECT count(*) FROM ai.col_dest_queue")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(queue_count, 0, "Queue should be empty after processing");
}

#[tokio::test]
async fn test_ollama_embedding_end_to_end() {
    if !ollama_available() {
        eprintln!("Skipping: Ollama not available");
        return;
    }

    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;
    setup_source_table_with_content(&pool, "ollama_docs", &["Rust is a systems programming language", "PostgreSQL is a relational database"]).await;
    setup_queue_table(&pool, "ai", "ollama_docs_queue", 2).await;
    setup_destination_table(&pool, "public", "ollama_docs_embeddings").await;

    VectorizerConfigBuilder::new("ollama_docs")
        .embedding_ollama("embeddinggemma:300m", Some(&ollama_url()))
        .chunking_none()
        .insert(&pool)
        .await;

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.ollama_docs_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 2, "Should have 2 embeddings");

    // embeddinggemma:300m returns 768-dim vectors
    let embedding: Vec<f32> = sqlx::query_scalar(
        "SELECT embedding FROM public.ollama_docs_embeddings WHERE id = 1",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(embedding.len(), 768, "embeddinggemma:300m should produce 768-dim vectors");

    // Embeddings should be non-trivial (not all zeros)
    let sum: f32 = embedding.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "Embedding should not be all zeros");
}

#[tokio::test]
async fn test_ollama_embedding_different_texts_produce_different_embeddings() {
    if !ollama_available() {
        eprintln!("Skipping: Ollama not available");
        return;
    }

    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;
    setup_source_table_with_content(
        &pool,
        "ollama_diff",
        &["The weather is sunny today", "Quantum mechanics describes subatomic particles"],
    )
    .await;
    setup_queue_table(&pool, "ai", "ollama_diff_queue", 2).await;
    setup_destination_table(&pool, "public", "ollama_diff_embeddings").await;

    VectorizerConfigBuilder::new("ollama_diff")
        .embedding_ollama("embeddinggemma:300m", Some(&ollama_url()))
        .chunking_none()
        .insert(&pool)
        .await;

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    let emb1: Vec<f32> = sqlx::query_scalar(
        "SELECT embedding FROM public.ollama_diff_embeddings WHERE id = 1",
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    let emb2: Vec<f32> = sqlx::query_scalar(
        "SELECT embedding FROM public.ollama_diff_embeddings WHERE id = 2",
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    // Different texts should produce different embeddings
    assert_ne!(emb1, emb2, "Different texts should have different embeddings");

    // Compute cosine similarity — should be less than 1.0 (not identical)
    let dot: f32 = emb1.iter().zip(&emb2).map(|(a, b)| a * b).sum();
    let mag1: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag2: f32 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cosine_sim = dot / (mag1 * mag2);
    assert!(
        cosine_sim < 0.99,
        "Cosine similarity should be < 0.99 for very different texts, got {}",
        cosine_sim
    );
}

#[tokio::test]
async fn test_ollama_embedding_with_chunking() {
    if !ollama_available() {
        eprintln!("Skipping: Ollama not available");
        return;
    }

    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;

    let long_text = "Rust is a multi-paradigm programming language. \
        It emphasizes performance, type safety, and concurrency. \
        Rust enforces memory safety without a garbage collector. \
        It was originally designed by Graydon Hoare at Mozilla Research.";
    setup_source_table_with_content(&pool, "ollama_chunk", &[long_text]).await;
    setup_queue_table(&pool, "ai", "ollama_chunk_queue", 1).await;
    setup_destination_table(&pool, "public", "ollama_chunk_embeddings").await;

    VectorizerConfigBuilder::new("ollama_chunk")
        .embedding_ollama("embeddinggemma:300m", Some(&ollama_url()))
        .chunking_recursive(80, 0)
        .insert(&pool)
        .await;

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    let rows = sqlx::query("SELECT chunk_seq, chunk, embedding FROM public.ollama_chunk_embeddings ORDER BY chunk_seq")
        .fetch_all(&pool)
        .await
        .unwrap();

    assert!(rows.len() > 1, "Long text should produce multiple chunks, got {}", rows.len());

    for (i, row) in rows.iter().enumerate() {
        let chunk_seq: i32 = row.get("chunk_seq");
        assert_eq!(chunk_seq, i as i32);

        let chunk: &str = row.get("chunk");
        assert!(chunk.chars().count() <= 80, "Chunk {} exceeded 80 chars", i);

        let embedding: Vec<f32> = row.get("embedding");
        assert_eq!(embedding.len(), 768, "Each chunk should have 768-dim embedding");
    }
}

// ============================================================
// Concurrency tests
// ============================================================

#[tokio::test]
async fn test_concurrency_processes_all_rows() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;
    setup_source_table(&pool, "conc_items", 20).await;
    setup_queue_table(&pool, "ai", "conc_items_queue", 20).await;
    setup_destination_table(&pool, "public", "conc_items_embeddings").await;

    let (mock_url, _request_log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("conc_items")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .batch_size(5)
        .concurrency(4)
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.conc_items_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 20, "All 20 rows should be embedded");

    // Queue should be fully drained
    let queue_count: i64 = sqlx::query_scalar("SELECT count(*) FROM ai.conc_items_queue")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(queue_count, 0, "Queue should be fully drained");
}

#[tokio::test]
async fn test_concurrency_no_duplicate_embeddings() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;
    setup_source_table(&pool, "nodup_items", 20).await;
    setup_queue_table(&pool, "ai", "nodup_items_queue", 20).await;
    setup_destination_table(&pool, "public", "nodup_items_embeddings").await;

    let (mock_url, _request_log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("nodup_items")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .batch_size(3)
        .concurrency(4)
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    // Total embeddings should equal number of source rows (no duplicates)
    let total: i64 = sqlx::query_scalar("SELECT count(*) FROM public.nodup_items_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(total, 20, "Should have exactly 20 embeddings");

    // Verify no duplicate (id, chunk_seq) pairs
    let distinct: i64 = sqlx::query_scalar(
        "SELECT count(DISTINCT (id, chunk_seq)) FROM public.nodup_items_embeddings"
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(distinct, total, "No duplicate (id, chunk_seq) pairs should exist");
}

// ============================================================
// Parallel vectorizer processing tests
// ============================================================

#[tokio::test]
async fn test_parallel_vectorizers_failing_one_does_not_block_others() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;

    // Vectorizer 1: will succeed (valid mock server)
    setup_source_table(&pool, "para_ok", 3).await;
    setup_queue_table(&pool, "ai", "para_ok_queue", 3).await;
    setup_destination_table(&pool, "public", "para_ok_embeddings").await;

    let (mock_url, _log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("para_ok")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    // Vectorizer 2: will fail (failing mock server)
    setup_source_table(&pool, "para_fail", 2).await;
    setup_queue_table(&pool, "ai", "para_fail_queue", 2).await;
    setup_destination_table(&pool, "public", "para_fail_embeddings").await;

    let (fail_url, _fail_log) = start_failing_mock_embedding_server().await;

    VectorizerConfigBuilder::new("para_fail")
        .embedding_openai("text-embedding-3-small", 3, &fail_url)
        .chunking_none()
        .insert(&pool)
        .await;

    // Vectorizer 3: will also succeed
    setup_source_table(&pool, "para_ok2", 2).await;
    setup_queue_table(&pool, "ai", "para_ok2_queue", 2).await;
    setup_destination_table(&pool, "public", "para_ok2_embeddings").await;

    VectorizerConfigBuilder::new("para_ok2")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        false, // exit_on_error = false
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    // Successful vectorizers should have completed
    let ok_count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.para_ok_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(ok_count, 3, "First successful vectorizer should have 3 embeddings");

    let ok2_count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.para_ok2_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(ok2_count, 2, "Second successful vectorizer should have 2 embeddings");

    // Failed vectorizer should have 0 embeddings
    let fail_count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.para_fail_embeddings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(fail_count, 0, "Failed vectorizer should have 0 embeddings");

    // Error should be recorded for the failed vectorizer
    let error_count: i64 = sqlx::query_scalar("SELECT count(*) FROM ai.vectorizer_errors")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert!(error_count > 0, "Should have recorded errors for the failed vectorizer");
}

#[tokio::test]
async fn test_parallel_vectorizers_cancellation() {
    let (node, pool) = start_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();

    setup_ai_schema(&pool).await;

    // Create two vectorizers with work to do
    setup_source_table(&pool, "cancel_a", 5).await;
    setup_queue_table(&pool, "ai", "cancel_a_queue", 5).await;
    setup_destination_table(&pool, "public", "cancel_a_embeddings").await;

    setup_source_table(&pool, "cancel_b", 5).await;
    setup_queue_table(&pool, "ai", "cancel_b_queue", 5).await;
    setup_destination_table(&pool, "public", "cancel_b_embeddings").await;

    let (mock_url, _log) = start_mock_embedding_server().await;

    VectorizerConfigBuilder::new("cancel_a")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    VectorizerConfigBuilder::new("cancel_b")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    std::env::set_var("MOCK_KEY", "test");

    // Cancel immediately — worker should exit gracefully
    let cancel = CancellationToken::new();
    cancel.cancel();

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true,
        cancel,
    )
    .await
    .unwrap();

    let result = worker.run().await;
    assert!(result.is_ok(), "Worker should exit gracefully on cancellation: {:?}", result);
}

// ============================================================
// Full pipeline test using real extension SQL
// ============================================================

#[tokio::test]
async fn test_full_pipeline_with_extension_sql() {
    // Start pgvector-enabled container and load the real extension SQL
    let (node, pool) = start_pgvector_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();
    load_extension_sql(&pool).await;

    // Create a source table with realistic content
    sqlx::query(
        "CREATE TABLE public.articles (
            id serial PRIMARY KEY,
            title text NOT NULL,
            content text NOT NULL,
            author text NOT NULL
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO public.articles (title, content, author) VALUES
            ('Getting Started with Rust', 'Rust is a systems programming language focused on safety and performance.', 'Alice'),
            ('PostgreSQL Tips', 'PostgreSQL offers powerful indexing and full-text search capabilities.', 'Bob'),
            ('Vector Embeddings 101', 'Embeddings transform text into dense numerical representations for similarity search.', 'Carol')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Start mock embedding server
    let (mock_url, _log) = start_mock_embedding_server().await;
    std::env::set_var("OPENAI_API_KEY", "test-key");

    // Call the REAL ai.create_vectorizer() function
    let vectorizer_id: i32 = sqlx::query_scalar(
        "SELECT ai.create_vectorizer(
            'public.articles'::regclass,
            loading    => ai.loading_column('content'),
            embedding  => ai.embedding_openai('text-embedding-3-small', 3, base_url => $1),
            chunking   => ai.chunking_none(),
            formatting => ai.formatting_chunk_value()
        )",
    )
    .bind(&mock_url)
    .fetch_one(&pool)
    .await
    .unwrap();

    assert!(vectorizer_id > 0, "Vectorizer ID should be positive");

    // Verify queue was populated with existing rows
    let queue_count: i64 =
        sqlx::query_scalar("SELECT ai.vectorizer_queue_pending($1)")
            .bind(vectorizer_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(queue_count, 3, "Queue should have 3 rows from existing data");

    // Run the worker
    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![vectorizer_id],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    // Verify queue is drained
    let queue_after: i64 =
        sqlx::query_scalar("SELECT ai.vectorizer_queue_pending($1)")
            .bind(vectorizer_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(queue_after, 0, "Queue should be empty after worker run");

    // Verify embeddings were created in the target table
    let emb_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM public.articles_embedding_store")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(emb_count, 3, "Should have 3 embeddings");

    // Verify embedding dimensions
    let emb: Vec<f32> = sqlx::query_scalar(
        "SELECT embedding::float4[] FROM public.articles_embedding_store WHERE id = 1",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(emb.len(), 3, "Embedding should have 3 dimensions");

    // Verify the view works (joins source + embeddings)
    let view_rows = sqlx::query(
        "SELECT title, author, chunk, embedding_uuid FROM public.articles_embedding ORDER BY id",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(view_rows.len(), 3, "View should return 3 rows");
    let first_title: &str = view_rows[0].get("title");
    assert_eq!(first_title, "Getting Started with Rust");

    // Test trigger: INSERT a new row → should be enqueued
    sqlx::query(
        "INSERT INTO public.articles (title, content, author) VALUES ('New Post', 'Fresh content about AI.', 'Dave')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let queue_after_insert: i64 =
        sqlx::query_scalar("SELECT ai.vectorizer_queue_pending($1)")
            .bind(vectorizer_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(queue_after_insert, 1, "New row should be enqueued");

    // Run worker again to process the new row
    let worker2 = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![vectorizer_id],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker2.run().await.unwrap();

    let emb_count_after: i64 =
        sqlx::query_scalar("SELECT count(*) FROM public.articles_embedding_store")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(emb_count_after, 4, "Should have 4 embeddings after insert");

    // Test trigger: UPDATE content → should be re-enqueued
    sqlx::query("UPDATE public.articles SET content = 'Updated content about Rust safety.' WHERE id = 1")
        .execute(&pool)
        .await
        .unwrap();

    let queue_after_update: i64 =
        sqlx::query_scalar("SELECT ai.vectorizer_queue_pending($1)")
            .bind(vectorizer_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(queue_after_update, 1, "Updated row should be re-enqueued");

    // Run worker to process the update
    let worker3 = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![vectorizer_id],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker3.run().await.unwrap();

    // Count should still be 4 (updated, not duplicated)
    let emb_count_final: i64 =
        sqlx::query_scalar("SELECT count(*) FROM public.articles_embedding_store")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(emb_count_final, 4, "Should still have 4 embeddings after update");

    // Cleanup
    sqlx::query("SELECT ai.drop_vectorizer($1, drop_all => true)")
        .bind(vectorizer_id)
        .execute(&pool)
        .await
        .unwrap();

}

#[tokio::test]
async fn test_full_pipeline_with_chunking() {
    let (node, pool) = start_pgvector_postgres().await;
    let port = node.get_host_port_ipv4(5432).await.unwrap();
    load_extension_sql(&pool).await;

    // Create source with long content that will be chunked
    sqlx::query(
        "CREATE TABLE public.long_docs (
            id serial PRIMARY KEY,
            content text NOT NULL
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    let long_content = "This is a paragraph about databases. ".repeat(30);
    sqlx::query("INSERT INTO public.long_docs (content) VALUES ($1)")
        .bind(&long_content)
        .execute(&pool)
        .await
        .unwrap();

    let (mock_url, _log) = start_mock_embedding_server().await;
    std::env::set_var("OPENAI_API_KEY", "test-key");

    let vectorizer_id: i32 = sqlx::query_scalar(
        "SELECT ai.create_vectorizer(
            'public.long_docs'::regclass,
            loading    => ai.loading_column('content'),
            embedding  => ai.embedding_openai('text-embedding-3-small', 3, base_url => $1),
            chunking   => ai.chunking_recursive_character_text_splitter(
                chunk_size => 100,
                chunk_overlap => 0
            ),
            formatting => ai.formatting_chunk_value()
        )",
    )
    .bind(&mock_url)
    .fetch_one(&pool)
    .await
    .unwrap();

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![vectorizer_id],
        true,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    worker.run().await.unwrap();

    // Should produce multiple chunks from long content
    let chunk_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM public.long_docs_embedding_store")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert!(
        chunk_count > 1,
        "Long content should produce multiple chunks, got {}",
        chunk_count
    );

    // Verify sequential chunk_seq values
    let rows = sqlx::query(
        "SELECT chunk_seq, chunk FROM public.long_docs_embedding_store ORDER BY chunk_seq",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    for (i, row) in rows.iter().enumerate() {
        let chunk_seq: i32 = row.get("chunk_seq");
        assert_eq!(chunk_seq, i as i32, "chunk_seq should be sequential");
    }

    // Queue should be empty
    let queue_after: i64 =
        sqlx::query_scalar("SELECT ai.vectorizer_queue_pending($1)")
            .bind(vectorizer_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(queue_after, 0, "Queue should be empty");

    sqlx::query("SELECT ai.drop_vectorizer($1, drop_all => true)")
        .bind(vectorizer_id)
        .execute(&pool)
        .await
        .unwrap();

}
