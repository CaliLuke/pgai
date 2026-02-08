mod common;

use common::*;
use sqlx::{Pool, Postgres, Row};
use std::time::Duration;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres as PostgresImage;
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

    VectorizerConfigBuilder::new("nokey")
        .embedding_openai("text-embedding-3-small", 3, &mock_url)
        .chunking_none()
        .insert(&pool)
        .await;

    // Ensure the key is NOT set
    std::env::remove_var("MOCK_KEY");

    let worker = Worker::new(
        &db_url_from_port(port),
        Duration::from_secs(1),
        true,
        vec![],
        true, // exit_on_error
    )
    .await
    .unwrap();

    let result = worker.run().await;
    assert!(result.is_err(), "Should fail when API key not set");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("MOCK_KEY") && err.contains("not found"),
        "Error should mention the missing key name, got: {}",
        err
    );

    // Restore for other tests
    std::env::set_var("MOCK_KEY", "test");
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
