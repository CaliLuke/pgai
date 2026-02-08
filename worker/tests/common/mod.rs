use axum::{routing::post, Json, Router};
use serde_json::json;
use sqlx::{Pool, Postgres};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

pub async fn setup_ai_schema(pool: &Pool<Postgres>) {
    sqlx::query("CREATE SCHEMA IF NOT EXISTS ai")
        .execute(pool)
        .await
        .unwrap();
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS ai.vectorizer (
            id SERIAL PRIMARY KEY,
            source_schema TEXT NOT NULL,
            source_table TEXT NOT NULL,
            queue_schema TEXT NOT NULL,
            queue_table TEXT NOT NULL,
            config JSONB NOT NULL,
            source_pk JSONB NOT NULL,
            disabled BOOLEAN NOT NULL DEFAULT false
        )
        "#,
    )
    .execute(pool)
    .await
    .unwrap();
    // Worker checks for this table to verify pgai is installed
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS ai.pgai_lib_version (
            name TEXT PRIMARY KEY,
            version TEXT NOT NULL
        )
        "#,
    )
    .execute(pool)
    .await
    .unwrap();
    sqlx::query("INSERT INTO ai.pgai_lib_version (name, version) VALUES ('ai', '0.0.0-test')")
        .execute(pool)
        .await
        .unwrap();
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS ai.vectorizer_errors (
            id INT NOT NULL REFERENCES ai.vectorizer(id) ON DELETE CASCADE,
            message TEXT,
            details JSONB,
            recorded TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        "#,
    )
    .execute(pool)
    .await
    .unwrap();
}

pub async fn setup_source_table(pool: &Pool<Postgres>, table: &str, num_rows: i32) {
    sqlx::query(&format!(
        "CREATE TABLE public.\"{table}\" (id SERIAL PRIMARY KEY, content TEXT)"
    ))
    .execute(pool)
    .await
    .unwrap();

    for i in 1..=num_rows {
        sqlx::query(&format!(
            "INSERT INTO public.\"{table}\" (content) VALUES ($1)"
        ))
        .bind(format!("post_{i}"))
        .execute(pool)
        .await
        .unwrap();
    }
}

pub async fn setup_source_table_with_content(
    pool: &Pool<Postgres>,
    table: &str,
    rows: &[&str],
) {
    sqlx::query(&format!(
        "CREATE TABLE public.\"{table}\" (id SERIAL PRIMARY KEY, content TEXT)"
    ))
    .execute(pool)
    .await
    .unwrap();

    for content in rows {
        sqlx::query(&format!(
            "INSERT INTO public.\"{table}\" (content) VALUES ($1)"
        ))
        .bind(*content)
        .execute(pool)
        .await
        .unwrap();
    }
}

pub async fn setup_queue_table(pool: &Pool<Postgres>, schema: &str, table: &str, num_rows: i32) {
    sqlx::query(&format!(
        "CREATE TABLE \"{schema}\".\"{table}\" (id INT)"
    ))
    .execute(pool)
    .await
    .unwrap();

    for i in 1..=num_rows {
        sqlx::query(&format!(
            "INSERT INTO \"{schema}\".\"{table}\" (id) VALUES ($1)"
        ))
        .bind(i)
        .execute(pool)
        .await
        .unwrap();
    }
}

pub async fn setup_destination_table(pool: &Pool<Postgres>, schema: &str, table: &str) {
    sqlx::query(&format!(
        "CREATE TABLE \"{schema}\".\"{table}\" (id INT, chunk_seq INT, chunk TEXT, embedding FLOAT4[])"
    ))
    .execute(pool)
    .await
    .unwrap();
}

pub struct VectorizerConfigBuilder {
    source_table: String,
    embedding: serde_json::Value,
    chunking: serde_json::Value,
    formatting: serde_json::Value,
    loading: serde_json::Value,
    destination: serde_json::Value,
    batch_size: Option<i32>,
    concurrency: Option<i32>,
    source_pk: Option<serde_json::Value>,
}

impl VectorizerConfigBuilder {
    pub fn new(source_table: &str) -> Self {
        Self {
            source_table: source_table.to_string(),
            embedding: json!({
                "implementation": "openai",
                "model": "text-embedding-3-small",
                "dimensions": 3,
                "api_key_name": "MOCK_KEY",
                "base_url": "http://127.0.0.1:0/v1"
            }),
            chunking: json!({"implementation": "none"}),
            formatting: json!({"implementation": "chunk_value"}),
            loading: json!({"implementation": "column", "column_name": "content"}),
            destination: json!({
                "implementation": "table",
                "target_table": format!("{source_table}_embeddings")
            }),
            batch_size: None,
            concurrency: None,
            source_pk: None,
        }
    }

    pub fn embedding_openai(mut self, model: &str, dims: i32, base_url: &str) -> Self {
        self.embedding = json!({
            "implementation": "openai",
            "model": model,
            "dimensions": dims,
            "api_key_name": "MOCK_KEY",
            "base_url": base_url,
        });
        self
    }

    pub fn embedding_ollama(mut self, model: &str, base_url: Option<&str>) -> Self {
        self.embedding = json!({
            "implementation": "ollama",
            "model": model,
        });
        if let Some(url) = base_url {
            self.embedding["base_url"] = json!(url);
        }
        self
    }

    pub fn chunking_none(mut self) -> Self {
        self.chunking = json!({"implementation": "none"});
        self
    }

    pub fn chunking_recursive(mut self, chunk_size: usize, chunk_overlap: usize) -> Self {
        self.chunking = json!({
            "implementation": "recursive_character_text_splitter",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        });
        self
    }

    pub fn chunking_character(
        mut self,
        chunk_size: usize,
        chunk_overlap: usize,
        separator: &str,
    ) -> Self {
        self.chunking = json!({
            "implementation": "character_text_splitter",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "separator": separator,
        });
        self
    }

    pub fn batch_size(mut self, size: i32) -> Self {
        self.batch_size = Some(size);
        self
    }

    pub fn concurrency(mut self, n: i32) -> Self {
        self.concurrency = Some(n);
        self
    }

    pub fn loading_column(mut self, col: &str) -> Self {
        self.loading = json!({"implementation": "column", "column_name": col});
        self
    }

    pub fn destination_table(mut self, table: &str) -> Self {
        self.destination = json!({
            "implementation": "table",
            "target_table": table,
        });
        self
    }

    pub fn destination_column(mut self, embedding_column: &str) -> Self {
        self.destination = json!({
            "implementation": "column",
            "embedding_column": embedding_column,
        });
        self
    }

    pub fn source_pk(mut self, pk: serde_json::Value) -> Self {
        self.source_pk = Some(pk);
        self
    }

    pub async fn insert(self, pool: &Pool<Postgres>) -> i32 {
        let mut config = json!({
            "version": "1.0",
            "embedding": self.embedding,
            "chunking": self.chunking,
            "formatting": self.formatting,
            "loading": self.loading,
            "destination": self.destination,
        });

        let mut processing = json!({"concurrency": self.concurrency.unwrap_or(1)});
        if let Some(bs) = self.batch_size {
            processing["batch_size"] = json!(bs);
        }
        config["processing"] = processing;

        let pk_info = self.source_pk.unwrap_or_else(|| json!([{"attname": "id", "pknum": 1, "attnum": 1}]));

        let id: i32 = sqlx::query_scalar(
            "INSERT INTO ai.vectorizer (source_schema, source_table, queue_schema, queue_table, config, source_pk)
             VALUES ('public', $1, 'ai', $2, $3, $4)
             RETURNING id",
        )
        .bind(&self.source_table)
        .bind(format!("{}_queue", self.source_table))
        .bind(config)
        .bind(pk_info)
        .fetch_one(pool)
        .await
        .unwrap();

        id
    }
}

/// Start a mock embedding server that always returns HTTP 500.
/// Returns (base_url, request_log).
pub async fn start_failing_mock_embedding_server() -> (String, Arc<Mutex<Vec<serde_json::Value>>>) {
    let request_log: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let log_clone = request_log.clone();

    let app = Router::new().route(
        "/v1/embeddings",
        post(move |Json(body): Json<serde_json::Value>| {
            let log = log_clone.clone();
            async move {
                log.lock().unwrap().push(body);
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": {"message": "mock server error", "type": "server_error"}})),
                )
            }
        }),
    );

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let mock_url = format!("http://{}/v1", addr);

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (mock_url, request_log)
}

/// Start a mock embedding server that returns deterministic embeddings.
/// Returns (base_url, request_log) where base_url includes the /v1 path.
pub async fn start_mock_embedding_server() -> (String, Arc<Mutex<Vec<serde_json::Value>>>) {
    let request_log: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let log_clone = request_log.clone();

    let app = Router::new().route(
        "/v1/embeddings",
        post(move |Json(body): Json<serde_json::Value>| {
            let log = log_clone.clone();
            async move {
                log.lock().unwrap().push(body.clone());

                let input = body.get("input").unwrap();
                let count = match input {
                    serde_json::Value::Array(arr) => arr.len(),
                    _ => 1,
                };

                let data: Vec<serde_json::Value> = (0..count)
                    .map(|i| {
                        json!({
                            "index": i,
                            "object": "embedding",
                            "embedding": [0.1 * (i as f64 + 1.0), 0.2 * (i as f64 + 1.0), 0.3 * (i as f64 + 1.0)]
                        })
                    })
                    .collect();

                Json(json!({
                    "object": "list",
                    "data": data,
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 0, "total_tokens": 0}
                }))
            }
        }),
    );

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let mock_url = format!("http://{}/v1", addr);

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (mock_url, request_log)
}
