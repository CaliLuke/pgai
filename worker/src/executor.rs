use anyhow::{anyhow, Result};
use backon::{ExponentialBuilder, Retryable};
use sqlx::{Pool, Postgres};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::{info, error, warn};
use crate::errors::EmbeddingError;
use crate::models::{Vectorizer, LoadingConfig, DestinationConfig, ChunkerConfig};
use crate::embedder::{create_embedder, Embedder};
use crate::worker_tracking::WorkerTracking;

/// Bind a serde_json::Value to a sqlx query, handling Number/String/other types.
fn bind_json_value<'q>(
    q: sqlx::query::Query<'q, sqlx::Postgres, sqlx::postgres::PgArguments>,
    val: &'q serde_json::Value,
) -> sqlx::query::Query<'q, sqlx::Postgres, sqlx::postgres::PgArguments> {
    match val {
        serde_json::Value::Number(n) => q.bind(n.as_i64()),
        serde_json::Value::String(s) => q.bind(s.as_str()),
        _ => q.bind(val.to_string()),
    }
}

pub struct Executor {
    pool: Pool<Postgres>,
    vectorizer: Vectorizer,
    embedder: Box<dyn Embedder>,
    cancel: CancellationToken,
    tracking: Arc<WorkerTracking>,
}

impl Executor {
    pub async fn new(
        pool: Pool<Postgres>,
        vectorizer: Vectorizer,
        cancel: CancellationToken,
        tracking: Arc<WorkerTracking>,
    ) -> Result<Self> {
        let embedder = create_embedder(&vectorizer.config.embedding, &pool).await?;
        Ok(Self { pool, vectorizer, embedder, cancel, tracking })
    }

    #[tracing::instrument(skip(self), fields(vectorizer_id = self.vectorizer.id))]
    pub async fn run(&self) -> Result<i32> {
        let mut total_processed = 0;

        loop {
            if self.cancel.is_cancelled() {
                info!("Shutdown requested, stopping executor for vectorizer {}", self.vectorizer.id);
                break;
            }
            let items_processed = self.do_batch().await?;
            if items_processed == 0 {
                break;
            }
            total_processed += items_processed;
        }

        Ok(total_processed)
    }

    async fn embed_with_retry(&self, chunks: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let vid = self.vectorizer.id;
        let chunks = chunks.to_vec();

        (|| {
            let inputs = chunks.clone();
            async move { self.embedder.embed(inputs).await }
        })
        .retry(
            ExponentialBuilder::default()
                .with_min_delay(Duration::from_secs(1))
                .with_max_delay(Duration::from_secs(10))
                .with_max_times(3),
        )
        .when(|e: &EmbeddingError| {
            if e.is_transient() {
                warn!("Transient embedding error for vectorizer {}, retrying: {}", vid, e);
                true
            } else {
                false
            }
        })
        .await
    }

    #[tracing::instrument(skip(self), fields(vectorizer_id = self.vectorizer.id))]
    async fn do_batch(&self) -> Result<i32> {
        let items = self.fetch_work().await?;
        if items.is_empty() {
            return Ok(0);
        }

        info!("Fetched {} items for vectorizer {}", items.len(), self.vectorizer.id);

        let mut all_chunks = Vec::new();
        let mut item_chunk_counts = Vec::new();
        let mut items_with_content: Vec<serde_json::Value> = Vec::new();

        for item in &items {
            let text = self.extract_text(item)?;
            match text {
                Some(t) => {
                    let chunks = self.perform_chunking(&t, &self.vectorizer.config.chunking)?;
                    item_chunk_counts.push(chunks.len());
                    for chunk in chunks {
                        let formatted = self.vectorizer.config.formatting.format(&chunk, item);
                        all_chunks.push(formatted);
                    }
                    items_with_content.push(item.clone());
                }
                None => {
                    // NULL or empty content — skip embedding but count as processed
                    info!("Skipping item with NULL/empty content for vectorizer {}", self.vectorizer.id);
                }
            }
        }

        if all_chunks.is_empty() {
            return Ok(items.len() as i32);
        }

        let embeddings = match self.embed_with_retry(&all_chunks).await {
            Ok(embs) => embs,
            Err(e) => {
                let err_msg = e.to_string();
                error!("Embedding failed for vectorizer {}: {}", self.vectorizer.id, err_msg);
                self.tracking.save_vectorizer_error(Some(self.vectorizer.id), &err_msg).await;
                if let Err(record_err) = self.record_error(
                    "embedding provider failed",
                    serde_json::json!({
                        "error_reason": err_msg,
                    }),
                ).await {
                    error!("Failed to record error: {}", record_err);
                }
                return Err(e.into_inner());
            }
        };
        self.write_results(&items_with_content, &item_chunk_counts, &all_chunks, &embeddings).await?;
        self.tracking.save_vectorizer_success(self.vectorizer.id, items.len() as i32).await;

        Ok(items.len() as i32)
    }

    /// Extract text from a source row. Returns None for NULL or empty/whitespace-only content.
    fn extract_text(&self, item: &serde_json::Value) -> Result<Option<String>> {
        match &self.vectorizer.config.loading {
            LoadingConfig::Column { column_name } => {
                let val = item.get(column_name);
                match val {
                    None => Err(anyhow!("Column {} not found", column_name)),
                    Some(serde_json::Value::Null) => Ok(None),
                    Some(v) => {
                        let s = v.as_str()
                            .ok_or_else(|| anyhow!("Column {} is not a string", column_name))?;
                        if s.trim().is_empty() {
                            Ok(None)
                        } else {
                            Ok(Some(s.to_string()))
                        }
                    }
                }
            }
            _ => Err(anyhow!("Unsupported loading configuration")),
        }
    }

    fn perform_chunking(&self, text: &str, config: &ChunkerConfig) -> Result<Vec<String>> {
        match config {
            ChunkerConfig::RecursiveCharacterTextSplitter {
                chunk_size,
                chunk_overlap,
                separators,
                is_separator_regex,
            } => {
                let mut splitter = pgai_text_splitter::RecursiveCharacterTextSplitter::new(
                    *chunk_size,
                    *chunk_overlap,
                );
                if let Some(seps) = separators {
                    splitter.separators = seps.clone();
                }
                if let Some(is_regex) = is_separator_regex {
                    splitter.is_separator_regex = *is_regex;
                }
                Ok(splitter.split_text(text))
            }
            ChunkerConfig::CharacterTextSplitter {
                chunk_size,
                chunk_overlap,
                separator,
                is_separator_regex,
            } => {
                let mut splitter = pgai_text_splitter::CharacterTextSplitter::new(
                    separator,
                    *chunk_size,
                    *chunk_overlap,
                );
                if let Some(is_regex) = is_separator_regex {
                    splitter.is_separator_regex = *is_regex;
                }
                Ok(splitter.split_text(text))
            }
            ChunkerConfig::SentenceChunker {
                chunk_size,
                chunk_overlap,
                delimiters,
                min_characters_per_sentence,
            } => {
                let mut chunker =
                    pgai_text_splitter::SentenceChunker::new(*chunk_size, *chunk_overlap);
                if let Some(delims) = delimiters {
                    chunker.delimiters = delims.clone();
                }
                if let Some(min_chars) = min_characters_per_sentence {
                    chunker.min_characters_per_sentence = *min_chars;
                }
                Ok(chunker.split_text(text))
            }
            ChunkerConfig::Semchunk {
                chunk_size,
                chunk_overlap,
            } => {
                let splitter =
                    pgai_text_splitter::SemchunkSplitter::new(*chunk_size, *chunk_overlap);
                Ok(splitter.split_text(text))
            }
            ChunkerConfig::None => Ok(vec![text.to_string()]),
        }
    }

    async fn write_results(
        &self, 
        items: &[serde_json::Value], 
        chunk_counts: &[usize], 
        chunks: &[String], 
        embeddings: &[Vec<f32>]
    ) -> Result<()> {
        match &self.vectorizer.config.destination {
            DestinationConfig::Table { target_schema, target_table } => {
                let schema = target_schema.as_deref().unwrap_or(&self.vectorizer.source_schema);
                self.write_to_table(schema, target_table, items, chunk_counts, chunks, embeddings).await
            }
            DestinationConfig::Column { embedding_column } => {
                self.write_to_column(embedding_column, items, chunk_counts, chunks, embeddings).await
            }
        }
    }

    async fn write_to_table(
        &self,
        schema: &str,
        table: &str,
        items: &[serde_json::Value],
        chunk_counts: &[usize],
        chunks: &[String],
        embeddings: &[Vec<f32>]
    ) -> Result<()> {
        let pk_count = self.vectorizer.source_pk.len();
        let pk_cols: Vec<String> = self.vectorizer.source_pk.iter()
            .map(|pk| format!("\"{}\"", pk.attname))
            .collect();
        let pk_list = pk_cols.join(", ");

        // Use an explicit transaction for atomicity
        let mut tx = self.pool.begin().await?;

        // --- Batched DELETE: remove all existing embeddings for items in one statement ---
        // DELETE FROM "s"."t" WHERE (pk_cols) IN (($1), ($2, $3), ...)
        {
            let mut placeholders = Vec::new();
            let mut param_idx = 1usize;
            for _ in items {
                let group: Vec<String> = (0..pk_count)
                    .map(|_| { let p = format!("${}", param_idx); param_idx += 1; p })
                    .collect();
                placeholders.push(format!("({})", group.join(", ")));
            }
            let delete_sql = format!(
                "DELETE FROM \"{}\".\"{}\" WHERE ({}) IN ({})",
                schema, table, pk_list, placeholders.join(", ")
            );
            let mut dq = sqlx::query(&delete_sql);
            for item in items {
                for pk in &self.vectorizer.source_pk {
                    let val = item.get(&pk.attname).ok_or_else(|| anyhow!("PK value not found"))?;
                    dq = bind_json_value(dq, val);
                }
            }
            dq.execute(&mut *tx).await?;
        }

        // --- Batched INSERT: insert all chunks in sub-batches ---
        // Each row needs pk_count + 3 params (PKs + chunk_seq + chunk + embedding).
        // Postgres limit is 65535 params; split into sub-batches if needed.
        let cols_per_row = pk_count + 3;
        let max_rows_per_batch = 65535 / cols_per_row;

        // Collect all rows to insert
        struct InsertRow<'a> {
            item: &'a serde_json::Value,
            seq: i32,
            chunk: &'a str,
            embedding: &'a [f32],
        }
        let mut rows: Vec<InsertRow<'_>> = Vec::new();
        let mut chunk_idx = 0;
        for (i, item) in items.iter().enumerate() {
            let count = chunk_counts[i];
            for seq in 0..count {
                rows.push(InsertRow {
                    item,
                    seq: seq as i32,
                    chunk: &chunks[chunk_idx],
                    embedding: &embeddings[chunk_idx],
                });
                chunk_idx += 1;
            }
        }

        for batch in rows.chunks(max_rows_per_batch) {
            let mut param_idx = 1usize;
            let value_groups: Vec<String> = batch.iter().map(|_| {
                let group: Vec<String> = (0..cols_per_row)
                    .map(|_| { let p = format!("${}", param_idx); param_idx += 1; p })
                    .collect();
                format!("({})", group.join(", "))
            }).collect();

            let insert_sql = format!(
                "INSERT INTO \"{}\".\"{}\" ({}, chunk_seq, chunk, embedding) VALUES {}",
                schema, table, pk_list, value_groups.join(", ")
            );

            let mut q = sqlx::query(&insert_sql);
            for row in batch {
                for pk in &self.vectorizer.source_pk {
                    let val = row.item.get(&pk.attname)
                        .ok_or_else(|| anyhow!("PK value not found"))?;
                    q = bind_json_value(q, val);
                }
                q = q.bind(row.seq)
                     .bind(row.chunk)
                     .bind(row.embedding);
            }
            q.execute(&mut *tx).await?;
        }

        tx.commit().await?;
        Ok(())
    }

    async fn write_to_column(
        &self,
        column: &str,
        items: &[serde_json::Value],
        _chunk_counts: &[usize],
        _chunks: &[String],
        embeddings: &[Vec<f32>]
    ) -> Result<()> {
        for (i, item) in items.iter().enumerate() {
            let embedding = &embeddings[i];
            // Embedding is $1, PK params start at $2
            let (where_clause, pk_vals) = self.build_pk_predicates(item, 2);
            let query = format!(
                "UPDATE \"{}\".\"{}\" SET \"{}\" = $1 WHERE {}",
                self.vectorizer.source_schema, self.vectorizer.source_table, column,
                where_clause
            );

            let mut q = sqlx::query(&query).bind(embedding);
            for val in &pk_vals {
                q = bind_json_value(q, val);
            }
            q.execute(&self.pool).await?;
        }
        Ok(())
    }

    /// Returns a parameterized WHERE clause and the corresponding bind values.
    /// Placeholders start at `$offset` (e.g. offset=1 gives `$1`, `$2`, ...).
    fn build_pk_predicates<'a>(
        &self,
        item: &'a serde_json::Value,
        offset: usize,
    ) -> (String, Vec<&'a serde_json::Value>) {
        let mut parts = Vec::new();
        let mut values = Vec::new();
        for (i, pk) in self.vectorizer.source_pk.iter().enumerate() {
            parts.push(format!("\"{}\" = ${}", pk.attname, offset + i));
            values.push(item.get(&pk.attname).unwrap());
        }
        (parts.join(" AND "), values)
    }

    async fn fetch_work(&self) -> Result<Vec<serde_json::Value>> {
        let pk_cols: Vec<String> = self.vectorizer.source_pk.iter()
            .map(|pk| format!("\"{}\"", pk.attname))
            .collect();
        let pk_list = pk_cols.join(", ");
        let order_by_list = self.vectorizer.source_pk.iter()
            .map(|pk| format!("l.\"{}\"", pk.attname))
            .collect::<Vec<_>>()
            .join(", ");
        let batch_size = self.vectorizer.config.processing.batch_size.unwrap_or(50);

        let query = format!(
            r#"
            WITH selected_rows AS (
                SELECT {pk_list}
                FROM "{queue_schema}"."{queue_table}"
                LIMIT $1
                FOR UPDATE SKIP LOCKED
            ),
            locked_items AS (
                SELECT
                    {pk_list},
                    pg_try_advisory_xact_lock(
                        $2::int,
                        hashtext(concat_ws('|', {pk_list}))::int
                    ) AS locked
                FROM (
                    SELECT DISTINCT {pk_list}
                    FROM selected_rows
                    ORDER BY {pk_list}
                ) as ids
            ),
            deleted_rows AS (
                DELETE FROM "{queue_schema}"."{queue_table}" AS w
                USING locked_items AS l
                WHERE l.locked = true
                AND {delete_join_predicates}
            )
            SELECT to_jsonb(s) as data
            FROM locked_items l
            LEFT JOIN LATERAL (
                SELECT *
                FROM "{source_schema}"."{source_table}" s
                WHERE {lateral_join_predicates}
                LIMIT 1
            ) AS s ON true
            WHERE l.locked = true
            ORDER BY {order_by_list}
            "#,
            pk_list = pk_list,
            order_by_list = order_by_list,
            queue_schema = self.vectorizer.queue_schema,
            source_schema = self.vectorizer.source_schema,
            source_table = self.vectorizer.source_table,
            queue_table = self.vectorizer.queue_table,
            delete_join_predicates = self.build_join_predicates("w", "l"),
            lateral_join_predicates = self.build_join_predicates("l", "s"),
        );

        let queue_table_oid: i32 = sqlx::query_scalar(&format!(
            "SELECT to_regclass('\"{}\".\"{}\"')::oid::int",
            self.vectorizer.queue_schema, self.vectorizer.queue_table
        ))
        .fetch_one(&self.pool)
        .await?;

        let json_rows: Vec<serde_json::Value> = sqlx::query_scalar(&query)
            .bind(batch_size)
            .bind(queue_table_oid)
            .fetch_all(&self.pool)
            .await?;

        Ok(json_rows)
    }

    async fn record_error(&self, message: &str, details: serde_json::Value) -> Result<()> {
        sqlx::query(&format!(
            "INSERT INTO \"{}\".\"{}\" (id, message, details) VALUES ($1, $2, $3)",
            self.vectorizer.errors_schema, self.vectorizer.errors_table
        ))
        .bind(self.vectorizer.id)
        .bind(message)
        .bind(details)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    fn build_join_predicates(&self, table1: &str, table2: &str) -> String {
        self.vectorizer.source_pk.iter()
            .map(|pk| format!("{}.\"{}\" = {}.\"{}\"", table1, pk.attname, table2, pk.attname))
            .collect::<Vec<_>>()
            .join(" AND ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_chunker_config_none() {
        let json = r#"{"implementation": "none"}"#;
        let config: ChunkerConfig = serde_json::from_str(json).unwrap();
        assert!(matches!(config, ChunkerConfig::None));
    }

    #[test]
    fn test_deserialize_chunker_config_recursive() {
        let json = r#"{"implementation": "recursive_character_text_splitter", "chunk_size": 500, "chunk_overlap": 50}"#;
        let config: ChunkerConfig = serde_json::from_str(json).unwrap();
        match config {
            ChunkerConfig::RecursiveCharacterTextSplitter { chunk_size, chunk_overlap, separators, is_separator_regex } => {
                assert_eq!(chunk_size, 500);
                assert_eq!(chunk_overlap, 50);
                assert!(separators.is_none());
                assert!(is_separator_regex.is_none());
            }
            _ => panic!("Expected RecursiveCharacterTextSplitter, got {:?}", config),
        }
    }

    #[test]
    fn test_deserialize_chunker_config_recursive_with_custom_fields() {
        let json = r#"{"implementation": "recursive_character_text_splitter", "chunk_size": 500, "chunk_overlap": 50, "separators": ["\n\n", "\n"], "is_separator_regex": true}"#;
        let config: ChunkerConfig = serde_json::from_str(json).unwrap();
        match config {
            ChunkerConfig::RecursiveCharacterTextSplitter { chunk_size, chunk_overlap, separators, is_separator_regex } => {
                assert_eq!(chunk_size, 500);
                assert_eq!(chunk_overlap, 50);
                assert_eq!(separators, Some(vec!["\n\n".to_string(), "\n".to_string()]));
                assert_eq!(is_separator_regex, Some(true));
            }
            _ => panic!("Expected RecursiveCharacterTextSplitter, got {:?}", config),
        }
    }

    #[test]
    fn test_deserialize_chunker_config_character() {
        let json = r#"{"implementation": "character_text_splitter", "chunk_size": 200, "chunk_overlap": 20, "separator": "\n"}"#;
        let config: ChunkerConfig = serde_json::from_str(json).unwrap();
        match config {
            ChunkerConfig::CharacterTextSplitter { chunk_size, chunk_overlap, separator, is_separator_regex } => {
                assert_eq!(chunk_size, 200);
                assert_eq!(chunk_overlap, 20);
                assert_eq!(separator, "\n");
                assert!(is_separator_regex.is_none());
            }
            _ => panic!("Expected CharacterTextSplitter, got {:?}", config),
        }
    }

    #[test]
    fn test_deserialize_chunker_config_character_with_regex() {
        let json = r#"{"implementation": "character_text_splitter", "chunk_size": 200, "chunk_overlap": 20, "separator": "\\s+", "is_separator_regex": true}"#;
        let config: ChunkerConfig = serde_json::from_str(json).unwrap();
        match config {
            ChunkerConfig::CharacterTextSplitter { separator, is_separator_regex, .. } => {
                assert_eq!(separator, r"\s+");
                assert_eq!(is_separator_regex, Some(true));
            }
            _ => panic!("Expected CharacterTextSplitter, got {:?}", config),
        }
    }

    #[test]
    fn test_deserialize_chunker_config_unknown_falls_to_none() {
        let json = r#"{"implementation": "some_future_splitter", "foo": 42}"#;
        let config: ChunkerConfig = serde_json::from_str(json).unwrap();
        assert!(matches!(config, ChunkerConfig::None));
    }
}