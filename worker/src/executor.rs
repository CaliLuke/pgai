use anyhow::{anyhow, Result};
use sqlx::{Pool, Postgres};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::{info, error, warn, debug};
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

fn split_into_sentences(text: &str, delimiters: &[String]) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        let mut earliest_pos: Option<usize> = None;
        let mut earliest_delim_len: usize = 0;

        for delim in delimiters {
            if let Some(pos) = remaining.find(delim.as_str()) {
                match earliest_pos {
                    None => {
                        earliest_pos = Some(pos);
                        earliest_delim_len = delim.len();
                    }
                    Some(ep) if pos < ep => {
                        earliest_pos = Some(pos);
                        earliest_delim_len = delim.len();
                    }
                    _ => {}
                }
            }
        }

        match earliest_pos {
            Some(pos) => {
                let end = pos + earliest_delim_len;
                let sentence = &remaining[..end];
                if !sentence.is_empty() {
                    sentences.push(sentence.to_string());
                }
                remaining = &remaining[end..];
            }
            None => {
                if !remaining.is_empty() {
                    sentences.push(remaining.to_string());
                }
                break;
            }
        }
    }

    sentences
}

fn merge_short_sentences(sentences: Vec<String>, min_chars: usize) -> Vec<String> {
    if sentences.is_empty() {
        return sentences;
    }

    let mut result: Vec<String> = Vec::new();
    let mut buffer = String::new();

    for sentence in sentences {
        buffer.push_str(&sentence);
        if buffer.chars().count() >= min_chars {
            result.push(buffer);
            buffer = String::new();
        }
    }

    if !buffer.is_empty() {
        if let Some(last) = result.last_mut() {
            last.push_str(&buffer);
        } else {
            result.push(buffer);
        }
    }

    result
}

fn greedy_sentence_chunks(sentences: &[String], chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    if sentences.is_empty() {
        return Vec::new();
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut current_len: usize = 0;

    for (i, sentence) in sentences.iter().enumerate() {
        let s_len = sentence.chars().count();
        if current.is_empty() {
            current.push(i);
            current_len = s_len;
            continue;
        }

        if current_len + s_len > chunk_size {
            let chunk = join_indices(sentences, &current);
            if !chunk.is_empty() {
                chunks.push(chunk);
            }

            current.clear();
            current_len = 0;

            if chunk_overlap > 0 {
                let mut overlap_len = 0usize;
                let mut overlap_start = i;
                while overlap_start > 0 {
                    let candidate = overlap_start - 1;
                    let candidate_len = sentences[candidate].chars().count();
                    if overlap_len + candidate_len > chunk_overlap {
                        break;
                    }
                    overlap_len += candidate_len;
                    overlap_start = candidate;
                }
                for j in overlap_start..i {
                    current.push(j);
                }
                current_len = overlap_len;
            }

            current.push(i);
            current_len += s_len;
        } else {
            current.push(i);
            current_len += s_len;
        }
    }

    if !current.is_empty() {
        let chunk = join_indices(sentences, &current);
        if !chunk.is_empty() {
            chunks.push(chunk);
        }
    }

    chunks
}

fn join_indices(sentences: &[String], indices: &[usize]) -> String {
    indices
        .iter()
        .map(|&i| sentences[i].as_str())
        .collect::<String>()
        .trim()
        .to_string()
}

fn chunk_by_boundaries(sentences: &[String], boundaries: &[usize]) -> Vec<String> {
    if sentences.is_empty() {
        return Vec::new();
    }
    if boundaries.is_empty() {
        return vec![sentences.concat().trim().to_string()];
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut start = 0usize;
    for &b in boundaries {
        if b <= start || b > sentences.len() {
            continue;
        }
        let chunk = sentences[start..b].concat().trim().to_string();
        if !chunk.is_empty() {
            chunks.push(chunk);
        }
        start = b;
    }
    if start < sentences.len() {
        let tail = sentences[start..].concat().trim().to_string();
        if !tail.is_empty() {
            chunks.push(tail);
        }
    }
    chunks
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

fn savgol_smooth(values: &[f32]) -> Vec<f32> {
    if values.len() < 5 {
        return values.to_vec();
    }
    let coeff: [f32; 5] = [-3.0 / 35.0, 12.0 / 35.0, 17.0 / 35.0, 12.0 / 35.0, -3.0 / 35.0];
    let mut out = values.to_vec();
    for i in 2..values.len() - 2 {
        let mut v = 0.0f32;
        for j in 0..5 {
            v += coeff[j] * values[i + j - 2];
        }
        out[i] = v;
    }
    out
}

fn local_minima(values: &[f32]) -> Vec<usize> {
    if values.len() < 3 {
        return Vec::new();
    }
    let mut mins: Vec<usize> = Vec::new();
    for i in 1..values.len() - 1 {
        if values[i] < values[i - 1] && values[i] <= values[i + 1] {
            mins.push(i);
        }
    }
    mins
}

pub struct Executor {
    pool: Pool<Postgres>,
    vectorizer: Vectorizer,
    embedder: Box<dyn Embedder>,
    cancel: CancellationToken,
    tracking: Arc<WorkerTracking>,
}

impl Executor {
    const MAX_EMBED_ATTEMPTS: u32 = 3;

    pub async fn new(
        pool: Pool<Postgres>,
        vectorizer: Vectorizer,
        cancel: CancellationToken,
        tracking: Arc<WorkerTracking>,
    ) -> Result<Self> {
        let embedder = create_embedder(&vectorizer.config.embedding, &pool).await?;
        Ok(Self { pool, vectorizer, embedder, cancel, tracking })
    }

    fn provider_name(&self) -> &'static str {
        match self.vectorizer.config.embedding {
            crate::models::EmbeddingConfig::OpenAI { .. } => "openai",
            crate::models::EmbeddingConfig::Ollama { .. } => "ollama",
            crate::models::EmbeddingConfig::Unknown => "unknown",
        }
    }

    fn model_name(&self) -> &str {
        match &self.vectorizer.config.embedding {
            crate::models::EmbeddingConfig::OpenAI { model, .. } => model,
            crate::models::EmbeddingConfig::Ollama { model, .. } => model,
            crate::models::EmbeddingConfig::Unknown => "unknown",
        }
    }

    fn embedding_error_class(err: &EmbeddingError) -> &'static str {
        if err.is_transient() { "transient" } else { "permanent" }
    }

    fn parse_status_code(message: &str) -> Option<u16> {
        let lower = message.to_ascii_lowercase();
        for marker in ["http ", "status "] {
            if let Some(idx) = lower.find(marker) {
                let rest = &lower[idx + marker.len()..];
                let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                if digits.len() == 3 {
                    if let Ok(code) = digits.parse::<u16>() {
                        return Some(code);
                    }
                }
            }
        }
        None
    }

    fn parse_error_code(message: &str) -> Option<String> {
        let lower = message.to_ascii_lowercase();
        if lower.contains("invalid_api_key") || lower.contains("invalid api key") {
            return Some("invalid_api_key".to_string());
        }
        if lower.contains("rate_limit_exceeded") || lower.contains("rate limit") {
            return Some("rate_limit_exceeded".to_string());
        }
        if lower.contains("model_not_found") || lower.contains("model not found") {
            return Some("model_not_found".to_string());
        }
        if let Some(idx) = lower.find("\"code\":\"") {
            let rest = &message[idx + 8..];
            if let Some(end) = rest.find('"') {
                let code = &rest[..end];
                if !code.is_empty() {
                    return Some(code.to_string());
                }
            }
        }
        None
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
                debug!(
                    vectorizer_id = self.vectorizer.id,
                    total_processed,
                    "Queue drained"
                );
                break;
            }
            total_processed += items_processed;
            info!(
                vectorizer_id = self.vectorizer.id,
                batch_items = items_processed,
                total_processed,
                "Batch complete"
            );
        }

        Ok(total_processed)
    }

    async fn embed_with_retry(&self, chunks: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let provider = self.provider_name();
        let model = self.model_name().to_string();
        let start = std::time::Instant::now();

        for attempt in 1..=Self::MAX_EMBED_ATTEMPTS {
            let inputs = chunks.to_vec();
            match self.embedder.embed(inputs).await {
                Ok(embeddings) => {
                    if attempt > 1 {
                        info!(
                            vectorizer_id = self.vectorizer.id,
                            provider,
                            model = %model,
                            attempt,
                            max_attempts = Self::MAX_EMBED_ATTEMPTS,
                            elapsed_ms = start.elapsed().as_millis() as u64,
                            "Embedding request recovered after retry"
                        );
                    }
                    return Ok(embeddings);
                }
                Err(err) => {
                    let error_class = Self::embedding_error_class(&err);
                    let msg = err.to_string();
                    let status_code = Self::parse_status_code(&msg);
                    let error_code = Self::parse_error_code(&msg);
                    if err.is_transient() && attempt < Self::MAX_EMBED_ATTEMPTS {
                        warn!(
                            vectorizer_id = self.vectorizer.id,
                            provider,
                            model = %model,
                            error_class,
                            attempt,
                            max_attempts = Self::MAX_EMBED_ATTEMPTS,
                            status_code = ?status_code,
                            error_code = ?error_code,
                            "Embedding attempt failed, retrying"
                        );
                        let backoff_secs = 2u64.saturating_pow(attempt.saturating_sub(1));
                        tokio::time::sleep(Duration::from_secs(backoff_secs.min(10))).await;
                        continue;
                    }

                    error!(
                        vectorizer_id = self.vectorizer.id,
                        provider,
                        model = %model,
                        error_class,
                        attempt,
                        max_attempts = Self::MAX_EMBED_ATTEMPTS,
                        status_code = ?status_code,
                        error_code = ?error_code,
                        elapsed_ms = start.elapsed().as_millis() as u64,
                        "Embedding attempts exhausted for batch"
                    );
                    return Err(err);
                }
            }
        }

        Err(EmbeddingError::Transient(anyhow!(
            "internal retry loop exited unexpectedly for vectorizer {}",
            self.vectorizer.id
        )))
    }

    #[tracing::instrument(
        skip(self),
        fields(
            vectorizer_id = self.vectorizer.id,
            provider = self.provider_name(),
            model = %self.model_name()
        )
    )]
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
                    let chunks = self.perform_chunking(&t, &self.vectorizer.config.chunking).await?;
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

        debug!(
            vectorizer_id = self.vectorizer.id,
            items = items_with_content.len(),
            total_chunks = all_chunks.len(),
            "Chunking complete"
        );

        if all_chunks.is_empty() {
            return Ok(items.len() as i32);
        }

        let embeddings = match self.embed_with_retry(&all_chunks).await {
            Ok(embs) => embs,
            Err(e) => {
                let err_msg = e.to_string();
                let error_class = Self::embedding_error_class(&e);
                let status_code = Self::parse_status_code(&err_msg);
                let error_code = Self::parse_error_code(&err_msg);
                error!(
                    vectorizer_id = self.vectorizer.id,
                    provider = self.provider_name(),
                    model = %self.model_name(),
                    error_class,
                    attempt = Self::MAX_EMBED_ATTEMPTS,
                    max_attempts = Self::MAX_EMBED_ATTEMPTS,
                    status_code = ?status_code,
                    error_code = ?error_code,
                    "Embedding batch failed"
                );
                self.tracking.save_vectorizer_error(Some(self.vectorizer.id), &err_msg).await;
                if let Err(record_err) = self.record_error(
                    "embedding provider failed",
                    serde_json::json!({
                        "provider": self.provider_name(),
                        "model": self.model_name(),
                        "error_class": error_class,
                        "status_code": status_code,
                        "error_code": error_code,
                        "max_attempts": Self::MAX_EMBED_ATTEMPTS,
                        "error_reason": err_msg,
                    }),
                ).await {
                    error!("Failed to record error: {}", record_err);
                }
                return Err(e.into_inner());
            }
        };
        self.write_results(&items_with_content, &item_chunk_counts, &all_chunks, &embeddings).await?;
        debug!(
            vectorizer_id = self.vectorizer.id,
            items = items_with_content.len(),
            chunks = all_chunks.len(),
            "Results written to database"
        );
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

    async fn perform_chunking(&self, text: &str, config: &ChunkerConfig) -> Result<Vec<String>> {
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
                min_sentences_per_chunk,
            } => {
                let mut chunker =
                    pgai_text_splitter::SentenceChunker::new(*chunk_size, *chunk_overlap);
                if let Some(delims) = delimiters {
                    chunker.delimiters = delims.clone();
                }
                if let Some(min_chars) = min_characters_per_sentence {
                    chunker.min_characters_per_sentence = *min_chars;
                }
                if let Some(min_sentences) = min_sentences_per_chunk {
                    chunker.min_sentences_per_chunk = *min_sentences;
                }
                Ok(chunker.split_text(text))
            }
            ChunkerConfig::Semchunk {
                chunk_size,
                chunk_overlap,
                memoize,
                strict_mode,
            } => {
                let mut splitter =
                    pgai_text_splitter::SemchunkSplitter::new(*chunk_size, *chunk_overlap);
                if let Some(m) = memoize {
                    splitter.memoize = *m;
                }
                if let Some(s) = strict_mode {
                    splitter.strict_mode = *s;
                }
                Ok(splitter.split_text(text))
            }
            ChunkerConfig::SemanticChunker {
                chunk_size,
                chunk_overlap,
                window_size,
                skip_window,
                reconnect_similarity_threshold,
                max_aside_length,
                delimiters,
                min_characters_per_sentence,
            } => {
                self.perform_semantic_chunking(
                    text,
                    *chunk_size,
                    *chunk_overlap,
                    window_size.unwrap_or(3),
                    skip_window.unwrap_or(0),
                    reconnect_similarity_threshold.unwrap_or(0.75),
                    max_aside_length.unwrap_or(512),
                    delimiters.clone().unwrap_or_else(|| vec![". ".to_string(), "! ".to_string(), "? ".to_string(), "\n".to_string()]),
                    min_characters_per_sentence.unwrap_or(12),
                ).await
            }
            ChunkerConfig::None => Ok(vec![text.to_string()]),
        }
    }

    async fn perform_semantic_chunking(
        &self,
        text: &str,
        chunk_size: usize,
        chunk_overlap: usize,
        window_size: usize,
        skip_window: usize,
        reconnect_similarity_threshold: f32,
        max_aside_length: usize,
        delimiters: Vec<String>,
        min_characters_per_sentence: usize,
    ) -> Result<Vec<String>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mut sentences = split_into_sentences(text, &delimiters);
        sentences = merge_short_sentences(sentences, min_characters_per_sentence);
        if sentences.is_empty() {
            return Ok(Vec::new());
        }

        if window_size == 0 || sentences.len() <= window_size {
            return Ok(greedy_sentence_chunks(&sentences, chunk_size, chunk_overlap));
        }

        let windows: Vec<String> = (0..=sentences.len() - window_size)
            .map(|i| sentences[i..i + window_size].concat())
            .collect();
        if windows.len() < 2 {
            return Ok(greedy_sentence_chunks(&sentences, chunk_size, chunk_overlap));
        }

        let embeddings = self
            .embed_with_retry(&windows)
            .await
            .map_err(|e| e.into_inner())?;
        let similarities: Vec<f32> = embeddings
            .windows(2)
            .map(|pair| cosine_similarity(&pair[0], &pair[1]))
            .collect();
        let smoothed = savgol_smooth(&similarities);
        let minima = local_minima(&smoothed);
        let mut boundaries: Vec<usize> = minima
            .into_iter()
            .map(|i| i + window_size)
            .filter(|&b| b > 0 && b < sentences.len())
            .collect();
        boundaries.sort_unstable();
        boundaries.dedup();

        let mut chunks = chunk_by_boundaries(&sentences, &boundaries);
        if chunks.is_empty() {
            chunks.push(sentences.concat().trim().to_string());
        }

        if skip_window > 0 && chunks.len() >= 3 {
            chunks = self
                .reconnect_skip_windows_with_embedder(
                    chunks,
                    skip_window,
                    reconnect_similarity_threshold,
                    max_aside_length,
                )
                .await?;
        }

        let mut final_chunks: Vec<String> = Vec::new();
        for chunk in chunks {
            if chunk.chars().count() <= chunk_size {
                final_chunks.push(chunk);
                continue;
            }
            let sub_sentences = split_into_sentences(&chunk, &delimiters);
            final_chunks.extend(greedy_sentence_chunks(
                &sub_sentences,
                chunk_size,
                chunk_overlap,
            ));
        }

        Ok(final_chunks)
    }

    async fn reconnect_skip_windows_with_embedder(
        &self,
        chunks: Vec<String>,
        skip_window: usize,
        threshold: f32,
        max_aside_length: usize,
    ) -> Result<Vec<String>> {
        if chunks.len() < 3 || skip_window == 0 {
            return Ok(chunks);
        }

        let embeddings = self
            .embed_with_retry(&chunks)
            .await
            .map_err(|e| e.into_inner())?;

        let mut out: Vec<String> = Vec::new();
        let mut i = 0usize;
        while i < chunks.len() {
            let mut best_end: Option<usize> = None;
            let max_gap = skip_window.min(chunks.len().saturating_sub(i + 2));
            for gap in 1..=max_gap {
                let j = i + gap + 1;
                let aside_len: usize = chunks[i + 1..j].iter().map(|c| c.chars().count()).sum();
                if aside_len > max_aside_length {
                    continue;
                }
                let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
                if sim >= threshold {
                    best_end = Some(j);
                }
            }

            if let Some(end) = best_end {
                out.push(chunks[i..=end].concat());
                i = end + 1;
            } else {
                out.push(chunks[i].clone());
                i += 1;
            }
        }
        Ok(out)
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
                    let val = item.get(&pk.attname).ok_or_else(|| {
                        anyhow!(
                            "Missing PK value '{}' for vectorizer {}",
                            pk.attname,
                            self.vectorizer.id
                        )
                    })?;
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
                        .ok_or_else(|| {
                            anyhow!(
                                "Missing PK value '{}' for vectorizer {}",
                                pk.attname,
                                self.vectorizer.id
                            )
                        })?;
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
            let (where_clause, pk_vals) = self.build_pk_predicates(item, 2)?;
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
    ) -> Result<(String, Vec<&'a serde_json::Value>)> {
        Self::build_pk_predicates_for_item(self.vectorizer.id, &self.vectorizer.source_pk, item, offset)
    }

    fn build_pk_predicates_for_item<'a>(
        vectorizer_id: i32,
        source_pk: &[crate::models::PkAtt],
        item: &'a serde_json::Value,
        offset: usize,
    ) -> Result<(String, Vec<&'a serde_json::Value>)> {
        let mut parts = Vec::new();
        let mut values = Vec::new();
        for (i, pk) in source_pk.iter().enumerate() {
            parts.push(format!("\"{}\" = ${}", pk.attname, offset + i));
            let value = item.get(&pk.attname).ok_or_else(|| {
                anyhow!(
                    "Missing PK value '{}' for vectorizer {}",
                    pk.attname,
                    vectorizer_id
                )
            })?;
            values.push(value);
        }
        Ok((parts.join(" AND "), values))
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
    use crate::models::{EmbeddingConfig, PkAtt};

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

    #[test]
    fn test_deserialize_chunker_config_sentence_chunker_with_options() {
        let json = r#"{"implementation":"sentence_chunker","chunk_size":256,"chunk_overlap":32,"delimiters":[". ","\n"],"min_characters_per_sentence":10,"min_sentences_per_chunk":2}"#;
        let config: ChunkerConfig = serde_json::from_str(json).unwrap();
        match config {
            ChunkerConfig::SentenceChunker {
                chunk_size,
                chunk_overlap,
                delimiters,
                min_characters_per_sentence,
                min_sentences_per_chunk,
            } => {
                assert_eq!(chunk_size, 256);
                assert_eq!(chunk_overlap, 32);
                assert_eq!(delimiters, Some(vec![". ".to_string(), "\n".to_string()]));
                assert_eq!(min_characters_per_sentence, Some(10));
                assert_eq!(min_sentences_per_chunk, Some(2));
            }
            _ => panic!("Expected SentenceChunker, got {:?}", config),
        }
    }

    #[test]
    fn test_deserialize_chunker_config_semchunk_with_options() {
        let json = r#"{"implementation":"semchunk","chunk_size":300,"chunk_overlap":40,"memoize":false,"strict_mode":true}"#;
        let config: ChunkerConfig = serde_json::from_str(json).unwrap();
        match config {
            ChunkerConfig::Semchunk {
                chunk_size,
                chunk_overlap,
                memoize,
                strict_mode,
            } => {
                assert_eq!(chunk_size, 300);
                assert_eq!(chunk_overlap, 40);
                assert_eq!(memoize, Some(false));
                assert_eq!(strict_mode, Some(true));
            }
            _ => panic!("Expected Semchunk, got {:?}", config),
        }
    }

    #[test]
    fn test_deserialize_chunker_config_semantic_chunker_with_options() {
        let json = r#"{"implementation":"semantic_chunker","chunk_size":320,"chunk_overlap":40,"window_size":3,"skip_window":1,"reconnect_similarity_threshold":0.8,"max_aside_length":256,"delimiters":[". ","\n"],"min_characters_per_sentence":6}"#;
        let config: ChunkerConfig = serde_json::from_str(json).unwrap();
        match config {
            ChunkerConfig::SemanticChunker {
                chunk_size,
                chunk_overlap,
                window_size,
                skip_window,
                reconnect_similarity_threshold,
                max_aside_length,
                delimiters,
                min_characters_per_sentence,
            } => {
                assert_eq!(chunk_size, 320);
                assert_eq!(chunk_overlap, 40);
                assert_eq!(window_size, Some(3));
                assert_eq!(skip_window, Some(1));
                assert_eq!(reconnect_similarity_threshold, Some(0.8));
                assert_eq!(max_aside_length, Some(256));
                assert_eq!(delimiters, Some(vec![". ".to_string(), "\n".to_string()]));
                assert_eq!(min_characters_per_sentence, Some(6));
            }
            _ => panic!("Expected SemanticChunker, got {:?}", config),
        }
    }

    #[test]
    fn test_deserialize_ollama_config_with_max_tokens() {
        let json = r#"{"implementation": "ollama", "model": "nomic-embed-text", "base_url": "http://localhost:11434", "max_tokens": 2048}"#;
        let config: EmbeddingConfig = serde_json::from_str(json).unwrap();
        match config {
            EmbeddingConfig::Ollama { model, base_url, max_tokens } => {
                assert_eq!(model, "nomic-embed-text");
                assert_eq!(base_url, Some("http://localhost:11434".to_string()));
                assert_eq!(max_tokens, Some(2048));
            }
            _ => panic!("Expected Ollama, got {:?}", config),
        }
    }

    #[test]
    fn test_deserialize_ollama_config_without_max_tokens() {
        let json = r#"{"implementation": "ollama", "model": "nomic-embed-text"}"#;
        let config: EmbeddingConfig = serde_json::from_str(json).unwrap();
        match config {
            EmbeddingConfig::Ollama { model, base_url, max_tokens } => {
                assert_eq!(model, "nomic-embed-text");
                assert!(base_url.is_none());
                assert!(max_tokens.is_none());
            }
            _ => panic!("Expected Ollama, got {:?}", config),
        }
    }

    #[test]
    fn test_build_pk_predicates_missing_pk_returns_error() {
        let item = serde_json::json!({"not_id": 1});
        let result = Executor::build_pk_predicates_for_item(
            99,
            &[PkAtt {
                attname: "id".to_string(),
                pknum: 1,
                attnum: 1,
            }],
            &item,
            2,
        );
        assert!(result.is_err(), "missing PK should return an error");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Missing PK value 'id' for vectorizer 99"),
            "got: {err}"
        );
    }

    #[test]
    fn test_parse_status_code_from_http_message() {
        let code = Executor::parse_status_code("HTTP 503 Service Unavailable");
        assert_eq!(code, Some(503));
    }

    #[test]
    fn test_parse_status_code_from_status_message() {
        let code = Executor::parse_status_code("request failed with status 429");
        assert_eq!(code, Some(429));
    }

    #[test]
    fn test_parse_error_code_from_known_patterns() {
        let code = Executor::parse_error_code("Error: invalid api key provided");
        assert_eq!(code.as_deref(), Some("invalid_api_key"));
    }

    #[test]
    fn test_parse_error_code_from_json_payload() {
        let code = Executor::parse_error_code(r#"{"error":{"code":"rate_limit_exceeded"}}"#);
        assert_eq!(code.as_deref(), Some("rate_limit_exceeded"));
    }
}
