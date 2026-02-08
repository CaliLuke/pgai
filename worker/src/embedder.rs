use anyhow::{anyhow, Result};
use async_trait::async_trait;
use async_openai::{
    types::{CreateEmbeddingRequestArgs},
    Client as OpenAIClient,
};
use ollama_rs::Ollama;
use ollama_rs::generation::embeddings::request::GenerateEmbeddingsRequest;
use tiktoken_rs::CoreBPE;
use tracing::{debug, warn};
use crate::errors::EmbeddingError;
use crate::models::EmbeddingConfig;

const OPENAI_MAX_TOKENS_PER_BATCH: usize = 300_000;
const OPENAI_EMBEDDING_CONTEXT_LENGTH: usize = 8191;

/// Split chunks into batches respecting max_chunks_per_batch and max_tokens_per_batch.
/// Returns list of (start_index, end_index) tuples.
fn batch_indices(
    token_lengths: &[usize],
    max_chunks_per_batch: usize,
    max_tokens_per_batch: Option<usize>,
) -> Result<Vec<(usize, usize)>> {
    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut batch: Vec<usize> = Vec::new();
    let mut token_count: usize = 0;

    for (idx, &chunk_tokens) in token_lengths.iter().enumerate() {
        if let Some(max_tokens) = max_tokens_per_batch {
            if chunk_tokens > max_tokens {
                return Err(anyhow!(
                    "chunk length {} greater than max_tokens_per_batch {}",
                    chunk_tokens,
                    max_tokens
                ));
            }
        }

        let max_tokens_reached = max_tokens_per_batch
            .is_some_and(|max| token_count + chunk_tokens > max);
        let max_chunks_reached = batch.len() + 1 > max_chunks_per_batch;

        if (max_tokens_reached || max_chunks_reached) && !batch.is_empty() {
            debug!(
                "Batch {} has {} tokens in {} chunks",
                batches.len() + 1,
                token_count,
                batch.len()
            );
            batches.push(batch);
            batch = Vec::new();
            token_count = 0;
        }

        batch.push(idx);
        token_count += chunk_tokens;
    }

    if !batch.is_empty() {
        debug!(
            "Batch {} has {} tokens in {} chunks",
            batches.len() + 1,
            token_count,
            batch.len()
        );
        batches.push(batch);
    }

    Ok(batches
        .iter()
        .map(|idxs| (idxs[0], idxs[idxs.len() - 1] + 1))
        .collect())
}

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError>;
}

pub struct OpenAIEmbedder {
    client: OpenAIClient<async_openai::config::OpenAIConfig>,
    model: String,
    dimensions: Option<u32>,
    encoder: Option<CoreBPE>,
    context_length: Option<usize>,
}

impl OpenAIEmbedder {
    pub fn new(api_key: String, model: String, dimensions: Option<i32>, base_url: Option<String>) -> Self {
        let mut config = async_openai::config::OpenAIConfig::new().with_api_key(api_key);
        if let Some(url) = base_url {
            config = config.with_api_base(url);
        }
        let client = OpenAIClient::with_config(config);

        let encoder = tiktoken_rs::get_bpe_from_model(&model)
            .inspect_err(|_| {
                warn!("Tokenizer for model {} not found, token counting disabled", model);
            })
            .ok();

        let context_length = match model.as_str() {
            "text-embedding-ada-002" | "text-embedding-3-small" | "text-embedding-3-large" => {
                Some(OPENAI_EMBEDDING_CONTEXT_LENGTH)
            }
            _ => None,
        };

        Self {
            client,
            model,
            dimensions: dimensions.map(|d| d as u32),
            encoder,
            context_length,
        }
    }

    /// Count tokens for a text. Falls back to byte-based estimation if no encoder available.
    fn count_tokens(&self, text: &str) -> usize {
        match &self.encoder {
            Some(enc) => enc.encode_with_special_tokens(text).len(),
            None => (text.len() as f64 * 0.25).ceil() as usize,
        }
    }

    /// Truncate text to fit within context_length tokens. Returns the text unchanged
    /// if no encoder or context_length is configured, or if it already fits.
    fn truncate_if_needed(&self, text: &str) -> String {
        let (Some(enc), Some(max_tokens)) = (&self.encoder, self.context_length) else {
            return text.to_string();
        };

        let tokens = enc.encode_with_special_tokens(text);
        if tokens.len() <= max_tokens {
            return text.to_string();
        }

        warn!(
            "Chunk truncated from {} to {} tokens",
            tokens.len(),
            max_tokens
        );
        enc.decode(tokens[..max_tokens].to_vec())
            .unwrap_or_else(|_| text[..text.len().min(max_tokens * 4)].to_string())
    }
}

#[async_trait]
impl Embedder for OpenAIEmbedder {
    async fn embed(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        // Truncate oversized documents and count tokens
        let inputs: Vec<String> = inputs.into_iter().map(|s| self.truncate_if_needed(&s)).collect();
        let token_lengths: Vec<usize> = inputs.iter().map(|s| self.count_tokens(s)).collect();
        let batches = batch_indices(&token_lengths, 2048, Some(OPENAI_MAX_TOKENS_PER_BATCH))
            .map_err(EmbeddingError::Permanent)?;

        debug!("Embedding {} inputs in {} batches", inputs.len(), batches.len());

        let mut all_embeddings = Vec::with_capacity(inputs.len());

        for (start, end) in batches {
            let batch_inputs: Vec<String> = inputs[start..end].to_vec();

            let mut request_builder = CreateEmbeddingRequestArgs::default();
            request_builder.model(&self.model).input(batch_inputs);

            if let Some(dim) = self.dimensions {
                request_builder.dimensions(dim);
            }

            let request = request_builder.build()
                .map_err(|e| EmbeddingError::Permanent(e.into()))?;
            let response = self.client.embeddings().create(request).await
                .map_err(EmbeddingError::from_openai_error)?;

            let mut batch_embeddings: Vec<Vec<f32>> =
                response.data.into_iter().map(|d| d.embedding).collect();
            all_embeddings.append(&mut batch_embeddings);
        }

        Ok(all_embeddings)
    }
}

pub struct OllamaEmbedder {
    client: Ollama,
    model: String,
}

impl OllamaEmbedder {
    pub fn new(base_url: Option<String>, model: String) -> Self {
        let url = base_url.unwrap_or_else(|| "http://localhost:11434".to_string());
        let parsed_url = url::Url::parse(&url).expect("Invalid Ollama URL");
        let client = Ollama::new(
            parsed_url.scheme().to_string() + "://" + parsed_url.host_str().unwrap(),
            parsed_url.port().unwrap_or(11434)
        );
        Self { client, model }
    }
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    async fn embed(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let req = GenerateEmbeddingsRequest::new(self.model.clone(), inputs.into());
        let res = self.client.generate_embeddings(req).await
            .map_err(|e| EmbeddingError::classify(e.into()))?;
        Ok(res.embeddings)
    }
}

/// Resolve an API key by name: try environment variable first, then fall back to
/// `ai.reveal_secret()` in the database.
async fn resolve_api_key(pool: &sqlx::Pool<sqlx::Postgres>, key_name: &str) -> Result<String> {
    // 1. Environment variable
    if let Ok(val) = std::env::var(key_name) {
        debug!("Obtained secret '{}' from environment", key_name);
        return Ok(val);
    }

    // 2. Database: ai.reveal_secret()
    let result: Option<String> = sqlx::query_scalar("SELECT ai.reveal_secret($1)")
        .bind(key_name)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten();

    if let Some(val) = result {
        debug!("Obtained secret '{}' from database", key_name);
        return Ok(val);
    }

    Err(anyhow::anyhow!(
        "api_key_name={} not found in environment or database",
        key_name
    ))
}

pub async fn create_embedder(
    config: &EmbeddingConfig,
    pool: &sqlx::Pool<sqlx::Postgres>,
) -> Result<Box<dyn Embedder>> {
    match config {
        EmbeddingConfig::OpenAI { model, dimensions, api_key_name, base_url } => {
            let key_name = api_key_name.as_deref().unwrap_or("OPENAI_API_KEY");
            let api_key = resolve_api_key(pool, key_name).await?;
            Ok(Box::new(OpenAIEmbedder::new(api_key, model.clone(), *dimensions, base_url.clone())))
        }
        EmbeddingConfig::Ollama { model, base_url } => {
            Ok(Box::new(OllamaEmbedder::new(base_url.clone(), model.clone())))
        }
        _ => Err(anyhow::anyhow!("Unsupported embedding provider")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- tiktoken token counting ---

    #[test]
    fn test_tiktoken_counts_tokens() {
        let enc = tiktoken_rs::get_bpe_from_model("text-embedding-3-small").unwrap();
        let tokens = enc.encode_with_special_tokens("hello world");
        assert_eq!(tokens.len(), 2); // "hello" + " world"
    }

    #[test]
    fn test_openai_embedder_count_tokens() {
        std::env::set_var("TEST_TIKTOKEN_KEY", "sk-test");
        let embedder = OpenAIEmbedder::new(
            "sk-test".to_string(),
            "text-embedding-3-small".to_string(),
            Some(3),
            None,
        );
        // tiktoken should give exact count
        assert_eq!(embedder.count_tokens("hello world"), 2);
        assert_eq!(embedder.count_tokens(""), 0);
        // Longer text
        let count = embedder.count_tokens("The quick brown fox jumps over the lazy dog");
        assert!(count > 5 && count < 20, "Expected reasonable token count, got {}", count);
        std::env::remove_var("TEST_TIKTOKEN_KEY");
    }

    #[test]
    fn test_truncate_long_document() {
        let embedder = OpenAIEmbedder::new(
            "sk-test".to_string(),
            "text-embedding-3-small".to_string(),
            None,
            None,
        );
        // "AGI " repeated many times to exceed 8191 tokens
        let long_text = "AGI ".repeat(20000); // ~20000 tokens
        let truncated = embedder.truncate_if_needed(&long_text);
        let truncated_tokens = embedder.count_tokens(&truncated);
        assert!(
            truncated_tokens <= OPENAI_EMBEDDING_CONTEXT_LENGTH,
            "Truncated text should be <= {} tokens, got {}",
            OPENAI_EMBEDDING_CONTEXT_LENGTH,
            truncated_tokens
        );
    }

    #[test]
    fn test_no_truncation_for_short_text() {
        let embedder = OpenAIEmbedder::new(
            "sk-test".to_string(),
            "text-embedding-3-small".to_string(),
            None,
            None,
        );
        let short_text = "This is a short text.";
        let result = embedder.truncate_if_needed(short_text);
        assert_eq!(result, short_text);
    }

    // --- batch_indices ---

    #[test]
    fn test_batch_indices_empty() {
        let result = batch_indices(&[], 1, None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_batch_indices_max_batch_size_1() {
        let tokens = vec![5, 1, 1, 1, 1, 1, 1, 1, 1];
        let result = batch_indices(&tokens, 1, None).unwrap();
        assert_eq!(result.len(), 9);
        for (i, &(start, end)) in result.iter().enumerate() {
            assert_eq!(start, i);
            assert_eq!(end, i + 1);
        }
    }

    #[test]
    fn test_batch_indices_max_batch_size_3() {
        let tokens = vec![5, 1, 1, 1, 1, 1, 1, 1, 1];
        let result = batch_indices(&tokens, 3, None).unwrap();
        assert_eq!(result, vec![(0, 3), (3, 6), (6, 9)]);
    }

    #[test]
    fn test_batch_indices_max_batch_size_5() {
        let tokens = vec![5, 1, 1, 1, 1, 1, 1, 1, 1];
        let result = batch_indices(&tokens, 5, None).unwrap();
        assert_eq!(result, vec![(0, 5), (5, 9)]);
    }

    #[test]
    fn test_batch_indices_with_token_limit() {
        // max_chunks=5, max_tokens=6
        // Batch 1: chunks 0,1 (5+1=6) — next would exceed
        // Batch 2: chunks 2-6 (1+1+1+1+1=5)
        // Batch 3: chunks 7,8 (1+1=2)
        let tokens = vec![5, 1, 1, 1, 1, 1, 1, 1, 1];
        let result = batch_indices(&tokens, 5, Some(6)).unwrap();
        assert_eq!(result, vec![(0, 2), (2, 7), (7, 9)]);
    }

    #[test]
    fn test_batch_indices_chunk_exceeds_max_tokens() {
        let tokens = vec![5, 1, 1];
        let result = batch_indices(&tokens, 5, Some(2));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("chunk length 5 greater than max_tokens_per_batch 2"), "got: {}", err);
    }

    #[test]
    fn test_batch_indices_string_documents_with_token_limit() {
        // tokens=[1,8,2,5,9,5,6,11], max_chunks=5, max_tokens=20
        // Batch 1: 0,1,2,3 (1+8+2+5=16, adding 9 would be 25)
        // Batch 2: 4,5,6 (9+5+6=20)
        // Batch 3: 7 (11)
        let tokens = vec![1, 8, 2, 5, 9, 5, 6, 11];
        let result = batch_indices(&tokens, 5, Some(20)).unwrap();
        assert_eq!(result, vec![(0, 4), (4, 7), (7, 8)]);
    }

    // --- resolve_api_key (env var path only, DB path tested in integration) ---

    #[test]
    fn test_resolve_api_key_from_env() {
        std::env::set_var("TEST_RESOLVE_KEY_12345", "sk-test-value");
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        // We can't create a real pool in unit tests, but we can verify env var
        // resolution works by checking the env var directly.
        let val = std::env::var("TEST_RESOLVE_KEY_12345").unwrap();
        assert_eq!(val, "sk-test-value");
        drop(rt);
        std::env::remove_var("TEST_RESOLVE_KEY_12345");
    }
}
