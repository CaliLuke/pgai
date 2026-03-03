use pgai_text_splitter::SemanticChunker;
use reqwest::blocking::Client;
use serde_json::Value;
use std::time::Duration;

fn env_enabled() -> bool {
    std::env::var("PGAI_TEST_OLLAMA")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn ollama_base_url() -> String {
    std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string())
}

fn ollama_model() -> String {
    std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string())
}

fn can_reach_ollama(client: &Client, base_url: &str) -> bool {
    client
        .get(format!("{base_url}/api/tags"))
        .send()
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

fn model_is_available(client: &Client, base_url: &str, model: &str) -> bool {
    let response = match client.get(format!("{base_url}/api/tags")).send() {
        Ok(r) => r,
        Err(_) => return false,
    };
    let body: Value = match response.json() {
        Ok(v) => v,
        Err(_) => return false,
    };

    body.get("models")
        .and_then(|m| m.as_array())
        .map(|models| {
            models.iter().any(|m| {
                let name = m.get("name").and_then(|n| n.as_str()).unwrap_or_default();
                let short = name.split(':').next().unwrap_or_default();
                name == model || short == model
            })
        })
        .unwrap_or(false)
}

fn ollama_embed(client: &Client, base_url: &str, model: &str, text: &str) -> Vec<f32> {
    let payload = serde_json::json!({
        "model": model,
        "prompt": text
    });

    let response = client
        .post(format!("{base_url}/api/embeddings"))
        .json(&payload)
        .send()
        .expect("failed to call ollama embeddings endpoint");

    assert!(
        response.status().is_success(),
        "ollama returned non-success status: {}",
        response.status()
    );

    let body: Value = response.json().expect("failed to decode ollama response");
    let embedding = body
        .get("embedding")
        .and_then(|v| v.as_array())
        .expect("missing embedding array in ollama response");

    embedding
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
        .collect()
}

#[test]
fn test_semantic_chunker_with_ollama_embeddings() {
    if !env_enabled() {
        eprintln!("Skipping Ollama test: set PGAI_TEST_OLLAMA=1 to enable");
        return;
    }

    let base_url = ollama_base_url();
    let model = ollama_model();
    let client = Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .expect("failed to build reqwest client");

    if !can_reach_ollama(&client, &base_url) {
        eprintln!("Skipping Ollama test: cannot reach {}", base_url);
        return;
    }

    if !model_is_available(&client, &base_url, &model) {
        eprintln!(
            "Skipping Ollama test: model '{}' not available on {}",
            model, base_url
        );
        return;
    }

    let embed_client = client.clone();
    let embed_base = base_url.clone();
    let embed_model = model.clone();

    let chunker = SemanticChunker {
        chunk_size: 120,
        chunk_overlap: 10,
        window_size: 2,
        min_characters_per_sentence: 1,
        embedding_fn: Some(Box::new(move |text: &str| {
            ollama_embed(&embed_client, &embed_base, &embed_model, text)
        })),
        ..SemanticChunker::new(120, 10)
    };

    let text = "SQL tables organize rows and columns. Database indexes speed up queries. \
    Weather forecasts predict rain and storms. Temperature trends vary by season.";

    let chunks = chunker.split_text(text);
    assert!(
        chunks.len() >= 2,
        "Expected semantic chunking to produce multiple chunks, got {:?}",
        chunks
    );
    for chunk in &chunks {
        assert!(
            chunk.chars().count() <= 120,
            "Chunk exceeded size budget: {:?}",
            chunk
        );
    }
}
