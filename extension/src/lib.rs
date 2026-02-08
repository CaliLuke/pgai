use pgrx::prelude::*;
use pgrx::datum::DatumWithOid;
use serde_json::Value;
use reqwest::blocking::Client;
use std::time::Duration;
use std::ffi::CString;

pgrx::pg_module_magic!();

// Load the vectorizer SQL infrastructure at CREATE EXTENSION time.
// This must run before any pg_extern functions that reference the ai schema.
pgrx::extension_sql_file!("../sql/setup.sql", name = "setup", bootstrap);

static OPENAI_API_KEY: pgrx::guc::GucSetting<Option<CString>> =
    pgrx::guc::GucSetting::<Option<CString>>::new(None);

#[pg_guard]
pub extern "C-unwind" fn _PG_init() {
    pgrx::guc::GucRegistry::define_string_guc(
        c"ai.openai_api_key",
        c"OpenAI API Key",
        c"The API key for OpenAI services",
        &OPENAI_API_KEY,
        pgrx::guc::GucContext::Userset,
        pgrx::guc::GucFlags::default(),
    );
}

#[pg_extern]
fn openai_embed(
    model: &str,
    input: &str,
    api_key: default!(Option<&str>, "NULL"),
    dimensions: default!(Option<i32>, "NULL"),
) -> Vec<f32> {
    let api_key = match api_key {
        Some(key) => key.to_string(),
        None => {
            match OPENAI_API_KEY.get() {
                Some(s) => s.to_str().expect("invalid UTF-8 in GUC").to_string(),
                None => error!("OpenAI API key not found in GUC 'ai.openai_api_key' or argument"),
            }
        }
    };

    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap_or_else(|e| error!("Failed to create HTTP client: {}", e));

    let mut body = serde_json::json!({
        "model": model,
        "input": input,
    });

    if let Some(dim) = dimensions {
        body.as_object_mut().unwrap().insert("dimensions".to_string(), serde_json::json!(dim));
    }

    let res = client.post("https://api.openai.com/v1/embeddings")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .unwrap_or_else(|e| error!("OpenAI API request failed: {}", e));

    if !res.status().is_success() {
        error!("OpenAI API error: {}", res.status());
    }

    let json: Value = res.json()
        .unwrap_or_else(|e| error!("Failed to parse OpenAI response: {}", e));
    
    let embedding = json["data"][0]["embedding"]
        .as_array()
        .unwrap_or_else(|| error!("Invalid embedding format in response"))
        .iter()
        .map(|v: &Value| v.as_f64().expect("embedding value is not a number") as f32)
        .collect();

    embedding
}

#[pg_extern]
fn ollama_embed(
    model: &str,
    input: &str,
    base_url: default!(Option<&str>, "NULL"),
) -> Vec<f32> {
    let base_url = base_url.unwrap_or("http://localhost:11434");
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap_or_else(|e| error!("Failed to create HTTP client: {}", e));

    let body = serde_json::json!({
        "model": model,
        "prompt": input,
    });

    let res = client.post(format!("{}/api/embeddings", base_url))
        .json(&body)
        .send()
        .unwrap_or_else(|e| error!("Ollama API request failed: {}", e));

    if !res.status().is_success() {
        error!("Ollama API error: {}", res.status());
    }

    let json: Value = res.json()
        .unwrap_or_else(|e| error!("Failed to parse Ollama response: {}", e));
    
    let embedding = json["embedding"]
        .as_array()
        .unwrap_or_else(|| error!("Invalid embedding format in response"))
        .iter()
        .map(|v: &Value| v.as_f64().expect("embedding value is not a number") as f32)
        .collect();

    embedding
}

#[pg_extern]
fn create_vectorizer(
    source_table: &str,
    config: pgrx::JsonB,
) -> i32 {
    let parts: Vec<&str> = source_table.split('.').collect();
    let (source_schema, source_name) = if parts.len() == 2 {
        (parts[0], parts[1])
    } else {
        ("public", parts[0])
    };

    let pk_info = unsafe { Spi::get_one_with_args::<pgrx::JsonB>(
        "SELECT jsonb_agg(jsonb_build_object('attname', a.attname, 'pknum', i.n, 'attnum', a.attnum))
         FROM pg_index i
         CROSS JOIN LATERAL generate_series(1, i.indnatts) n(n)
         JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[n.n-1]
         WHERE i.indrelid = $1::regclass AND i.indisprimary",
        &[DatumWithOid::new(source_table.into_datum(), PgBuiltInOids::TEXTOID.oid().value())],
    ) }.unwrap_or_else(|e| error!("Failed to get PK info: {}", e))
     .unwrap_or_else(|| error!("Table has no primary key"));

    let queue_table = format!("{}_vectorizer_queue", source_name);

    Spi::run(&format!(
        "CREATE TABLE IF NOT EXISTS ai.\"{}\" (LIKE {}.\"{}\" INCLUDING ALL)",
        queue_table, source_schema, source_name
    )).unwrap_or_else(|e| error!("Failed to create queue table: {}", e));

    let id = unsafe { Spi::get_one_with_args::<i32>(
        "INSERT INTO ai.vectorizer (source_schema, source_table, queue_schema, queue_table, config, source_pk)
         VALUES ($1, $2, 'ai', $3, $4, $5)
         RETURNING id",
        &[
            DatumWithOid::new(source_schema.into_datum(), PgBuiltInOids::TEXTOID.oid().value()),
            DatumWithOid::new(source_name.into_datum(), PgBuiltInOids::TEXTOID.oid().value()),
            DatumWithOid::new(queue_table.into_datum(), PgBuiltInOids::TEXTOID.oid().value()),
            DatumWithOid::new(config.into_datum(), PgBuiltInOids::JSONBOID.oid().value()),
            DatumWithOid::new(pk_info.into_datum(), PgBuiltInOids::JSONBOID.oid().value()),
        ],
    ) }.unwrap_or_else(|e| error!("Failed to insert vectorizer: {}", e))
     .unwrap_or_else(|| error!("Failed to get inserted vectorizer id"));

    id
}

#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec![]
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_create_vectorizer_logic() {
        Spi::run("CREATE TABLE public.test_table (id serial primary key, content text)").unwrap();

        let config = serde_json::json!({
            "version": "1.0",
            "embedding": {"implementation": "openai", "model": "text-embedding-3-small"},
            "chunking": {"implementation": "none"},
            "formatting": {"implementation": "chunk_value"},
            "loading": {"implementation": "column", "column_name": "content"},
            "destination": {"implementation": "table", "target_table": "test_embeddings"}
        });

        let id = crate::create_vectorizer("public.test_table", pgrx::JsonB(config));
        assert!(id > 0);

        let count = Spi::get_one::<i64>("SELECT count(*) FROM ai.vectorizer WHERE source_table = 'test_table'").unwrap();
        assert_eq!(count, Some(1));
    }
}