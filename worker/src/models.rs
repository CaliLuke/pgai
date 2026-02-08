use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Version {
    pub ext_version: Option<String>,
    pub pgai_lib_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PkAtt {
    pub attname: String,
    pub pknum: i32,
    pub attnum: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vectorizer {
    pub id: i32,
    pub config: Config,
    pub queue_schema: String,
    pub queue_table: String,
    pub source_schema: String,
    pub source_table: String,
    pub source_pk: Vec<PkAtt>,
    pub queue_failed_table: Option<String>,
    #[serde(default = "default_errors_schema")]
    pub errors_schema: String,
    #[serde(default = "default_errors_table")]
    pub errors_table: String,
    #[serde(default)]
    pub disabled: bool,
}

fn default_errors_schema() -> String {
    "ai".to_string()
}

fn default_errors_table() -> String {
    "vectorizer_errors".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub version: String,
    pub embedding: EmbeddingConfig,
    pub chunking: ChunkerConfig,
    pub formatting: FormatterConfig,
    pub loading: LoadingConfig,
    pub destination: DestinationConfig,
    #[serde(default)]
    pub processing: ProcessingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "implementation", rename_all = "lowercase")]
pub enum EmbeddingConfig {
    OpenAI {
        model: String,
        dimensions: Option<i32>,
        api_key_name: Option<String>,
        base_url: Option<String>,
    },
    Ollama {
        model: String,
        base_url: Option<String>,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "implementation")]
pub enum ChunkerConfig {
    #[serde(rename = "recursive_character_text_splitter")]
    RecursiveCharacterTextSplitter {
        chunk_size: usize,
        chunk_overlap: usize,
        #[serde(default)]
        separators: Option<Vec<String>>,
        #[serde(default)]
        is_separator_regex: Option<bool>,
    },
    #[serde(rename = "character_text_splitter")]
    CharacterTextSplitter {
        chunk_size: usize,
        chunk_overlap: usize,
        separator: String,
        #[serde(default)]
        is_separator_regex: Option<bool>,
    },
    #[serde(rename = "sentence_chunker")]
    SentenceChunker {
        chunk_size: usize,
        chunk_overlap: usize,
        #[serde(default)]
        delimiters: Option<Vec<String>>,
        #[serde(default)]
        min_characters_per_sentence: Option<usize>,
    },
    #[serde(rename = "semchunk")]
    Semchunk {
        chunk_size: usize,
        chunk_overlap: usize,
    },
    #[serde(rename = "none")]
    #[serde(other)]
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "implementation", rename_all = "lowercase")]
pub enum FormatterConfig {
    PythonTemplate {
        template: String,
    },
    #[serde(rename = "chunk_value")]
    ChunkValue,
    #[serde(other)]
    Unknown,
}

impl FormatterConfig {
    /// Format a chunk using the configured formatter.
    ///
    /// For `PythonTemplate`, performs `$variable` substitution using `$chunk` and
    /// all columns from the source row item. Mirrors Python's `string.Template.substitute()`.
    ///
    /// For `ChunkValue` (or unknown), returns the chunk as-is.
    pub fn format(&self, chunk: &str, item: &serde_json::Value) -> String {
        match self {
            FormatterConfig::PythonTemplate { template } => {
                let mut result = template.replace("$chunk", chunk);
                // Substitute $column_name for each field in the row
                if let Some(obj) = item.as_object() {
                    for (key, val) in obj {
                        let placeholder = format!("${}", key);
                        if result.contains(&placeholder) {
                            let val_str = match val {
                                serde_json::Value::String(s) => s.clone(),
                                serde_json::Value::Null => String::new(),
                                other => other.to_string(),
                            };
                            result = result.replace(&placeholder, &val_str);
                        }
                    }
                }
                result
            }
            FormatterConfig::ChunkValue | FormatterConfig::Unknown => chunk.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "implementation", rename_all = "lowercase")]
pub enum LoadingConfig {
    Column {
        column_name: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "implementation", rename_all = "lowercase")]
pub enum DestinationConfig {
    Table {
        target_schema: Option<String>,
        target_table: String,
    },
    Column {
        embedding_column: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub batch_size: Option<i32>,
    #[serde(default = "default_concurrency")]
    pub concurrency: i32,
}

fn default_concurrency() -> i32 {
    1
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            batch_size: None,
            concurrency: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_chunk_value_returns_chunk_as_is() {
        let fmt = FormatterConfig::ChunkValue;
        let item = json!({"title": "Hello", "body": "World"});
        assert_eq!(fmt.format("my chunk text", &item), "my chunk text");
    }

    #[test]
    fn test_python_template_chunk_only() {
        let fmt = FormatterConfig::PythonTemplate {
            template: "$chunk".to_string(),
        };
        let item = json!({"id": 1});
        assert_eq!(fmt.format("hello world", &item), "hello world");
    }

    #[test]
    fn test_python_template_with_row_fields() {
        let fmt = FormatterConfig::PythonTemplate {
            template: "Title: $title\n\n$chunk".to_string(),
        };
        let item = json!({"title": "My Doc", "id": 42});
        assert_eq!(fmt.format("body text", &item), "Title: My Doc\n\nbody text");
    }

    #[test]
    fn test_python_template_multiple_fields() {
        let fmt = FormatterConfig::PythonTemplate {
            template: "size: $size shape: $shape $chunk".to_string(),
        };
        let item = json!({"size": "large", "shape": "round"});
        assert_eq!(
            fmt.format("content here", &item),
            "size: large shape: round content here"
        );
    }

    #[test]
    fn test_python_template_numeric_field() {
        let fmt = FormatterConfig::PythonTemplate {
            template: "ID=$id $chunk".to_string(),
        };
        let item = json!({"id": 99});
        assert_eq!(fmt.format("text", &item), "ID=99 text");
    }

    #[test]
    fn test_python_template_null_field() {
        let fmt = FormatterConfig::PythonTemplate {
            template: "val=$val $chunk".to_string(),
        };
        let item = json!({"val": null});
        assert_eq!(fmt.format("text", &item), "val= text");
    }

    #[test]
    fn test_python_template_missing_field_left_as_is() {
        let fmt = FormatterConfig::PythonTemplate {
            template: "missing=$missing $chunk".to_string(),
        };
        let item = json!({"other": "x"});
        // $missing not in item, so left as literal
        assert_eq!(fmt.format("text", &item), "missing=$missing text");
    }

    #[test]
    fn test_unknown_formatter_returns_chunk() {
        let fmt = FormatterConfig::Unknown;
        assert_eq!(fmt.format("raw chunk", &json!({})), "raw chunk");
    }
}