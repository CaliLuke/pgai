use pgai_text_splitter::{
    CharacterTextSplitter, RecursiveCharacterTextSplitter, SemchunkSplitter, SentenceChunker,
};
use std::path::PathBuf;

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("examples")
}

#[test]
fn test_recursive_split_sql_file() {
    let path = examples_dir().join("summarize_article.sql");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Could not read {:?}", path));

    let splitter = RecursiveCharacterTextSplitter::new(200, 20);
    let chunks = splitter.split_text(&text);

    assert!(!chunks.is_empty(), "Should produce at least one chunk");
    for chunk in &chunks {
        assert!(
            chunk.chars().count() <= 200,
            "Chunk exceeded chunk_size: {} chars",
            chunk.chars().count()
        );
    }
}

#[test]
fn test_recursive_split_markdown_file() {
    let path = examples_dir().join("embeddings_from_documents/documents/pgai.md");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Could not read {:?}", path));

    let splitter = RecursiveCharacterTextSplitter::new(500, 50);
    let chunks = splitter.split_text(&text);

    assert!(
        chunks.len() > 1,
        "610-line markdown file should produce multiple chunks, got {}",
        chunks.len()
    );
    for chunk in &chunks {
        assert!(
            chunk.chars().count() <= 500,
            "Chunk exceeded chunk_size: {} chars",
            chunk.chars().count()
        );
    }
}

#[test]
fn test_character_split_sql_file() {
    let path = examples_dir().join("summarize_article.sql");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Could not read {:?}", path));

    // Use a large enough chunk_size that individual lines fit
    let splitter = CharacterTextSplitter::new("\n", 500, 0);
    let chunks = splitter.split_text(&text);

    assert!(!chunks.is_empty(), "Should produce at least one chunk");
    assert!(
        chunks.len() > 1,
        "SQL file should produce multiple chunks, got {}",
        chunks.len()
    );
}

#[test]
fn test_sentence_chunker_markdown_file() {
    let path = examples_dir().join("embeddings_from_documents/documents/pgai.md");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Could not read {:?}", path));

    let chunker = SentenceChunker::new(500, 50);
    let chunks = chunker.split_text(&text);

    assert!(
        chunks.len() > 1,
        "Markdown file should produce multiple chunks with SentenceChunker, got {}",
        chunks.len()
    );
    for chunk in &chunks {
        assert!(
            chunk.chars().count() <= 500,
            "SentenceChunker chunk exceeded chunk_size: {} chars",
            chunk.chars().count()
        );
    }
}

#[test]
fn test_sentence_chunker_sql_file() {
    let path = examples_dir().join("summarize_article.sql");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Could not read {:?}", path));

    let chunker = SentenceChunker::new(300, 0);
    let chunks = chunker.split_text(&text);

    assert!(!chunks.is_empty(), "Should produce at least one chunk");
}

#[test]
fn test_semchunk_markdown_file() {
    let path = examples_dir().join("embeddings_from_documents/documents/pgai.md");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Could not read {:?}", path));

    let splitter = SemchunkSplitter::new(500, 50);
    let chunks = splitter.split_text(&text);

    assert!(
        chunks.len() > 1,
        "Markdown file should produce multiple chunks with SemchunkSplitter, got {}",
        chunks.len()
    );
    for chunk in &chunks {
        assert!(
            chunk.chars().count() <= 500,
            "SemchunkSplitter chunk exceeded chunk_size: {} chars",
            chunk.chars().count()
        );
    }
}

#[test]
fn test_semchunk_sql_file() {
    let path = examples_dir().join("summarize_article.sql");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Could not read {:?}", path));

    let splitter = SemchunkSplitter::new(200, 20);
    let chunks = splitter.split_text(&text);

    assert!(!chunks.is_empty(), "Should produce at least one chunk");
    for chunk in &chunks {
        assert!(
            chunk.chars().count() <= 200,
            "SemchunkSplitter chunk exceeded chunk_size: {} chars",
            chunk.chars().count()
        );
    }
}
