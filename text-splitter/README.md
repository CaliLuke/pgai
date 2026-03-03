# pgai-text-splitter

`pgai-text-splitter` is a Rust library for chunking text for retrieval and embedding pipelines.

It provides multiple splitter strategies, from simple character-based chunking to semantic (embedding-aware) chunking.

## Features

- Character-based splitters:
  - `CharacterTextSplitter`
  - `RecursiveCharacterTextSplitter`
- Sentence-aware splitters:
  - `SentenceChunker`
- Semchunk-inspired recursive splitter:
  - `SemchunkSplitter`
- Embedding-boundary splitter:
  - `SemanticChunker`
- Pluggable length function (`LengthFn`) for character, word, or tokenizer-based sizing.
- Pluggable embedding function (`EmbeddingFn`) for semantic chunking.

## Splitters

### CharacterTextSplitter

LangChain-style single-separator splitting + merge with overlap.

Best for simple and predictable chunk boundaries.

### RecursiveCharacterTextSplitter

LangChain-style recursive fallback over separators (`\n\n` -> `\n` -> ` ` -> char fallback).

Best general-purpose default when text structure varies.

### SentenceChunker

Sentence-aware chunking with sentence-preserving overlap backtracking.

Supports:
- `min_characters_per_sentence`
- `min_sentences_per_chunk`
- custom sentence delimiters

### SemchunkSplitter

Semchunk-inspired recursive splitter with punctuation-aware hierarchy.

Supports:
- adaptive merge (binary-search span fitting)
- optional memoization (`memoize`)
- stricter delimiter precedence mode (`strict_mode`)
- configurable `length_fn` (including tokenizers)

### SemanticChunker

Embedding-similarity boundary detection:
- sentence windows
- cosine similarity between adjacent windows
- Savitzky-Golay smoothing
- local minima boundary detection
- optional skip-window reconnection for tangential asides

Falls back to sentence-based greedy splitting if no `embedding_fn` is provided.

## Quick Start

```rust
use pgai_text_splitter::{RecursiveCharacterTextSplitter, SentenceChunker, SemchunkSplitter, SemanticChunker};

let text = "Hello world. This is a document with multiple sentences.";

let recursive = RecursiveCharacterTextSplitter::new(500, 50);
let recursive_chunks = recursive.split_text(text);

let sentence = SentenceChunker::new(500, 50);
let sentence_chunks = sentence.split_text(text);

let semchunk = SemchunkSplitter::new(500, 50);
let semchunk_chunks = semchunk.split_text(text);

let semantic = SemanticChunker::new(500, 50);
let semantic_chunks = semantic.split_text(text);
```

## Token-Aware Length Example

```rust
use pgai_text_splitter::SemchunkSplitter;

fn word_len(s: &str) -> usize {
    s.split_whitespace().count()
}

let splitter = SemchunkSplitter {
    chunk_size: 90,
    chunk_overlap: 15,
    length_fn: Some(Box::new(word_len)),
    ..SemchunkSplitter::new(90, 15)
};

let chunks = splitter.split_text("Some longer text...");
```

## SemanticChunker Example

```rust
use pgai_text_splitter::SemanticChunker;

fn simple_embedding(text: &str) -> Vec<f32> {
    let lower = text.to_lowercase();
    let a = ["sql", "table", "vectorizer"].iter().map(|k| lower.matches(k).count() as f32).sum::<f32>();
    let b = ["weather", "rain", "forecast"].iter().map(|k| lower.matches(k).count() as f32).sum::<f32>();
    vec![a, b]
}

let chunker = SemanticChunker {
    chunk_size: 500,
    chunk_overlap: 50,
    window_size: 3,
    skip_window: 1,
    reconnect_similarity_threshold: 0.75,
    max_aside_length: 512,
    embedding_fn: Some(Box::new(simple_embedding)),
    ..SemanticChunker::new(500, 50)
};

let chunks = chunker.split_text("...");
```

## Testing

```bash
cargo test -p pgai-text-splitter
```

Integration tests use fixtures under:
- `examples/summarize_article.sql`
- `examples/embeddings_from_documents/documents/pgai.md`

## Benchmarks

```bash
cargo bench -p pgai-text-splitter --bench splitters_bench
```

The benchmark compares all splitter strategies on markdown and SQL fixtures,
including word-count and `tiktoken` length profiles for semchunk.
