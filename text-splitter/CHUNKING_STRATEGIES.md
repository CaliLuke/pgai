# Chunking Strategies Report

Survey of chunking approaches from Chonkie, semchunk, and LangChain, evaluated
for potential implementation in `pgai-text-splitter`.

## Current State

We have two splitters ported from LangChain:

- **CharacterTextSplitter** — split on a single separator, merge pieces greedily
  up to `chunk_size` characters, with `chunk_overlap` and `is_separator_regex`.
- **RecursiveCharacterTextSplitter** — try separators in priority order
  (`\n\n`, `\n`, ` `, ``), recurse into oversized pieces with finer separators.

Both measure size in **characters**, not tokens.

---

## Strategies Worth Implementing

### 1. Sentence Chunker (from Chonkie)

**What:** Split text at sentence boundaries, then greedily pack whole sentences
into chunks up to a token budget.

**Algorithm:**

1. Split on sentence-ending delimiters (`.`, `!`, `?`, `\n`).
2. Count tokens for each sentence.
3. Greedily merge consecutive sentences until the next would exceed `chunk_size`.
4. For overlap: backtrack from each split point, including sentences that fit
   within the overlap token budget.

**Why it matters:** Most RAG content is prose. Sentence-aware chunking keeps
complete thoughts together instead of cutting mid-sentence. This directly
improves retrieval quality since search queries are typically about concepts
that align with sentence boundaries.

**Parameters:** `chunk_size` (tokens), `chunk_overlap`, `min_sentences_per_chunk`,
`delimiters`, `min_characters_per_sentence`.

**Complexity:** Low — straightforward to implement in Rust.

**Priority: High.**

---

### 2. Semantic Recursive Splitter (from semchunk)

**What:** A single recursive algorithm with a fine-grained, fixed hierarchy of
30+ delimiter types ordered by semantic importance.

**Algorithm:**

1. Find the highest-priority delimiter present in the text:
   - Newline sequences (longest first: `\n\n\n` > `\n\n` > `\n`)
   - Tabs
   - Whitespace (longest sequence first)
   - Sentence terminators: `. ? ! *`
   - Clause separators: `; , ( ) [ ] " '`
   - Sentence interrupters: `: -- ...`
   - Word joiners: `/ \ & -`
   - Individual characters (final fallback)
2. Split on that delimiter.
3. Greedily merge splits using adaptive binary search (maintains a running
   chars-per-token ratio to minimize tokenizer calls).
4. If any merged chunk still exceeds the limit, recurse with the next
   delimiter level.
5. Reattach non-whitespace delimiters (like `.`) to the preceding chunk.

**Key insight over LangChain:** Where LangChain jumps from `\n` to ` ` to `""`,
semchunk has nuanced priority among punctuation types. It also prefers the
_longest_ sequence of a delimiter type (so `\n\n` beats `\n`), and it handles
punctuation-preceded spaces specially (splitting `". "` keeps the period with
its sentence).

**Key insight over Chonkie:** semchunk's adaptive binary search in `merge_splits`
uses a running chars-per-token ratio estimate to predict merge points, then
verifies with the actual tokenizer. This minimizes expensive tokenizer calls
compared to Chonkie's linear scan.

**Parameters:** `chunk_size` (tokens), `token_counter` (callable),
`overlap`, `memoize`.

**Complexity:** Medium — the algorithm is well-defined, main effort is the
adaptive merge and delimiter hierarchy.

**Priority: High.**

---

### 3. Token-Aware Measurement (cross-cutting)

Both semchunk and Chonkie measure chunk size in **tokens**, not characters.
This matters because:

- LLMs have token limits, not character limits.
- A 512-character chunk might be 100 or 200 tokens depending on content.
- Token-based sizing guarantees chunks fit model context windows.

This should be a configurable option on all our splitters: accept either a
character count or a token-counting function.

**Priority: High** (can be added incrementally).

---

### 4. Semantic Chunker (from Chonkie)

**What:** Use embedding similarity between consecutive sentence windows to
detect topic boundaries.

**Algorithm:**

1. Split text into sentences.
2. Compute embeddings for sliding windows of N sentences (default 3).
3. Calculate cosine similarity between consecutive windows.
4. Apply Savitzky-Golay smoothing to the similarity curve to reduce noise.
5. Detect local minima — these are semantic boundary candidates.
6. Optional skip-window: reconnect non-adjacent groups that are still
   semantically similar (handles tangential asides).
7. Enforce `chunk_size` by subdividing any oversized chunks.

**Why it's better:** Delimiter-based approaches detect _structural_ boundaries
(paragraphs, sentences). Semantic chunking detects _meaning_ boundaries — it
can tell when the topic actually changes within a paragraph or when two
paragraphs are really about the same thing.

**Tradeoff:** Requires an embedding model at chunking time. For a vectorizer
worker that already has embedding infrastructure, this is feasible but adds
latency. Best offered as an opt-in advanced mode.

**Priority: Medium** — high impact but requires embedding infrastructure at
chunk time.

---

### 5. Code Chunker (from Chonkie)

**What:** AST-aware splitting using tree-sitter. Keeps functions, classes, and
blocks together.

**Algorithm:**

1. Parse code into an AST via tree-sitter (supports language auto-detection).
2. Recursively traverse child nodes, accumulating into groups up to `chunk_size`.
3. If a single node exceeds the limit, recurse into its children.
4. Reconstruct text from byte offsets to preserve formatting.

**Why it matters:** Regex-based code splitting (LangChain's approach with
language-specific separators like `\nfn`, `\nclass`) is brittle. It can
split inside string literals, break nested blocks, and doesn't understand
scope.

**Priority: Low** — our vectorizer primarily processes text columns, not source
code.

---

## Strategies Not Worth Implementing

| Strategy           | Source  | Reason to skip                                                                                                                           |
| ------------------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **FastChunker**    | Chonkie | Byte-level SIMD splitting. We're already in Rust — native speed. Sacrifices token accuracy.                                              |
| **LateChunker**    | Chonkie | Embed entire document first, then slice embeddings per chunk. Requires full-doc embedding just to chunk — defeats the pipeline model.    |
| **NeuralChunker**  | Chonkie | Trained DistilBERT/ModernBERT for boundary detection. Requires specialized ML model at chunk time — too heavy.                           |
| **SlumberChunker** | Chonkie | LLM call per chunk operation. Absurdly expensive for a vectorizer processing millions of rows.                                           |
| **TableChunker**   | Chonkie | Markdown table row splitting. Very niche; could be added later if needed.                                                                |
| **TokenChunker**   | Chonkie | Simple sliding window over tokens. Equivalent to our CharacterTextSplitter but token-counted. Covered by adding token-aware measurement. |

---

## Implementation Roadmap

### Phase 1: Token-aware measurement

Add an optional `token_counter: fn(&str) -> usize` parameter to existing
splitters. When provided, use it instead of `str.chars().count()`. This
upgrades both `CharacterTextSplitter` and `RecursiveCharacterTextSplitter`
from character-based to token-based without changing their algorithms.

### Phase 2: Sentence chunker

New `SentenceChunker` struct. Simple greedy sentence-packing with
configurable delimiters and minimum sentence count. Token-aware from the
start.

### Phase 3: Semantic recursive splitter (semchunk-style)

New `SemanticRecursiveSplitter` with the fixed 30+ delimiter hierarchy,
adaptive binary search merging, and punctuation reattachment. This replaces
the need for users to manually configure separator lists.

### Phase 4: Semantic chunker (embedding-based)

New `SemanticChunker` that takes an embedding function, computes similarity
between sentence windows, and splits at meaning boundaries. Requires the
Savitzky-Golay filter (a small numerical computation — no external dep needed).

---

## Design Decisions

**Length function abstraction:** Both semchunk and Chonkie accept a callable
for measuring text length. We should do the same — a `Fn(&str) -> usize` that
defaults to `str.chars().count()` but can be swapped for a tokenizer.

**Memoization:** semchunk memoizes token counter calls with LRU cache. In Rust,
we can use a `HashMap<String, usize>` or a bounded LRU. This matters for the
adaptive binary search where the same substrings get measured repeatedly.

**Delimiter attachment:** Both Chonkie and semchunk take care to attach
punctuation to the correct chunk. Our `KeepSeparator::{Start, End}` enum
already supports this — we should expose it more prominently.

**Overlap:** semchunk's stride-based overlap (chunk into small pieces, then
create overlapping windows) is cleaner than LangChain's approach of popping
from a running buffer. It guarantees overlap boundaries align with split
points.
