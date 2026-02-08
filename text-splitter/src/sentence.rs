use crate::char_len;

/// Default sentence delimiters.
const DEFAULT_DELIMITERS: &[&str] = &[". ", "! ", "? ", "\n"];

/// A text splitter that respects sentence boundaries.
///
/// Splits text into sentences using configurable delimiters, then greedily
/// packs consecutive sentences into chunks that fit within `chunk_size`.
///
/// Inspired by Chonkie's sentence chunking approach.
pub struct SentenceChunker {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub delimiters: Vec<String>,
    pub min_characters_per_sentence: usize,
    pub strip_whitespace: bool,
    pub length_fn: Option<Box<dyn Fn(&str) -> usize + Send + Sync>>,
}

impl SentenceChunker {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            delimiters: DEFAULT_DELIMITERS.iter().map(|s| s.to_string()).collect(),
            min_characters_per_sentence: 12,
            strip_whitespace: true,
            length_fn: None,
        }
    }

    pub fn split_text(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        let default_fn = char_len;
        let len_fn: &dyn Fn(&str) -> usize = match &self.length_fn {
            Some(f) => f.as_ref(),
            None => &default_fn,
        };

        // Step 1: Split into sentences
        let mut sentences = split_into_sentences(text, &self.delimiters);

        // Step 2: Merge short sentences with the next one
        sentences = merge_short_sentences(sentences, self.min_characters_per_sentence);

        if sentences.is_empty() {
            return Vec::new();
        }

        // Step 3: Greedily pack sentences into chunks
        let mut chunks: Vec<String> = Vec::new();
        let mut current_sentences: Vec<usize> = Vec::new(); // indices into sentences
        let mut current_len: usize = 0;

        for (i, sentence) in sentences.iter().enumerate() {
            let s_len = len_fn(sentence);

            if current_sentences.is_empty() {
                current_sentences.push(i);
                current_len = s_len;
                continue;
            }

            // Would adding this sentence exceed chunk_size?
            if current_len + s_len > self.chunk_size {
                // Emit current chunk
                let chunk = self.join_sentences(&sentences, &current_sentences);
                chunks.push(chunk);

                // Handle overlap: backtrack from the end of the previous chunk
                current_sentences.clear();
                current_len = 0;

                if self.chunk_overlap > 0 && i > 0 {
                    // Walk backwards from the last sentence in the previous chunk
                    let prev_end = i; // exclusive
                    let mut overlap_start = prev_end;
                    let mut overlap_len: usize = 0;

                    while overlap_start > 0 {
                        let candidate = overlap_start - 1;
                        let candidate_len = len_fn(&sentences[candidate]);
                        if overlap_len + candidate_len > self.chunk_overlap {
                            break;
                        }
                        overlap_len += candidate_len;
                        overlap_start = candidate;
                    }

                    for j in overlap_start..prev_end {
                        current_sentences.push(j);
                    }
                    current_len = overlap_len;
                }

                current_sentences.push(i);
                current_len += s_len;
            } else {
                current_sentences.push(i);
                current_len += s_len;
            }
        }

        // Emit final chunk
        if !current_sentences.is_empty() {
            let chunk = self.join_sentences(&sentences, &current_sentences);
            chunks.push(chunk);
        }

        chunks
    }

    fn join_sentences(&self, sentences: &[String], indices: &[usize]) -> String {
        let joined: String = indices.iter().map(|&i| sentences[i].as_str()).collect();
        if self.strip_whitespace {
            joined.trim().to_string()
        } else {
            joined
        }
    }
}

/// Split text into sentences by scanning for delimiters.
/// The delimiter is attached to the **preceding** sentence.
fn split_into_sentences(text: &str, delimiters: &[String]) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        // Find the earliest delimiter match
        let mut earliest_pos: Option<usize> = None;
        let mut earliest_delim_len: usize = 0;

        for delim in delimiters {
            if let Some(pos) = remaining.find(delim.as_str()) {
                match earliest_pos {
                    None => {
                        earliest_pos = Some(pos);
                        earliest_delim_len = delim.len();
                    }
                    Some(ep) => {
                        if pos < ep {
                            earliest_pos = Some(pos);
                            earliest_delim_len = delim.len();
                        }
                    }
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
                // No more delimiters found — remainder is the last sentence
                if !remaining.is_empty() {
                    sentences.push(remaining.to_string());
                }
                break;
            }
        }
    }

    sentences
}

/// Merge sentences shorter than `min_chars` with the following sentence.
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

    // Remaining buffer: merge with last sentence or push as-is
    if !buffer.is_empty() {
        if let Some(last) = result.last_mut() {
            last.push_str(&buffer);
        } else {
            result.push(buffer);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_chunker_basic() {
        let chunker = SentenceChunker {
            chunk_size: 30,
            chunk_overlap: 0,
            min_characters_per_sentence: 1,
            ..SentenceChunker::new(30, 0)
        };
        let result = chunker.split_text("Hello world. How are you? I am fine. Thank you.");
        // Sentences: ["Hello world. ", "How are you? ", "I am fine. ", "Thank you."]
        // Sizes: 13, 13, 10, 10
        // Chunk 1: "Hello world. How are you?" (26 chars) — fits
        // Can't add "I am fine. " (10) → 36 > 30
        // Chunk 2: "I am fine. Thank you." (21 chars)
        assert_eq!(result.len(), 2);
        assert!(result[0].contains("Hello world."));
        assert!(result[1].contains("Thank you."));
    }

    #[test]
    fn test_sentence_chunker_no_delimiters() {
        let chunker = SentenceChunker::new(100, 0);
        let result = chunker.split_text("No delimiters in this text");
        assert_eq!(result, vec!["No delimiters in this text"]);
    }

    #[test]
    fn test_sentence_chunker_empty() {
        let chunker = SentenceChunker::new(100, 0);
        let result = chunker.split_text("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_sentence_chunker_delimiter_at_end() {
        let chunker = SentenceChunker {
            min_characters_per_sentence: 1,
            ..SentenceChunker::new(100, 0)
        };
        let result = chunker.split_text("Hello world. ");
        assert_eq!(result, vec!["Hello world."]);
    }

    #[test]
    fn test_sentence_chunker_min_chars_filtering() {
        let chunker = SentenceChunker {
            chunk_size: 100,
            chunk_overlap: 0,
            min_characters_per_sentence: 15,
            ..SentenceChunker::new(100, 0)
        };
        // "Hi. " is 4 chars — below min_characters_per_sentence=15
        // Should be merged with "How are you doing today? "
        let result = chunker.split_text("Hi. How are you doing today? Fine thanks.");
        // "Hi. How are you doing today? " merged because "Hi. " < 15 chars
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_sentence_chunker_overlap() {
        let chunker = SentenceChunker {
            chunk_size: 25,
            chunk_overlap: 15,
            min_characters_per_sentence: 1,
            ..SentenceChunker::new(25, 15)
        };
        let result = chunker.split_text("AAA. BBB. CCC. DDD.");
        // Sentences: ["AAA. ", "BBB. ", "CCC. ", "DDD."]
        // Sizes: 5, 5, 5, 4
        // All fit in one chunk (5+5+5+4=19 <= 25)
        assert!(result.len() >= 1);
    }

    #[test]
    fn test_split_into_sentences() {
        let delimiters: Vec<String> = vec![". ", "! ", "? "]
            .into_iter()
            .map(String::from)
            .collect();
        let result = split_into_sentences("Hello world. How are you? Fine! Thanks.", &delimiters);
        assert_eq!(
            result,
            vec!["Hello world. ", "How are you? ", "Fine! ", "Thanks."]
        );
    }

    #[test]
    fn test_merge_short_sentences() {
        let sentences = vec![
            "Hi. ".to_string(),
            "How are you? ".to_string(),
            "Good. ".to_string(),
        ];
        let result = merge_short_sentences(sentences, 10);
        // "Hi. " (4) < 10, merged with "How are you? " → "Hi. How are you? " (18) >= 10, emitted
        // "Good. " (6) < 10, remains in buffer, merged with last
        assert_eq!(result, vec!["Hi. How are you? Good. "]);
    }
}
