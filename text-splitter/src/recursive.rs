use regex::Regex;

use crate::char_len;
use crate::merge::merge_splits;
use crate::split::{split_text_with_regex, KeepSeparator};

/// Default separators for recursive character text splitting.
pub const DEFAULT_SEPARATORS: &[&str] = &["\n\n", "\n", " ", ""];

/// A text splitter that recursively tries different separators to split text
/// into chunks of at most `chunk_size` characters.
///
/// This is a Rust port of LangChain's `RecursiveCharacterTextSplitter`.
pub struct RecursiveCharacterTextSplitter {
    pub separators: Vec<String>,
    pub is_separator_regex: bool,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub keep_separator: bool,
    pub strip_whitespace: bool,
    pub length_fn: Option<Box<dyn Fn(&str) -> usize + Send + Sync>>,
}

impl RecursiveCharacterTextSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            separators: DEFAULT_SEPARATORS.iter().map(|s| s.to_string()).collect(),
            is_separator_regex: false,
            chunk_size,
            chunk_overlap,
            keep_separator: true,
            strip_whitespace: true,
            length_fn: None,
        }
    }

    pub fn split_text(&self, text: &str) -> Vec<String> {
        let default_fn = char_len;
        let len_fn: &dyn Fn(&str) -> usize = match &self.length_fn {
            Some(f) => f.as_ref(),
            None => &default_fn,
        };
        self.split_text_recursive(text, &self.separators, len_fn)
    }

    fn split_text_recursive(
        &self,
        text: &str,
        separators: &[String],
        length_fn: &dyn Fn(&str) -> usize,
    ) -> Vec<String> {
        let mut final_chunks: Vec<String> = Vec::new();

        // Find the appropriate separator
        let last_sep = separators.last().map(|s| s.as_str()).unwrap_or("");
        let mut separator = last_sep;
        let mut new_separators: &[String] = &[];

        for (i, s) in separators.iter().enumerate() {
            let sep_pattern = if self.is_separator_regex {
                s.clone()
            } else {
                regex::escape(s)
            };

            if s.is_empty() {
                separator = s;
                break;
            }

            if let Ok(re) = Regex::new(&sep_pattern) {
                if re.is_match(text) {
                    separator = s;
                    new_separators = &separators[i + 1..];
                    break;
                }
            }
        }

        let sep_pattern = if self.is_separator_regex {
            separator.to_string()
        } else {
            regex::escape(separator)
        };

        let keep = if self.keep_separator {
            Some(KeepSeparator::Start)
        } else {
            None
        };

        let splits = split_text_with_regex(text, &sep_pattern, keep);

        // Merge separator for recombining good_splits:
        // If keep_separator is true, separator is already part of the splits,
        // so we merge with empty string.
        let merge_sep = if self.keep_separator {
            ""
        } else {
            separator
        };

        let mut good_splits: Vec<String> = Vec::new();

        for s in &splits {
            if length_fn(s) < self.chunk_size {
                good_splits.push(s.clone());
            } else {
                if !good_splits.is_empty() {
                    let merged = merge_splits(
                        &good_splits,
                        merge_sep,
                        self.chunk_size,
                        self.chunk_overlap,
                        self.strip_whitespace,
                        length_fn,
                    );
                    final_chunks.extend(merged);
                    good_splits.clear();
                }
                if new_separators.is_empty() {
                    final_chunks.push(s.clone());
                } else {
                    let other = self.split_text_recursive(s, new_separators, length_fn);
                    final_chunks.extend(other);
                }
            }
        }

        if !good_splits.is_empty() {
            let merged = merge_splits(
                &good_splits,
                merge_sep,
                self.chunk_size,
                self.chunk_overlap,
                self.strip_whitespace,
                length_fn,
            );
            final_chunks.extend(merged);
        }

        final_chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recursive_basic() {
        let splitter = RecursiveCharacterTextSplitter {
            separators: DEFAULT_SEPARATORS.iter().map(|s| s.to_string()).collect(),
            is_separator_regex: false,
            chunk_size: 10,
            chunk_overlap: 1,
            keep_separator: true,
            strip_whitespace: true,
            length_fn: None,
        };
        let text =
            "Hi.\n\nI'm Harrison.\n\nHow? Are? You?\nOkay then f f f f.\nThis is a weird text to write, but gotta test the splittting am I right?";
        let result = splitter.split_text(text);

        // All chunks should respect chunk_size
        for chunk in &result {
            assert!(
                chunk.chars().count() <= 10,
                "Chunk exceeded chunk_size: {:?} ({})",
                chunk,
                chunk.chars().count()
            );
        }
        assert!(result.len() > 1, "Expected multiple chunks");
    }

    #[test]
    fn test_recursive_custom_separators() {
        let splitter = RecursiveCharacterTextSplitter {
            separators: vec!["X".to_string(), "Y".to_string()],
            is_separator_regex: false,
            chunk_size: 5,
            chunk_overlap: 0,
            keep_separator: false,
            strip_whitespace: true,
            length_fn: None,
        };
        let result = splitter.split_text("aaXbbYcc");
        // Should split on X first: ["aa", "bbYcc"]
        // "bbYcc" > 5? no (5 chars), so it stays as one chunk
        assert!(result.contains(&"aa".to_string()));
    }

    #[test]
    fn test_recursive_default_separators() {
        let splitter = RecursiveCharacterTextSplitter::new(20, 0);
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let result = splitter.split_text(text);

        assert!(result.len() >= 2);
        for chunk in &result {
            assert!(
                chunk.chars().count() <= 20,
                "Chunk too long: {:?}",
                chunk
            );
        }
    }

    #[test]
    fn test_recursive_empty_text() {
        let splitter = RecursiveCharacterTextSplitter::new(10, 0);
        let result = splitter.split_text("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_recursive_single_chunk_fits() {
        let splitter = RecursiveCharacterTextSplitter::new(100, 0);
        let result = splitter.split_text("Short text");
        assert_eq!(result, vec!["Short text"]);
    }

    #[test]
    fn test_recursive_with_overlap() {
        let splitter = RecursiveCharacterTextSplitter::new(10, 3);
        let text = "aaaa\n\nbbbb\n\ncccc\n\ndddd";
        let result = splitter.split_text(text);

        assert!(result.len() >= 2);
        for chunk in &result {
            assert!(
                chunk.chars().count() <= 10,
                "Chunk too long: {:?}",
                chunk
            );
        }
    }

    #[test]
    fn test_recursive_keep_separator_false() {
        let splitter = RecursiveCharacterTextSplitter {
            separators: DEFAULT_SEPARATORS.iter().map(|s| s.to_string()).collect(),
            is_separator_regex: false,
            chunk_size: 10,
            chunk_overlap: 0,
            keep_separator: false,
            strip_whitespace: true,
            length_fn: None,
        };
        let text = "Hello\n\nWorld\n\nFoo";
        let result = splitter.split_text(text);
        for chunk in &result {
            assert!(!chunk.starts_with("\n\n"), "Separator should not be at start");
        }
    }

    #[test]
    fn test_recursive_is_separator_regex() {
        let splitter = RecursiveCharacterTextSplitter {
            separators: vec![r"\d+".to_string(), "".to_string()],
            is_separator_regex: true,
            chunk_size: 10,
            chunk_overlap: 0,
            keep_separator: false,
            strip_whitespace: true,
            length_fn: None,
        };
        let result = splitter.split_text("abc123def456ghi");
        // With keep_separator=false, separator is stripped:
        // splits on \d+: ["abc", "def", "ghi"]
        // all fit in chunk_size=10, merged with separator "\d+" (literal string)
        // "abc" + "\d+" + "def" = 9 chars ≤ 10
        // + "\d+" + "ghi" would be 15 > 10
        // So result should be two chunks
        assert_eq!(result.len(), 2);
        assert!(result[0].contains("abc"));
        assert!(result[1].contains("ghi"));
    }

    #[test]
    fn test_recursive_respects_chunk_size_on_long_text() {
        let text = "a ".repeat(200);
        let splitter = RecursiveCharacterTextSplitter::new(50, 5);
        let result = splitter.split_text(&text);

        assert!(result.len() > 1);
        for chunk in &result {
            assert!(
                chunk.chars().count() <= 50,
                "Chunk exceeded 50 chars: {} chars",
                chunk.chars().count()
            );
        }
    }
}
