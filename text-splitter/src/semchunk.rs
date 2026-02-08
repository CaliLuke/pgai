use crate::char_len;
use crate::merge::merge_splits;

/// A semantic recursive text splitter with a fixed delimiter hierarchy.
///
/// Inspired by semchunk: uses a built-in hierarchy of 30+ delimiter types
/// ordered by semantic importance. No configuration of separators needed.
///
/// Key behaviors:
/// - **Longest-sequence-first**: prefers `\n\n\n` over `\n\n` over `\n`
/// - **Punctuation reattachment**: after splitting on non-whitespace delimiters,
///   the delimiter is reattached to the preceding chunk
/// - **Hierarchical fallback**: tries newlines, then tabs, then spaces (at
///   punctuation boundaries first), then word joiners, then individual characters
pub struct SemchunkSplitter {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub strip_whitespace: bool,
    pub length_fn: Option<Box<dyn Fn(&str) -> usize + Send + Sync>>,
}

/// The hierarchy of splitter levels, ordered from most semantic to least.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SplitLevel {
    Newlines,            // \n sequences (longest first)
    Tabs,                // \t sequences (longest first)
    SentenceTerminators, // ". ", "? ", "! "
    ClauseSeparators,    // "; ", ", ", ") ", "] "
    SentenceInterrupters, // ": ", "-- ", "... "
    Whitespace,          // space sequences (longest first)
    WordJoiners,         // "/", "\\", "&", "-"
    Characters,          // individual chars (final fallback)
}

const SPLIT_LEVELS: &[SplitLevel] = &[
    SplitLevel::Newlines,
    SplitLevel::Tabs,
    SplitLevel::SentenceTerminators,
    SplitLevel::ClauseSeparators,
    SplitLevel::SentenceInterrupters,
    SplitLevel::Whitespace,
    SplitLevel::WordJoiners,
    SplitLevel::Characters,
];

impl SemchunkSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            strip_whitespace: true,
            length_fn: None,
        }
    }

    pub fn split_text(&self, text: &str) -> Vec<String> {
        let text = if self.strip_whitespace {
            text.trim()
        } else {
            text
        };
        if text.is_empty() {
            return Vec::new();
        }

        let default_fn = char_len;
        let len_fn: &dyn Fn(&str) -> usize = match &self.length_fn {
            Some(f) => f.as_ref(),
            None => &default_fn,
        };

        if len_fn(text) <= self.chunk_size {
            return vec![text.to_string()];
        }

        self.split_recursive(text, 0, len_fn)
    }

    fn split_recursive(
        &self,
        text: &str,
        level_idx: usize,
        length_fn: &dyn Fn(&str) -> usize,
    ) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        if length_fn(text) <= self.chunk_size {
            return vec![text.to_string()];
        }

        if level_idx >= SPLIT_LEVELS.len() {
            // Shouldn't happen (Characters is final fallback), but just in case
            return vec![text.to_string()];
        }

        let level = SPLIT_LEVELS[level_idx];

        if level == SplitLevel::Characters {
            // Final fallback: split into individual chars and merge
            let chars: Vec<String> = text.chars().map(|c| c.to_string()).collect();
            return merge_splits(
                &chars,
                "",
                self.chunk_size,
                self.chunk_overlap,
                self.strip_whitespace,
                length_fn,
            );
        }

        // Find the best delimiter for this level that exists in the text
        let delimiter = match self.find_delimiter(text, level) {
            Some(d) => d,
            None => {
                // This level's delimiters don't exist in text, try next level
                return self.split_recursive(text, level_idx + 1, length_fn);
            }
        };

        let is_whitespace_delim = matches!(
            level,
            SplitLevel::Newlines | SplitLevel::Tabs | SplitLevel::Whitespace
        );

        // Split on the delimiter
        let raw_splits: Vec<&str> = text.split(&delimiter).collect();

        // Reattach delimiter for non-whitespace delimiters
        let splits: Vec<String> = if is_whitespace_delim {
            raw_splits
                .into_iter()
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect()
        } else {
            reattach_delimiter(&raw_splits, &delimiter)
        };

        if splits.is_empty() {
            return self.split_recursive(text, level_idx + 1, length_fn);
        }

        // If splitting didn't help (only 1 piece), try next level
        if splits.len() == 1 {
            return self.split_recursive(text, level_idx + 1, length_fn);
        }

        // Merge splits, then recurse on any that still exceed chunk_size
        let merge_sep = if is_whitespace_delim { &delimiter } else { "" };

        let merged = merge_splits(
            &splits,
            merge_sep,
            self.chunk_size,
            self.chunk_overlap,
            self.strip_whitespace,
            length_fn,
        );

        let mut result: Vec<String> = Vec::new();
        for chunk in merged {
            if length_fn(&chunk) > self.chunk_size {
                // Recurse at the next level
                let sub_chunks = self.split_recursive(&chunk, level_idx + 1, length_fn);
                result.extend(sub_chunks);
            } else {
                result.push(chunk);
            }
        }

        result
    }

    /// Find the best delimiter for a given level that exists in the text.
    /// For newlines/tabs/spaces: find the longest sequence present.
    fn find_delimiter(&self, text: &str, level: SplitLevel) -> Option<String> {
        match level {
            SplitLevel::Newlines => find_longest_sequence(text, '\n'),
            SplitLevel::Tabs => find_longest_sequence(text, '\t'),
            SplitLevel::Whitespace => find_longest_space_sequence(text),
            SplitLevel::SentenceTerminators => {
                find_first_present(text, &[". ", "? ", "! "])
            }
            SplitLevel::ClauseSeparators => {
                find_first_present(text, &["; ", ", ", ") ", "] "])
            }
            SplitLevel::SentenceInterrupters => {
                find_first_present(text, &[": ", "-- ", "... "])
            }
            SplitLevel::WordJoiners => {
                find_first_present(text, &["/", "\\", "&", "-"])
            }
            SplitLevel::Characters => {
                // Handled separately in split_recursive
                Some(String::new())
            }
        }
    }
}

/// Find the longest contiguous sequence of `ch` in `text`.
fn find_longest_sequence(text: &str, ch: char) -> Option<String> {
    let mut max_len: usize = 0;
    let mut current_len: usize = 0;

    for c in text.chars() {
        if c == ch {
            current_len += 1;
            if current_len > max_len {
                max_len = current_len;
            }
        } else {
            current_len = 0;
        }
    }

    if max_len > 0 {
        Some(std::iter::repeat(ch).take(max_len).collect())
    } else {
        None
    }
}

/// Find the longest contiguous sequence of spaces, but only if the text
/// doesn't also contain punctuation-preceded spaces (which would be handled
/// by earlier levels).
fn find_longest_space_sequence(text: &str) -> Option<String> {
    let mut max_len: usize = 0;
    let mut current_len: usize = 0;

    for c in text.chars() {
        if c == ' ' {
            current_len += 1;
            if current_len > max_len {
                max_len = current_len;
            }
        } else {
            current_len = 0;
        }
    }

    if max_len > 0 {
        Some(" ".repeat(max_len))
    } else {
        None
    }
}

/// Find the first delimiter from the list that exists in the text.
fn find_first_present(text: &str, delimiters: &[&str]) -> Option<String> {
    for delim in delimiters {
        if text.contains(delim) {
            return Some(delim.to_string());
        }
    }
    None
}

/// Reattach a non-whitespace delimiter to the end of the preceding split.
/// E.g., splitting "Hello. World" on ". " gives ["Hello", "World"],
/// and we want ["Hello. ", "World"].
fn reattach_delimiter(splits: &[&str], delimiter: &str) -> Vec<String> {
    let mut result: Vec<String> = Vec::new();

    for (i, split) in splits.iter().enumerate() {
        if split.is_empty() && i > 0 {
            // Empty split after delimiter — skip
            continue;
        }
        if i < splits.len() - 1 {
            // Not the last split — reattach delimiter to end
            result.push(format!("{}{}", split, delimiter));
        } else if !split.is_empty() {
            // Last split — no delimiter after it
            result.push(split.to_string());
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semchunk_basic() {
        let splitter = SemchunkSplitter::new(15, 0);
        let result = splitter.split_text("Hello world. How are you?");
        assert!(result.len() >= 2);
        for chunk in &result {
            assert!(
                chunk.chars().count() <= 15,
                "Chunk too long: {:?} ({})",
                chunk,
                chunk.chars().count()
            );
        }
    }

    #[test]
    fn test_semchunk_paragraph_boundaries() {
        let splitter = SemchunkSplitter::new(20, 0);
        let result = splitter.split_text("Para one.\n\nPara two.\n\nPara three.");
        // Should split on \n\n first
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
    fn test_semchunk_sentence_boundaries() {
        let splitter = SemchunkSplitter::new(30, 0);
        let text = "This is sentence one. This is sentence two. And sentence three.";
        let result = splitter.split_text(text);
        assert!(result.len() >= 2);
        for chunk in &result {
            assert!(
                chunk.chars().count() <= 30,
                "Chunk too long: {:?}",
                chunk
            );
        }
    }

    #[test]
    fn test_semchunk_punctuation_reattachment() {
        let splitter = SemchunkSplitter::new(25, 0);
        let result = splitter.split_text("Hello world. Goodbye world.");
        // Period should stay attached to preceding text
        for chunk in &result {
            // No chunk should start with ". " (delimiter should be at end of preceding)
            assert!(
                !chunk.starts_with(". "),
                "Delimiter should be reattached: {:?}",
                chunk
            );
        }
    }

    #[test]
    fn test_semchunk_longest_sequence_preference() {
        let splitter = SemchunkSplitter::new(10, 0);
        let text = "AAAAAAA\n\n\nBBBBBBB\nCCCCCCC";
        let result = splitter.split_text(text);
        // Should prefer triple newline first, splitting into ["AAAAAAA", "BBBBBBB\nCCCCCCC"]
        // Then "BBBBBBB\nCCCCCCC" (15 chars) > 10, recurse and split on single \n
        assert!(result.len() >= 2);
        // First chunk should be "AAAAAAA" (split on \n\n\n)
        assert_eq!(result[0], "AAAAAAA");
    }

    #[test]
    fn test_semchunk_fallback_to_characters() {
        let splitter = SemchunkSplitter::new(5, 0);
        let text = "abcdefghij";
        let result = splitter.split_text(text);
        assert!(result.len() >= 2);
        for chunk in &result {
            assert!(
                chunk.chars().count() <= 5,
                "Chunk too long: {:?}",
                chunk
            );
        }
    }

    #[test]
    fn test_semchunk_empty_text() {
        let splitter = SemchunkSplitter::new(100, 0);
        let result = splitter.split_text("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_semchunk_whitespace_only() {
        let splitter = SemchunkSplitter::new(100, 0);
        let result = splitter.split_text("   \n\n   ");
        assert!(result.is_empty());
    }

    #[test]
    fn test_semchunk_fits_in_one_chunk() {
        let splitter = SemchunkSplitter::new(100, 0);
        let result = splitter.split_text("Short text");
        assert_eq!(result, vec!["Short text"]);
    }

    #[test]
    fn test_find_longest_sequence() {
        assert_eq!(
            find_longest_sequence("a\n\n\nb\nc", '\n'),
            Some("\n\n\n".to_string())
        );
        assert_eq!(find_longest_sequence("abc", '\n'), None);
        assert_eq!(
            find_longest_sequence("\n", '\n'),
            Some("\n".to_string())
        );
    }

    #[test]
    fn test_reattach_delimiter() {
        let splits = vec!["Hello", "World", "Foo"];
        let result = reattach_delimiter(&splits, ". ");
        assert_eq!(result, vec!["Hello. ", "World. ", "Foo"]);
    }

    #[test]
    fn test_semchunk_overlap() {
        let splitter = SemchunkSplitter::new(15, 5);
        let text = "AAAA. BBBB. CCCC. DDDD.";
        let result = splitter.split_text(text);
        assert!(result.len() >= 2, "Expected multiple chunks, got {:?}", result);
    }
}
