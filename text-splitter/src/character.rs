use crate::char_len;
use crate::merge::merge_splits;
use crate::split::split_text_with_regex;

/// A text splitter that splits on a single separator, then merges the pieces
/// into chunks respecting `chunk_size` and `chunk_overlap`.
///
/// This is a Rust port of LangChain's `CharacterTextSplitter`.
pub struct CharacterTextSplitter {
    pub separator: String,
    pub is_separator_regex: bool,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub strip_whitespace: bool,
    pub length_fn: Option<Box<dyn Fn(&str) -> usize + Send + Sync>>,
}

impl CharacterTextSplitter {
    pub fn new(separator: &str, chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            separator: separator.to_string(),
            is_separator_regex: false,
            chunk_size,
            chunk_overlap,
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

        let sep_pattern = if self.is_separator_regex {
            self.separator.clone()
        } else {
            regex::escape(&self.separator)
        };

        // Split without keeping separator (LangChain default for CharacterTextSplitter)
        let splits = split_text_with_regex(text, &sep_pattern, None);

        // Merge using the literal separator (re-inserted between pieces)
        let merge_sep = &self.separator;

        merge_splits(
            &splits,
            merge_sep,
            self.chunk_size,
            self.chunk_overlap,
            self.strip_whitespace,
            len_fn,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_character_splitter_basic() {
        // From LangChain test suite
        let splitter = CharacterTextSplitter {
            separator: " ".to_string(),
            is_separator_regex: false,
            chunk_size: 7,
            chunk_overlap: 3,
            strip_whitespace: true,
            length_fn: None,
        };
        let result = splitter.split_text("foo bar baz 123");
        assert_eq!(result, vec!["foo bar", "bar baz", "baz 123"]);
    }

    #[test]
    fn test_character_splitter_empty_doc_filtering() {
        // "foo  bar" split on " " gives ["foo", "bar"] (empties filtered)
        // Then merged with " " separator → "foo bar" (fits in chunk_size=9)
        let splitter = CharacterTextSplitter {
            separator: " ".to_string(),
            is_separator_regex: false,
            chunk_size: 9,
            chunk_overlap: 0,
            strip_whitespace: true,
            length_fn: None,
        };
        let result = splitter.split_text("foo  bar");
        assert_eq!(result, vec!["foo bar"]);
    }

    #[test]
    fn test_character_splitter_small_chunks() {
        let splitter = CharacterTextSplitter {
            separator: " ".to_string(),
            is_separator_regex: false,
            chunk_size: 3,
            chunk_overlap: 1,
            strip_whitespace: true,
            length_fn: None,
        };
        let result = splitter.split_text("foo bar baz a a");
        assert_eq!(result, vec!["foo", "bar", "baz", "a a"]);
    }

    #[test]
    fn test_character_splitter_empty_input() {
        let splitter = CharacterTextSplitter::new("\n", 100, 0);
        let result = splitter.split_text("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_character_splitter_whitespace_only() {
        let splitter = CharacterTextSplitter::new(" ", 100, 0);
        let result = splitter.split_text("   ");
        assert!(result.is_empty());
    }

    #[test]
    fn test_character_splitter_no_separator_match() {
        let splitter = CharacterTextSplitter::new("X", 100, 0);
        let result = splitter.split_text("hello world");
        assert_eq!(result, vec!["hello world"]);
    }

    #[test]
    fn test_character_splitter_single_word() {
        let splitter = CharacterTextSplitter::new(" ", 100, 0);
        let result = splitter.split_text("hello");
        assert_eq!(result, vec!["hello"]);
    }

    #[test]
    fn test_character_splitter_is_separator_regex() {
        let splitter = CharacterTextSplitter {
            separator: r"\s+".to_string(),
            is_separator_regex: true,
            chunk_size: 7,
            chunk_overlap: 0,
            strip_whitespace: true,
            length_fn: None,
        };
        let result = splitter.split_text("foo  bar\tbaz");
        // Splits: ["foo", "bar", "baz"]
        // Merged with separator="\s+" (literal string used as merge separator)
        // Each piece is ≤7 chars, so they get merged:
        // "foo" + "\s+" + "bar" = "foo\s+bar" (9 chars) > 7, so first chunk is "foo"
        // then "bar" + "\s+" + "baz" = "bar\s+baz" (9 chars) > 7, so "bar" alone
        // then "baz"
        assert_eq!(result, vec!["foo", "bar", "baz"]);
    }

    #[test]
    fn test_character_splitter_regex_special_chars() {
        // When is_separator_regex=false, special chars should be escaped
        let splitter = CharacterTextSplitter {
            separator: ".".to_string(),
            is_separator_regex: false,
            chunk_size: 10,
            chunk_overlap: 0,
            strip_whitespace: true,
            length_fn: None,
        };
        let result = splitter.split_text("hello.world.test");
        // Split on literal ".": ["hello", "world", "test"]
        // Merged with ".": "hello.world" (11 chars) > 10, so "hello" first
        // then "world.test" (10 chars) ≤ 10
        assert_eq!(result, vec!["hello", "world.test"]);
    }

    #[test]
    fn test_character_splitter_newline() {
        let splitter = CharacterTextSplitter::new("\n\n", 20, 0);
        let result = splitter.split_text("Hello World\n\nFoo Bar\n\nBaz");
        // Split on "\n\n": ["Hello World", "Foo Bar", "Baz"]
        // Merged with "\n\n":
        // "Hello World" (11) + "\n\n" (2) + "Foo Bar" (7) = 20 ≤ 20
        assert_eq!(result, vec!["Hello World\n\nFoo Bar", "Baz"]);
    }

    #[test]
    fn test_character_splitter_multichar_sep() {
        let splitter = CharacterTextSplitter::new("<SEP>", 20, 0);
        let result = splitter.split_text("part one<SEP>part two<SEP>part three");
        assert!(result.len() >= 2);
        for chunk in &result {
            assert!(chunk.chars().count() <= 20, "Chunk too long: {:?}", chunk);
        }
    }
}
