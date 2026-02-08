/// Merge small splits into chunks approaching `chunk_size`, with overlap.
///
/// This mirrors LangChain's `_merge_splits` method from `TextSplitter`.
///
/// - `splits`: the pieces of text to merge
/// - `separator`: the string to join splits with
/// - `chunk_size`: maximum chunk length (measured by `length_fn`)
/// - `chunk_overlap`: how much overlap between consecutive chunks (measured by `length_fn`)
/// - `strip_whitespace`: if true, trim whitespace from chunks and drop empty results
/// - `length_fn`: measures the length of a string (e.g. character count, token count)
pub fn merge_splits(
    splits: &[String],
    separator: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    strip_whitespace: bool,
    length_fn: &dyn Fn(&str) -> usize,
) -> Vec<String> {
    let separator_len = length_fn(separator);
    let mut docs: Vec<String> = Vec::new();
    let mut current_doc: Vec<&str> = Vec::new();
    let mut total: usize = 0;

    for d in splits {
        let len = length_fn(d);
        let sep_cost = if current_doc.is_empty() {
            0
        } else {
            separator_len
        };

        if total + len + sep_cost > chunk_size {
            if !current_doc.is_empty() {
                let doc = join_docs(&current_doc, separator, strip_whitespace);
                if let Some(doc) = doc {
                    docs.push(doc);
                }
                // Pop from front until under chunk_overlap
                while total > chunk_overlap
                    || (total + len + if current_doc.is_empty() { 0 } else { separator_len }
                        > chunk_size
                        && total > 0)
                {
                    let removed_len = length_fn(current_doc[0]);
                    let sep = if current_doc.len() > 1 {
                        separator_len
                    } else {
                        0
                    };
                    total = total.saturating_sub(removed_len + sep);
                    current_doc.remove(0);
                }
            }
        }

        current_doc.push(d);
        total += len + if current_doc.len() > 1 { separator_len } else { 0 };
    }

    if let Some(doc) = join_docs(&current_doc, separator, strip_whitespace) {
        docs.push(doc);
    }

    docs
}

fn join_docs(docs: &[&str], separator: &str, strip_whitespace: bool) -> Option<String> {
    let text = docs.join(separator);
    let text = if strip_whitespace {
        text.trim().to_string()
    } else {
        text
    };
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::char_len;

    #[test]
    fn test_merge_basic() {
        let splits: Vec<String> = vec!["a", "b", "c"]
            .into_iter()
            .map(String::from)
            .collect();
        let result = merge_splits(&splits, " ", 5, 0, true, &char_len);
        assert_eq!(result, vec!["a b c"]);
    }

    #[test]
    fn test_merge_exceeds_chunk_size() {
        let splits: Vec<String> = vec!["foo", "bar", "baz", "123"]
            .into_iter()
            .map(String::from)
            .collect();
        let result = merge_splits(&splits, " ", 7, 3, true, &char_len);
        assert_eq!(result, vec!["foo bar", "bar baz", "baz 123"]);
    }

    #[test]
    fn test_merge_no_overlap() {
        let splits: Vec<String> = vec!["aa", "bb", "cc", "dd"]
            .into_iter()
            .map(String::from)
            .collect();
        let result = merge_splits(&splits, " ", 5, 0, true, &char_len);
        assert_eq!(result, vec!["aa bb", "cc dd"]);
    }

    #[test]
    fn test_merge_empty_splits() {
        let splits: Vec<String> = vec![];
        let result = merge_splits(&splits, " ", 10, 0, true, &char_len);
        assert!(result.is_empty());
    }

    #[test]
    fn test_merge_strip_whitespace() {
        let splits: Vec<String> = vec![" a ", " b "]
            .into_iter()
            .map(String::from)
            .collect();
        let result = merge_splits(&splits, " ", 100, 0, true, &char_len);
        // " a " + " " + " b " = " a   b " → trimmed = "a   b"
        assert_eq!(result, vec!["a   b"]);
    }

    #[test]
    fn test_merge_custom_length_fn() {
        // Use word count instead of char count
        let word_len = |s: &str| -> usize { s.split_whitespace().count().max(1) };
        let splits: Vec<String> = vec!["hello world", "foo bar", "baz"]
            .into_iter()
            .map(String::from)
            .collect();
        // Each split is 2, 2, 1 words. Separator " " is 1 word.
        // chunk_size=3 words: "hello world" (2) + " " (1) = can't fit "foo bar" (2+1=5 > 3)
        let result = merge_splits(&splits, " ", 3, 0, true, &word_len);
        assert_eq!(result, vec!["hello world", "foo bar", "baz"]);
    }
}
