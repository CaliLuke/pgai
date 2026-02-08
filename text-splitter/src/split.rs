use regex::Regex;

/// Where to attach the separator when keeping it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepSeparator {
    Start,
    End,
}

/// Split text using a regex pattern.
///
/// - If `separator_pattern` is empty, splits into individual characters.
/// - `keep_separator` controls whether the separator is attached to the start
///   or end of each split, or discarded (`None`).
///
/// Empty strings are filtered out of the result.
pub fn split_text_with_regex(
    text: &str,
    separator_pattern: &str,
    keep_separator: Option<KeepSeparator>,
) -> Vec<String> {
    if separator_pattern.is_empty() {
        return text.chars().map(|c| c.to_string()).collect();
    }

    let re = Regex::new(separator_pattern).expect("invalid regex pattern");

    match keep_separator {
        None => re
            .split(text)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect(),

        Some(position) => {
            // Split with capturing group to keep delimiters
            let capturing = format!("({})", separator_pattern);
            let cap_re = Regex::new(&capturing).expect("invalid regex pattern");
            let parts: Vec<&str> = cap_re.split(text).collect();

            // cap_re.split interleaves text and separators:
            // [text0, sep0, text1, sep1, text2, ...]
            // We need to find actual separator matches to pair them correctly.
            let separators: Vec<&str> = cap_re
                .find_iter(text)
                .map(|m| m.as_str())
                .collect();

            // Rebuild: plain splits are at even indices, separators at odd indices
            // parts[0], separators[0], parts[1], separators[1], parts[2], ...
            let mut result = Vec::new();

            match position {
                KeepSeparator::End => {
                    // Attach separator to the preceding split
                    for i in 0..separators.len() {
                        let combined = format!("{}{}", parts[i], separators[i]);
                        if !combined.is_empty() {
                            result.push(combined);
                        }
                    }
                    // Last part has no separator after it
                    let last = parts[separators.len()..].join("");
                    if !last.is_empty() {
                        result.push(last);
                    }
                }
                KeepSeparator::Start => {
                    // First part has no separator before it
                    if !parts[0].is_empty() {
                        result.push(parts[0].to_string());
                    }
                    // Attach separator to the following split
                    for i in 0..separators.len() {
                        let text_part = if i + 1 < parts.len() {
                            parts[i + 1]
                        } else {
                            ""
                        };
                        let combined = format!("{}{}", separators[i], text_part);
                        if !combined.is_empty() {
                            result.push(combined);
                        }
                    }
                }
            }

            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_no_keep() {
        let result = split_text_with_regex("hello world foo", " ", None);
        assert_eq!(result, vec!["hello", "world", "foo"]);
    }

    #[test]
    fn test_split_keep_start() {
        let result = split_text_with_regex("hello world foo", " ", Some(KeepSeparator::Start));
        assert_eq!(result, vec!["hello", " world", " foo"]);
    }

    #[test]
    fn test_split_keep_end() {
        let result = split_text_with_regex("hello world foo", " ", Some(KeepSeparator::End));
        assert_eq!(result, vec!["hello ", "world ", "foo"]);
    }

    #[test]
    fn test_split_empty_separator() {
        let result = split_text_with_regex("abc", "", None);
        assert_eq!(result, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_split_no_match() {
        let result = split_text_with_regex("hello", "X", None);
        assert_eq!(result, vec!["hello"]);
    }

    #[test]
    fn test_split_filters_empty() {
        let result = split_text_with_regex("a  b", " ", None);
        assert_eq!(result, vec!["a", "b"]);
    }

    #[test]
    fn test_split_regex_pattern() {
        let result = split_text_with_regex("foo123bar456baz", r"\d+", None);
        assert_eq!(result, vec!["foo", "bar", "baz"]);
    }
}
