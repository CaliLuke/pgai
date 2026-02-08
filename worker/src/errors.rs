use std::fmt;

/// Classifies embedding errors as transient (retryable) or permanent (fail immediately).
#[derive(Debug)]
pub enum EmbeddingError {
    /// Transient errors that may succeed on retry (network issues, rate limits, server errors).
    Transient(anyhow::Error),
    /// Permanent errors that will never succeed (bad credentials, invalid model, bad input).
    Permanent(anyhow::Error),
}

impl EmbeddingError {
    /// Classify an arbitrary error by inspecting its string representation for known patterns.
    pub fn classify(err: anyhow::Error) -> Self {
        let msg = err.to_string().to_lowercase();

        // Permanent patterns: auth failures, bad requests, invalid config
        let permanent_patterns = [
            "401",
            "403",
            "invalid api key",
            "incorrect api key",
            "model not found",
            "bad request",
            "400",
            "invalid_request_error",
            "billing",
            "quota exceeded",
        ];

        for pattern in &permanent_patterns {
            if msg.contains(pattern) {
                return Self::Permanent(err);
            }
        }

        // Transient patterns: rate limits, server errors, network issues
        let transient_patterns = [
            "429",
            "500",
            "502",
            "503",
            "504",
            "timeout",
            "timed out",
            "connection",
            "rate limit",
            "rate_limit",
            "too many requests",
            "temporarily unavailable",
            "server error",
            "internal server error",
            "broken pipe",
            "reset by peer",
            "dns",
        ];

        for pattern in &transient_patterns {
            if msg.contains(pattern) {
                return Self::Transient(err);
            }
        }

        // Default unknown errors to transient (safer for retries)
        Self::Transient(err)
    }

    pub fn is_transient(&self) -> bool {
        matches!(self, Self::Transient(_))
    }

    /// Consume the error and return the inner anyhow::Error.
    pub fn into_inner(self) -> anyhow::Error {
        match self {
            Self::Transient(e) | Self::Permanent(e) => e,
        }
    }
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transient(e) => write!(f, "transient embedding error: {}", e),
            Self::Permanent(e) => write!(f, "permanent embedding error: {}", e),
        }
    }
}

impl std::error::Error for EmbeddingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transient(e) | Self::Permanent(e) => Some(e.as_ref()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_401_as_permanent() {
        let err = anyhow::anyhow!("HTTP 401 Unauthorized");
        assert!(!EmbeddingError::classify(err).is_transient());
    }

    #[test]
    fn test_classify_403_as_permanent() {
        let err = anyhow::anyhow!("HTTP 403 Forbidden");
        assert!(!EmbeddingError::classify(err).is_transient());
    }

    #[test]
    fn test_classify_invalid_api_key_as_permanent() {
        let err = anyhow::anyhow!("Error: Invalid API key provided");
        assert!(!EmbeddingError::classify(err).is_transient());
    }

    #[test]
    fn test_classify_model_not_found_as_permanent() {
        let err = anyhow::anyhow!("The model `foo` does not exist: model not found");
        assert!(!EmbeddingError::classify(err).is_transient());
    }

    #[test]
    fn test_classify_429_as_transient() {
        let err = anyhow::anyhow!("HTTP 429 Too Many Requests");
        assert!(EmbeddingError::classify(err).is_transient());
    }

    #[test]
    fn test_classify_500_as_transient() {
        let err = anyhow::anyhow!("HTTP 500 Internal Server Error");
        assert!(EmbeddingError::classify(err).is_transient());
    }

    #[test]
    fn test_classify_timeout_as_transient() {
        let err = anyhow::anyhow!("request timeout after 30s");
        assert!(EmbeddingError::classify(err).is_transient());
    }

    #[test]
    fn test_classify_connection_as_transient() {
        let err = anyhow::anyhow!("connection refused");
        assert!(EmbeddingError::classify(err).is_transient());
    }

    #[test]
    fn test_classify_unknown_as_transient() {
        let err = anyhow::anyhow!("some mysterious error");
        assert!(EmbeddingError::classify(err).is_transient());
    }

    #[test]
    fn test_into_inner() {
        let err = EmbeddingError::Transient(anyhow::anyhow!("test"));
        let inner = err.into_inner();
        assert_eq!(inner.to_string(), "test");
    }

    #[test]
    fn test_display() {
        let err = EmbeddingError::Transient(anyhow::anyhow!("network error"));
        assert!(err.to_string().contains("transient"));
        assert!(err.to_string().contains("network error"));

        let err = EmbeddingError::Permanent(anyhow::anyhow!("bad key"));
        assert!(err.to_string().contains("permanent"));
        assert!(err.to_string().contains("bad key"));
    }
}
