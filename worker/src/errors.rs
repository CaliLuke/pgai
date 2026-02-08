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
    /// Classify using an HTTP status code directly.
    pub fn from_status(status: u16, err: anyhow::Error) -> Self {
        match status {
            400 | 401 | 403 | 404 | 422 => Self::Permanent(err),
            429 | 500 | 502 | 503 | 504 => Self::Transient(err),
            s if (400..500).contains(&s) => Self::Permanent(err),
            _ => Self::Transient(err),
        }
    }

    /// Classify from an `async_openai::error::OpenAIError`, extracting structured
    /// information (HTTP status, API error code) before falling back to string matching.
    pub fn from_openai_error(err: async_openai::error::OpenAIError) -> Self {
        match &err {
            async_openai::error::OpenAIError::Reqwest(reqwest_err) => {
                if let Some(status) = reqwest_err.status() {
                    return Self::from_status(status.as_u16(), err.into());
                }
                // No status code — network-level error, likely transient
                Self::Transient(err.into())
            }
            async_openai::error::OpenAIError::ApiError(api_err) => {
                // OpenAI API error codes that indicate permanent failures
                let permanent_codes = [
                    "invalid_api_key",
                    "insufficient_quota",
                    "model_not_found",
                    "invalid_request_error",
                    "billing_hard_limit_reached",
                ];
                if let Some(code) = &api_err.code {
                    let code_lower = code.to_lowercase();
                    if permanent_codes.iter().any(|p| code_lower.contains(p)) {
                        return Self::Permanent(err.into());
                    }
                    if code_lower == "rate_limit_exceeded" {
                        return Self::Transient(err.into());
                    }
                }
                // Fall back to string matching on the message
                Self::classify(err.into())
            }
            _ => Self::classify(err.into()),
        }
    }

    /// Classify an arbitrary error by inspecting its string representation for known patterns.
    /// This is the fallback when structured error info is not available.
    pub fn classify(err: anyhow::Error) -> Self {
        let msg = err.to_string().to_lowercase();

        // Permanent patterns: auth failures, bad requests, invalid config.
        // Use specific prefixes to avoid false positives (e.g. "401 Main St").
        let permanent_patterns = [
            "http 401",
            "http 403",
            "http 400",
            "status 401",
            "status 403",
            "status 400",
            "invalid api key",
            "incorrect api key",
            "model not found",
            "bad request",
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
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "status 429",
            "status 500",
            "status 502",
            "status 503",
            "status 504",
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

    // --- from_status tests ---

    #[test]
    fn test_from_status_401_permanent() {
        let err = anyhow::anyhow!("unauthorized");
        assert!(!EmbeddingError::from_status(401, err).is_transient());
    }

    #[test]
    fn test_from_status_403_permanent() {
        let err = anyhow::anyhow!("forbidden");
        assert!(!EmbeddingError::from_status(403, err).is_transient());
    }

    #[test]
    fn test_from_status_429_transient() {
        let err = anyhow::anyhow!("rate limited");
        assert!(EmbeddingError::from_status(429, err).is_transient());
    }

    #[test]
    fn test_from_status_503_transient() {
        let err = anyhow::anyhow!("service unavailable");
        assert!(EmbeddingError::from_status(503, err).is_transient());
    }

    #[test]
    fn test_from_status_unknown_4xx_permanent() {
        let err = anyhow::anyhow!("teapot");
        assert!(!EmbeddingError::from_status(418, err).is_transient());
    }

    // --- classify string fallback tests ---

    #[test]
    fn test_classify_http_401_as_permanent() {
        let err = anyhow::anyhow!("HTTP 401 Unauthorized");
        assert!(!EmbeddingError::classify(err).is_transient());
    }

    #[test]
    fn test_classify_http_403_as_permanent() {
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
    fn test_classify_http_429_as_transient() {
        let err = anyhow::anyhow!("HTTP 429 Too Many Requests");
        assert!(EmbeddingError::classify(err).is_transient());
    }

    #[test]
    fn test_classify_http_500_as_transient() {
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
    fn test_classify_no_false_positive_on_bare_number() {
        // "401 Main Street" should NOT be classified as permanent
        let err = anyhow::anyhow!("Address: 401 Main Street, Suite 500");
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
