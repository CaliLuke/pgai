use anyhow::Result;
use clap::Parser;
use opentelemetry::KeyValue;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::{WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::SdkTracerProvider;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Database URL
    #[arg(short, long, env = "DB_URL")]
    db_url: String,

    /// Poll interval in seconds
    #[arg(short, long, default_value_t = 60)]
    poll_interval: u64,

    /// Run once and exit
    #[arg(long, default_value_t = false)]
    once: bool,

    /// Exit on error
    #[arg(long, default_value_t = false)]
    exit_on_error: bool,

    /// Specific vectorizer IDs to run (comma-separated)
    #[arg(long, value_delimiter = ',')]
    vectorizer_ids: Vec<i32>,

    /// Log output format: "text" or "json"
    #[arg(long, default_value = "text")]
    log_format: String,
}

/// Redact the password in a database connection URL.
fn redact_db_url(url: &str) -> String {
    match url::Url::parse(url) {
        Ok(mut parsed) => {
            if parsed.password().is_some() {
                let _ = parsed.set_password(Some("***"));
            }
            parsed.to_string()
        }
        Err(_) => "***".to_string(),
    }
}

/// Prefer signal-specific endpoint when provided; fall back to generic endpoint.
fn otel_endpoint_from_env() -> Option<String> {
    std::env::var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        .ok()
        .or_else(|| std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok())
}

/// Build an OTEL tracer provider if endpoint configuration is valid.
fn build_otel_tracer_provider(
    otel_endpoint: Option<String>,
    service_name: &str,
) -> Option<SdkTracerProvider> {
    let resource = Resource::builder()
        .with_attributes(vec![
            KeyValue::new("service.name", service_name.to_string()),
            KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
        ])
        .build();

    match otel_endpoint {
        Some(endpoint) => match opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_http_client(reqwest::Client::new())
            .with_endpoint(endpoint.clone())
            .build()
        {
            Ok(exporter) => Some(
                SdkTracerProvider::builder()
                    .with_resource(resource)
                    .with_batch_exporter(exporter)
                    .build(),
            ),
            Err(e) => {
                eprintln!(
                    "Failed to create OTLP exporter for endpoint '{}': {}. Continuing without OTLP sink.",
                    endpoint, e
                );
                None
            }
        },
        None => None,
    }
}

/// Initialize telemetry: fmt layer + env filter + optional OTLP traces.
/// Returns the TracerProvider if OTLP was configured (caller must shut it down).
fn init_telemetry(json: bool) -> Option<SdkTracerProvider> {
    let otel_endpoint = otel_endpoint_from_env();
    let otel_endpoint_configured = otel_endpoint.is_some();
    let service_name =
        std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "pgai-worker".to_string());
    let provider = build_otel_tracer_provider(otel_endpoint, &service_name);

    // Each combination of (json, otel) needs its own subscriber stack because
    // the concrete types differ and tracing_subscriber is generic.
    match (json, &provider) {
        (true, Some(p)) => {
            let tracer = p.tracer("pgai-worker");
            tracing_subscriber::registry()
                .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
                .with(tracing_subscriber::fmt::layer().json())
                .with(tracing_opentelemetry::layer().with_tracer(tracer))
                .init();
        }
        (false, Some(p)) => {
            let tracer = p.tracer("pgai-worker");
            tracing_subscriber::registry()
                .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
                .with(tracing_subscriber::fmt::layer())
                .with(tracing_opentelemetry::layer().with_tracer(tracer))
                .init();
        }
        (true, None) => {
            tracing_subscriber::registry()
                .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
                .with(tracing_subscriber::fmt::layer().json())
                .init();
        }
        (false, None) => {
            tracing_subscriber::registry()
                .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
                .with(tracing_subscriber::fmt::layer())
                .init();
        }
    }

    if provider.is_some() {
        info!(
            otel_service_name = %service_name,
            "OpenTelemetry tracing enabled"
        );
    } else if otel_endpoint_configured {
        warn!("OpenTelemetry endpoint configured but exporter initialization failed; using local logs only");
    }

    provider
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let args = Args::parse();
    let tracer_provider = init_telemetry(args.log_format == "json");

    let cancel = CancellationToken::new();

    // Spawn signal handler
    let cancel_for_signal = cancel.clone();
    tokio::spawn(async move {
        let ctrl_c = tokio::signal::ctrl_c();
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm = signal(SignalKind::terminate())
                .expect("failed to install SIGTERM handler");
            tokio::select! {
                _ = ctrl_c => {
                    info!("Received SIGINT, initiating graceful shutdown");
                }
                _ = sigterm.recv() => {
                    info!("Received SIGTERM, initiating graceful shutdown");
                }
            }
        }
        #[cfg(not(unix))]
        {
            ctrl_c.await.expect("failed to listen for ctrl_c");
            info!("Received Ctrl+C, initiating graceful shutdown");
        }
        cancel_for_signal.cancel();
    });

    info!("Starting pgai worker-rs");

    info!(
        db_url = %redact_db_url(&args.db_url),
        poll_interval_secs = args.poll_interval,
        once = args.once,
        exit_on_error = args.exit_on_error,
        vectorizer_ids = ?args.vectorizer_ids,
        log_format = %args.log_format,
        "Worker configuration"
    );

    let worker = worker::Worker::new(
        &args.db_url,
        Duration::from_secs(args.poll_interval),
        args.once,
        args.vectorizer_ids,
        args.exit_on_error,
        cancel,
    )
    .await?;

    worker.run().await?;

    if let Some(provider) = tracer_provider {
        provider.shutdown().ok();
    }

    info!("Worker shut down cleanly");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvGuard {
        traces_endpoint: Option<String>,
        endpoint: Option<String>,
        service_name: Option<String>,
    }

    impl EnvGuard {
        fn capture() -> Self {
            Self {
                traces_endpoint: std::env::var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT").ok(),
                endpoint: std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok(),
                service_name: std::env::var("OTEL_SERVICE_NAME").ok(),
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.traces_endpoint {
                Some(v) => std::env::set_var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", v),
                None => std::env::remove_var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"),
            }
            match &self.endpoint {
                Some(v) => std::env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", v),
                None => std::env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT"),
            }
            match &self.service_name {
                Some(v) => std::env::set_var("OTEL_SERVICE_NAME", v),
                None => std::env::remove_var("OTEL_SERVICE_NAME"),
            }
        }
    }

    fn lock_env() -> (MutexGuard<'static, ()>, EnvGuard) {
        let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let guard = EnvGuard::capture();
        (lock, guard)
    }

    #[test]
    fn test_build_otel_tracer_provider_none_when_no_endpoint() {
        let (_lock, _env) = lock_env();
        std::env::remove_var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT");
        std::env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
        let provider = build_otel_tracer_provider(None, "test-worker");
        assert!(provider.is_none());
    }

    #[test]
    fn test_build_otel_tracer_provider_invalid_endpoint_falls_back() {
        let (_lock, _env) = lock_env();
        let provider = build_otel_tracer_provider(Some("http://[::1".to_string()), "test-worker");
        assert!(provider.is_none());
    }

    #[test]
    fn test_build_otel_tracer_provider_valid_endpoint_returns_provider() {
        let (_lock, _env) = lock_env();
        let provider = build_otel_tracer_provider(
            Some("http://localhost:4318/v1/traces".to_string()),
            "test-worker",
        );
        assert!(provider.is_some());
        if let Some(p) = provider {
            p.shutdown().ok();
        }
    }

    #[test]
    fn test_otel_endpoint_from_env_prefers_traces_endpoint() {
        let (_lock, _env) = lock_env();
        std::env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318");
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "http://localhost:4318/v1/traces",
        );
        let endpoint = otel_endpoint_from_env();
        assert_eq!(endpoint.as_deref(), Some("http://localhost:4318/v1/traces"));
    }
}
