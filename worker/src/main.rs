use anyhow::Result;
use clap::Parser;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace::SdkTracerProvider;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::info;
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
}

/// Initialize telemetry: fmt layer + env filter + optional OTLP traces.
/// Returns the TracerProvider if OTLP was configured (caller must shut it down).
fn init_telemetry() -> Option<SdkTracerProvider> {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let fmt_layer = tracing_subscriber::fmt::layer();

    let otel_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok();

    if let Some(endpoint) = otel_endpoint {
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_endpoint(endpoint)
            .build()
            .expect("failed to create OTLP exporter");

        let provider = SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .build();

        let tracer = provider.tracer("pgai-worker");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .with(otel_layer)
            .init();

        info!("OpenTelemetry tracing enabled");
        Some(provider)
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .init();

        None
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let tracer_provider = init_telemetry();

    let args = Args::parse();
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
