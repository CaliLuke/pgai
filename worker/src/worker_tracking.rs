use sqlx::{Pool, Postgres};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Atomic counters for heartbeat reporting.
struct Counters {
    successes: AtomicI64,
    errors: AtomicI64,
    last_error_msg: Mutex<Option<String>>,
}

impl Counters {
    fn new() -> Self {
        Self {
            successes: AtomicI64::new(0),
            errors: AtomicI64::new(0),
            last_error_msg: Mutex::new(None),
        }
    }

    fn add_successes(&self, count: i64) {
        self.successes.fetch_add(count, Ordering::Relaxed);
    }

    fn add_error(&self, msg: String) {
        self.errors.fetch_add(1, Ordering::Relaxed);
        *self.last_error_msg.lock().unwrap() = Some(msg);
    }

    /// Atomically swap counters to zero and return the accumulated values.
    fn swap(&self) -> (i64, i64, Option<String>) {
        let successes = self.successes.swap(0, Ordering::Relaxed);
        let errors = self.errors.swap(0, Ordering::Relaxed);
        let last_error = self.last_error_msg.lock().unwrap().take();
        (successes, errors, last_error)
    }

    /// Restore counters (used when heartbeat DB call fails).
    fn restore(&self, successes: i64, errors: i64, last_error: Option<String>) {
        self.successes.fetch_add(successes, Ordering::Relaxed);
        self.errors.fetch_add(errors, Ordering::Relaxed);
        if let Some(msg) = last_error {
            *self.last_error_msg.lock().unwrap() = Some(msg);
        }
    }
}

/// Tracks worker lifecycle and sends periodic heartbeats to the database.
///
/// When heartbeat tracking is enabled (the required DB tables exist), it:
/// - Registers the worker on start via `ai._worker_start()`
/// - Sends periodic heartbeats via `ai._worker_heartbeat()`
/// - Reports per-vectorizer progress via `ai._worker_progress()`
/// - Sends a final heartbeat on shutdown
pub struct WorkerTracking {
    pool: Pool<Postgres>,
    worker_id: Option<Uuid>,
    enabled: bool,
    counters: Arc<Counters>,
    heartbeat_handle: Option<JoinHandle<()>>,
    cancel: CancellationToken,
    poll_interval: Duration,
}

impl WorkerTracking {
    pub fn new(pool: Pool<Postgres>, poll_interval: Duration, cancel: CancellationToken) -> Self {
        Self {
            pool,
            worker_id: None,
            enabled: false,
            counters: Arc::new(Counters::new()),
            heartbeat_handle: None,
            cancel,
            poll_interval,
        }
    }

    /// Feature-detect heartbeat support and register the worker.
    pub async fn start(&mut self, version: &str) {
        // Check if the tracking table exists
        let table_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'ai' AND table_name = 'vectorizer_worker_process'
            )",
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(false);

        if !table_exists {
            info!("Worker tracking table not found, heartbeat disabled");
            return;
        }

        // Register the worker with version and expected heartbeat interval
        let interval_secs = self.poll_interval.as_secs() as f64;
        let pg_interval = sqlx::postgres::types::PgInterval {
            months: 0,
            days: 0,
            microseconds: (interval_secs * 1_000_000.0) as i64,
        };
        let worker_id = match sqlx::query_scalar::<_, Uuid>(
            "SELECT ai._worker_start($1::text, $2::interval)",
        )
        .bind(version)
        .bind(pg_interval)
        .fetch_one(&self.pool)
        .await
        {
            Ok(id) => id,
            Err(e) => {
                warn!("Failed to register worker, heartbeat disabled: {}", e);
                return;
            }
        };

        info!("Worker registered with id {}", worker_id);
        self.worker_id = Some(worker_id);
        self.enabled = true;

        // Spawn heartbeat background task
        let pool = self.pool.clone();
        let counters = Arc::clone(&self.counters);
        let cancel = self.cancel.clone();
        let interval = self.poll_interval;

        self.heartbeat_handle = Some(tokio::spawn(async move {
            heartbeat_loop(pool, worker_id, counters, cancel, interval).await;
        }));
    }

    /// Record a successful embedding batch for a vectorizer.
    pub async fn save_vectorizer_success(&self, vectorizer_id: i32, count: i32) {
        self.counters.add_successes(count as i64);

        if !self.enabled {
            return;
        }

        let Some(worker_id) = self.worker_id else {
            return;
        };

        if let Err(e) = sqlx::query("SELECT ai._worker_progress($1, $2, $3, NULL)")
            .bind(worker_id)
            .bind(vectorizer_id)
            .bind(count)
            .execute(&self.pool)
            .await
        {
            debug!("Failed to report vectorizer progress: {}", e);
        }
    }

    /// Record an embedding error for a vectorizer.
    pub async fn save_vectorizer_error(&self, vectorizer_id: Option<i32>, msg: &str) {
        self.counters.add_error(msg.to_string());

        if !self.enabled {
            return;
        }

        let Some(worker_id) = self.worker_id else {
            return;
        };

        if let Some(vid) = vectorizer_id {
            if let Err(e) = sqlx::query("SELECT ai._worker_progress($1, $2, 0, $3)")
                .bind(worker_id)
                .bind(vid)
                .bind(msg)
                .execute(&self.pool)
                .await
            {
                debug!("Failed to report vectorizer error progress: {}", e);
            }
        }
    }

    /// Send a final heartbeat and clean up.
    pub async fn stop(&mut self) {
        if let Some(handle) = self.heartbeat_handle.take() {
            handle.abort();
            let _ = handle.await;
        }

        if !self.enabled {
            return;
        }

        let Some(worker_id) = self.worker_id else {
            return;
        };

        // Send one final heartbeat with remaining counters
        let (successes, errors, last_error) = self.counters.swap();
        if let Err(e) = sqlx::query("SELECT ai._worker_heartbeat($1, $2, $3, $4)")
            .bind(worker_id)
            .bind(successes)
            .bind(errors)
            .bind(last_error)
            .execute(&self.pool)
            .await
        {
            warn!("Failed to send final heartbeat: {}", e);
        } else {
            debug!("Sent final heartbeat for worker {}", worker_id);
        }
    }
}

async fn heartbeat_loop(
    pool: Pool<Postgres>,
    worker_id: Uuid,
    counters: Arc<Counters>,
    cancel: CancellationToken,
    interval: Duration,
) {
    let mut consecutive_failures = 0u32;
    const MAX_CONSECUTIVE_FAILURES: u32 = 3;

    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = cancel.cancelled() => {
                debug!("Heartbeat loop cancelled");
                break;
            }
        }

        let (successes, errors, last_error) = counters.swap();

        match sqlx::query("SELECT ai._worker_heartbeat($1, $2, $3, $4)")
            .bind(worker_id)
            .bind(successes)
            .bind(errors)
            .bind(&last_error)
            .execute(&pool)
            .await
        {
            Ok(_) => {
                consecutive_failures = 0;
                debug!(
                    "Heartbeat sent: worker={}, successes={}, errors={}",
                    worker_id, successes, errors
                );
            }
            Err(e) => {
                consecutive_failures += 1;
                warn!(
                    "Heartbeat failed ({}/{}): {}",
                    consecutive_failures, MAX_CONSECUTIVE_FAILURES, e
                );
                // Restore counters so they aren't lost
                counters.restore(successes, errors, last_error);

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                    error!(
                        "Heartbeat failed {} consecutive times, stopping heartbeat loop",
                        MAX_CONSECUTIVE_FAILURES
                    );
                    break;
                }
            }
        }
    }
}
