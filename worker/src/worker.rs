use anyhow::{anyhow, Result};
use sqlx::{postgres::PgRow, Pool, Postgres, Row};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn, error, debug};
use crate::models::{Version, Vectorizer};
use crate::worker_tracking::WorkerTracking;

pub struct Worker {
    pool: Pool<Postgres>,
    poll_interval: Duration,
    once: bool,
    vectorizer_ids: Vec<i32>,
    exit_on_error: bool,
    cancel: CancellationToken,
}

impl Worker {
    pub async fn new(
        db_url: &str,
        poll_interval: Duration,
        once: bool,
        vectorizer_ids: Vec<i32>,
        exit_on_error: bool,
        cancel: CancellationToken,
    ) -> Result<Self> {
        let pool = Pool::connect(db_url).await?;
        Ok(Self {
            pool,
            poll_interval,
            once,
            vectorizer_ids,
            exit_on_error,
            cancel,
        })
    }

    pub async fn run(self) -> Result<()> {
        info!("Starting worker loop");

        let mut tracking = WorkerTracking::new(
            self.pool.clone(),
            self.poll_interval,
            self.cancel.clone(),
        );
        tracking.start(env!("CARGO_PKG_VERSION")).await;
        let tracking = Arc::new(tracking);

        let result = self.run_loop(&tracking).await;

        // Send final heartbeat on shutdown
        match Arc::try_unwrap(tracking) {
            Ok(mut t) => t.stop().await,
            Err(arc) => {
                // Other references still exist; best-effort stop via a clone isn't possible,
                // so just log a warning. This shouldn't happen in practice.
                warn!("Could not unwrap tracking Arc ({} refs remain), skipping final heartbeat",
                    Arc::strong_count(&arc));
            }
        }

        result
    }

    async fn run_loop(&self, tracking: &Arc<WorkerTracking>) -> Result<()> {
        loop {
            if self.cancel.is_cancelled() {
                info!("Shutdown requested, exiting worker loop");
                break;
            }

            match self.run_once(tracking).await {
                Ok(_) => {
                    if self.once {
                        break;
                    }
                }
                Err(e) => {
                    let msg = e.to_string();
                    error!("Error in worker loop: {}", msg);
                    tracking.save_vectorizer_error(None, &msg).await;
                    if self.exit_on_error {
                        return Err(e);
                    }
                }
            }

            if self.once {
                break;
            }

            debug!("Sleeping for {:?}", self.poll_interval);
            tokio::select! {
                _ = tokio::time::sleep(self.poll_interval) => {}
                _ = self.cancel.cancelled() => {
                    info!("Shutdown requested during sleep, exiting worker loop");
                    break;
                }
            }
        }

        Ok(())
    }

    #[tracing::instrument(skip(self, tracking))]
    async fn run_once(&self, tracking: &Arc<WorkerTracking>) -> Result<()> {
        let version = self.get_pgai_version().await?;
        if version.ext_version.is_none() && version.pgai_lib_version.is_none() {
            return Err(anyhow!("pgai is not installed in the database"));
        }

        let vectorizer_ids = self.get_vectorizer_ids().await?;
        if vectorizer_ids.is_empty() {
            warn!("No vectorizers found");
            return Ok(());
        }

        // Load all vectorizers, filtering out disabled and failed-to-load ones
        let mut vectorizers = Vec::new();
        for id in vectorizer_ids {
            let vectorizer = match self.get_vectorizer(id).await {
                Ok(v) => v,
                Err(e) => {
                    error!(vectorizer_id = id, "Failed to load vectorizer: {e}");
                    continue;
                }
            };
            if vectorizer.disabled {
                info!("Skipping disabled vectorizer {}", id);
                continue;
            }
            vectorizers.push(vectorizer);
        }

        // Process all vectorizers concurrently
        let mut join_set = tokio::task::JoinSet::new();
        for vectorizer in vectorizers {
            let pool = self.pool.clone();
            let cancel = self.cancel.clone();
            let tracking = Arc::clone(tracking);
            let vid = vectorizer.id;
            join_set.spawn(async move {
                info!("Running vectorizer {}", vid);
                let result = process_vectorizer(pool, cancel, vectorizer, &tracking).await;
                (vid, result)
            });
        }

        let mut first_error: Option<anyhow::Error> = None;
        while let Some(result) = join_set.join_next().await {
            if self.cancel.is_cancelled() {
                join_set.abort_all();
                info!("Shutdown requested, aborting remaining vectorizers");
                break;
            }
            match result {
                Ok((vid, Ok(()))) => debug!("Vectorizer {} completed", vid),
                Ok((vid, Err(e))) => {
                    error!(vectorizer_id = vid, "Failed to process vectorizer: {e}");
                    tracking.save_vectorizer_error(Some(vid), &e.to_string()).await;
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
                Err(e) => error!("Vectorizer task panicked: {e}"),
            }
        }

        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    async fn get_pgai_version(&self) -> Result<Version> {
        let ext_version: Option<String> = sqlx::query_scalar(
            "select extversion from pg_catalog.pg_extension where extname = 'ai'"
        )
        .fetch_optional(&self.pool)
        .await?;

        let table_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'ai' AND table_name = 'pgai_lib_version'
            )"
        )
        .fetch_one(&self.pool)
        .await?;

        let mut pgai_lib_version = None;
        if table_exists {
            pgai_lib_version = sqlx::query_scalar(
                "select version from ai.pgai_lib_version where name = 'ai'"
            )
            .fetch_optional(&self.pool)
            .await?;
        }

        Ok(Version {
            ext_version,
            pgai_lib_version,
        })
    }

    async fn get_vectorizer_ids(&self) -> Result<Vec<i32>> {
        let ids = if self.vectorizer_ids.is_empty() {
            sqlx::query_scalar("select id from ai.vectorizer")
                .fetch_all(&self.pool)
                .await?
        } else {
            sqlx::query_scalar("select id from ai.vectorizer where id = ANY($1)")
                .bind(&self.vectorizer_ids)
                .fetch_all(&self.pool)
                .await?
        };
        Ok(ids)
    }

    async fn get_vectorizer(&self, id: i32) -> Result<Vectorizer> {
        let row: PgRow = sqlx::query(
            "select pg_catalog.to_jsonb(v) as vectorizer from ai.vectorizer v where v.id = $1"
        )
        .bind(id)
        .fetch_one(&self.pool)
        .await?;

        let val: serde_json::Value = row.try_get("vectorizer")?;
        let vectorizer: Vectorizer = serde_json::from_value(val)?;
        Ok(vectorizer)
    }

}

#[tracing::instrument(skip(pool, cancel, tracking), fields(vectorizer_id = vectorizer.id))]
async fn process_vectorizer(
    pool: Pool<Postgres>,
    cancel: CancellationToken,
    vectorizer: Vectorizer,
    tracking: &Arc<WorkerTracking>,
) -> Result<()> {
    let concurrency = (vectorizer.config.processing.concurrency.max(1) as usize).min(10);

    if concurrency == 1 {
        let executor = crate::executor::Executor::new(
            pool, vectorizer, cancel, Arc::clone(tracking),
        ).await?;
        executor.run().await?;
        return Ok(());
    }

    info!("Running vectorizer {} with concurrency {}", vectorizer.id, concurrency);
    let mut join_set = tokio::task::JoinSet::new();
    for _ in 0..concurrency {
        let pool = pool.clone();
        let v = vectorizer.clone();
        let cancel = cancel.clone();
        let tracking = Arc::clone(tracking);
        join_set.spawn(async move {
            let executor = crate::executor::Executor::new(pool, v, cancel, tracking).await?;
            executor.run().await
        });
    }

    while let Some(result) = join_set.join_next().await {
        if cancel.is_cancelled() {
            join_set.abort_all();
            info!("Shutdown requested, aborting remaining executors for vectorizer {}", vectorizer.id);
            break;
        }
        match result {
            Ok(Ok(count)) => info!("Executor for vectorizer {} processed {} items", vectorizer.id, count),
            Ok(Err(e)) => error!("Executor failed for vectorizer {}: {}", vectorizer.id, e),
            Err(e) => error!("Executor panicked for vectorizer {}: {}", vectorizer.id, e),
        }
    }
    Ok(())
}
