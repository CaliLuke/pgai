# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

pgai is a Rust vectorizer worker that turns PostgreSQL into a retrieval engine for RAG applications. It automatically creates and synchronizes vector embeddings from PostgreSQL data.

## Repository Structure

A Cargo workspace at the repo root with two crates:

- **`worker/`** — Vectorizer worker. Uses tokio, sqlx, async-openai, ollama-rs, backon, text-splitter.
- **`extension/`** — PostgreSQL extension via pgrx. Builds as a cdylib. Supports PG 13-18.

## Prerequisites

- **Rust toolchain** (stable)
- **Podman or Docker** — for integration tests (testcontainers)

## Common Commands

### Worker

```bash
cargo build -p worker
cargo test -p worker --lib              # Unit tests
cargo test --test integration -p worker -- --nocapture  # Integration tests
```

### Extension

```bash
cargo build -p extension
cargo test -p extension
```

Uses pgrx — select PG version via feature flags (default: `pg18`).

## Architecture — Worker (`worker/src/`)

- **`main.rs`** — CLI entry point (clap)
- **`worker.rs`** — Main loop: poll DB for vectorizers, dispatch executors
- **`executor.rs`** — Pipeline: fetch work -> chunk -> format -> embed with retry -> write results
- **`embedder.rs`** — Trait + implementations for embedding providers (OpenAI, Ollama)
- **`errors.rs`** — `EmbeddingError` enum (Transient/Permanent) with pattern-based classification
- **`worker_tracking.rs`** — Heartbeat + per-vectorizer progress reporting to DB
- **`models.rs`** — Serde models for vectorizer config (embedding, chunking, formatting, loading, destination)
- **`lib.rs`** — Module exports

## Container Runtime

This machine uses **Podman**, not Docker. For integration tests:

```bash
DOCKER_HOST=unix:///var/folders/ly/gzmh62m90k162x5tz_30m6fm0000gn/T/podman/podman-machine-default-api.sock \
TESTCONTAINERS_RYUK_DISABLED=true \
cargo test --test integration -p worker -- --nocapture
```

## Code Style

- **Rust**: standard rustfmt / clippy conventions
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) required. Body lines max 100 chars.
