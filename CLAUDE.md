# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Documentation

Project documentation lives in `docs/`:

- `docs/setup-guide.md` — getting started: install, create vectorizer, run worker, troubleshooting
- `docs/architecture.md` — workspace layout, data flow, module descriptions
- `docs/sql-reference.md` — all SQL functions, config helpers, table schemas
- `docs/worker-tracking.md` — worker health monitoring, heartbeat, progress reporting
- `docs/worker-config.md` — worker CLI flags and environment variables

## Project Overview

pgai is a Rust vectorizer worker that turns PostgreSQL into a retrieval engine for RAG applications. It automatically creates and synchronizes vector embeddings from PostgreSQL data.

## Repository Structure

A Cargo workspace at the repo root with two crates:

- **`worker/`** — Vectorizer worker. Uses tokio, sqlx, async-openai, ollama-rs, backon, text-splitter.
- **`extension/`** — PostgreSQL extension via pgrx. Builds as a cdylib. Supports PG 13-18.

## Prerequisites

- **Rust toolchain** (stable)
- **Podman or Docker** — for integration tests (testcontainers)
- **cargo-pgrx 0.16.1** — for extension development (`cargo install cargo-pgrx --version 0.16.1`)

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
cargo pgrx test pg18 -p extension       # Rust-level pg_test (runs in real PG)
cargo pgrx regress -p extension pg18    # SQL regression tests (pg_regress)
```

Uses pgrx 0.16.1 — select PG version via feature flags (default: `pg18`).

## Architecture — Worker (`worker/src/`)

- **`main.rs`** — CLI entry point (clap)
- **`worker.rs`** — Main loop: poll DB for vectorizers, dispatch executors
- **`executor.rs`** — Pipeline: fetch work -> chunk -> format -> embed with retry -> write results
- **`embedder.rs`** — Trait + implementations for embedding providers (OpenAI, Ollama)
- **`errors.rs`** — `EmbeddingError` enum (Transient/Permanent) with pattern-based classification
- **`worker_tracking.rs`** — Heartbeat + per-vectorizer progress reporting to DB (requires `ai.vectorizer_worker_process` table; feature-detected at startup)
- **`models.rs`** — Serde models for vectorizer config (embedding, chunking, formatting, loading, destination)
- **`lib.rs`** — Module exports

## Extension Setup (pgrx)

The extension uses pgrx 0.16.1. **Do not use Homebrew PostgreSQL** for `cargo pgrx test` —
it lacks the static server library needed to link the test binary.

### One-time setup

1. **Initialize pgrx with source-built PostgreSQL:**

   ```bash
   # macOS needs ICU paths for the PG build
   ICU_CFLAGS="-I$(brew --prefix icu4c)/include" \
   ICU_LIBS="-L$(brew --prefix icu4c)/lib -licuuc -licudata -licui18n" \
   PKG_CONFIG_PATH="$(brew --prefix icu4c)/lib/pkgconfig:$PKG_CONFIG_PATH" \
   cargo pgrx init --pg18 download
   ```

   This downloads and compiles PG 18 to `~/.pgrx/`. Takes ~5 min; cached after that.

2. **Install pgvector into pgrx-managed PG** (required by `extension.control`):

   ```bash
   cd /tmp && git clone --depth 1 https://github.com/pgvector/pgvector.git && cd pgvector
   PG_CONFIG=~/.pgrx/18.1/pgrx-install/bin/pg_config make && make install
   ```

### Key extension files

| File                              | Purpose                                                        |
| --------------------------------- | -------------------------------------------------------------- |
| `extension.control`               | PG metadata; `requires = 'vector'` handles pgvector dep        |
| `sql/setup.sql`                   | SQL loaded at `CREATE EXTENSION` time (bootstrap)              |
| `src/lib.rs`                      | Rust extension code + `#[pg_test]` tests                       |
| `src/bin/pgrx_embed_extension.rs` | pgrx embed binary (name must match package name)               |
| `Cargo.toml`                      | `crate-type = ["cdylib", "lib"]` — lib target needed for tests |
| `.cargo/config.toml`              | macOS `-Wl,-undefined,dynamic_lookup` linker flag              |
| `tests/pg_regress/sql/*.sql`      | SQL regression test files                                      |
| `tests/pg_regress/expected/*.out` | Expected regression output                                     |

### Troubleshooting

- **Stale pg_regress DB** (vectorizer IDs don't start at 1): drop and recreate:

  ```bash
  cargo pgrx start pg18 -p extension
  ~/.pgrx/18.1/pgrx-install/bin/psql -h localhost -p 28818 -d postgres \
    -c "DROP DATABASE IF EXISTS extension_regress;"
  cargo pgrx regress -p extension pg18
  ```

- **Link errors** (palloc, errstart not found): you're using Homebrew PG.
  Run `cargo pgrx init --pg18 download` to use pgrx-managed PG instead.
- **`pgrx_embed` binary not found**: binary must be named `pgrx_embed_{package_name}.rs`.
- **Nested CREATE EXTENSION error**: the `requires = 'vector'` in `extension.control`
  handles this. Do not add `CREATE EXTENSION vector` to `setup.sql`.

## Container Runtime

For worker integration tests, set `DOCKER_HOST` to your container runtime socket
and disable Ryuk:

```bash
DOCKER_HOST=unix://$DOCKER_SOCKET \
TESTCONTAINERS_RYUK_DISABLED=true \
cargo test --test integration -p worker -- --nocapture
```

## Code Style

- **Rust**: standard rustfmt / clippy conventions
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) required. Body lines max 100 chars. **Do NOT add `Co-Authored-By` lines.**
