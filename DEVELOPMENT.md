# Development

## Prerequisites

- **Rust toolchain** (stable) — `rustup`, `cargo`
- **Podman or Docker** — for integration tests (testcontainers)

## Project structure

```text
Cargo.toml          — Workspace root
worker/             — Vectorizer worker (tokio, sqlx, async-openai, ollama-rs)
extension/          — PostgreSQL extension (pgrx 0.16.1, PG 13-18)
text-splitter/      — Text chunking library
scripts/            — Load test tooling
plans/              — Coordination docs (temporary)
```

## Building

```bash
cargo build -p worker              # worker binary
cargo check -p extension           # extension type-check (full build needs cargo pgrx)
cargo build -p pgai-text-splitter  # text splitter library
cargo check --workspace            # everything
```

## Testing

```bash
# Text splitter (60 tests)
cargo test -p pgai-text-splitter

# Worker unit tests (37 tests)
cargo test -p worker --lib

# Worker integration tests (needs container runtime)
DOCKER_HOST=unix:///var/folders/ly/gzmh62m90k162x5tz_30m6fm0000gn/T/podman/podman-machine-default-api.sock \
TESTCONTAINERS_RYUK_DISABLED=true \
cargo test --test integration -p worker -- --nocapture
```

## Commit standards

[Conventional Commits](https://www.conventionalcommits.org/). Body lines max 100 chars.

```text
feat: add sentence chunker
fix: handle null embedding text gracefully
chore: bump dependencies
```
