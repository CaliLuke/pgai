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
```

### Worker integration tests (testcontainers + PostgreSQL container)

Integration tests require a running container runtime that testcontainers can reach.

#### Option A: Docker

1. Start Docker Desktop.
2. Verify socket is available:

```bash
ls -l /var/run/docker.sock
```

3. Run integration tests:

```bash
cargo test --test integration -p worker -- --nocapture
```

#### Option B: Podman (macOS)

1. Start the Podman machine:

```bash
podman machine start
```

2. Point testcontainers to the Podman socket and disable Ryuk:

```bash
export DOCKER_HOST="unix://$(podman machine inspect --format '{{.ConnectionInfo.PodmanSocket.Path}}')"
export TESTCONTAINERS_RYUK_DISABLED=true
```

3. Run integration tests:

```bash
cargo test --test integration -p worker -- --nocapture
```

#### Run a single integration test

```bash
cargo test --test integration -p worker test_concurrency_failure_propagates_with_exit_on_error -- --nocapture
```

#### Troubleshooting

- `SocketNotFoundError(\"/var/run/docker.sock\")`:
  - Docker is not running, or you are using Podman without `DOCKER_HOST` set.
- Containers fail to start with Podman:
  - Ensure `TESTCONTAINERS_RYUK_DISABLED=true` is set.
- Connection errors right after container start:
  - Re-run once; testcontainers may race briefly with PostgreSQL readiness on slower machines.

## Commit standards

[Conventional Commits](https://www.conventionalcommits.org/). Body lines max 100 chars.

```text
feat: add sentence chunker
fix: handle null embedding text gracefully
chore: bump dependencies
```
