# Integration Testing Guide (Rust Port)

To verify the Rust port works "for real" against a live database, follow these steps.

## Option 1: Automated Integration Test
Uses `testcontainers` to spin up a Postgres instance automatically.

```bash
cd worker
just test-e2e
```
*Note: Requires Podman/Docker to be running.*

## Option 2: Manual Verification (The "Real" Way)
Follow the project's quickstart pattern using the new Rust worker.

### 1. Start the project's test environment
```bash
# From the root
cd projects/extension
docker compose up -d
```

### 2. Prepare the database
```bash
# Install the extension (this runs our new Rust extension if built/mounted, 
# or the original one for worker cross-compat testing)
psql -h localhost -p 5432 -U postgres -c "CREATE EXTENSION IF NOT EXISTS ai CASCADE;"

# Create a test table
psql -h localhost -p 5432 -U postgres -c "CREATE TABLE notes (id serial primary key, content text);"
psql -h localhost -p 5432 -U postgres -c "INSERT INTO notes (content) VALUES ('Rust is the future of systems programming'), ('Postgres is the ultimate database');"

# Create a vectorizer
psql -h localhost -p 5432 -U postgres -c "SELECT ai.create_vectorizer('notes'::regclass, config => '{"version": "1.0", "embedding": {"implementation": "openai", "model": "text-embedding-3-small"}, "chunking": {"implementation": "none"}, "formatting": {"implementation": "chunk_value"}, "loading": {"implementation": "column", "column_name": "content"}, "destination": {"implementation": "table", "target_table": "notes_embeddings"}}'::jsonb);"
```

### 3. Run the Rust Worker
```bash
cd worker
cargo run -- --db-url "postgres://postgres:postgres@localhost:5432/postgres" --once
```

### 4. Verify Results
```bash
psql -h localhost -p 5432 -U postgres -c "SELECT * FROM ai.notes_embeddings;"
```

## Why this is "Real"
1. It uses the **actual Postgres process**.
2. It uses the **actual `ai.vectorizer` table** structure.
3. It exercises the **entire polling logic** including advisory locks.
4. It verifies the **embedding generation and write-back** logic.
