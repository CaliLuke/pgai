-- pgai vectorizer infrastructure
-- Loaded at CREATE EXTENSION time by pgrx

-------------------------------------------------------------------------------
-- Schema and metadata tables
-------------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS ai;

CREATE SEQUENCE IF NOT EXISTS ai.vectorizer_id_seq;

CREATE TABLE IF NOT EXISTS ai.vectorizer (
    id               int4 NOT NULL PRIMARY KEY DEFAULT nextval('ai.vectorizer_id_seq'),
    source_schema    name NOT NULL,
    source_table     name NOT NULL,
    source_pk        jsonb NOT NULL,
    trigger_name     name NOT NULL,
    queue_schema     name,
    queue_table      name,
    queue_failed_table name,
    config           jsonb NOT NULL,
    name             text UNIQUE,
    disabled         bool NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS ai.vectorizer_errors (
    id       int4 NOT NULL REFERENCES ai.vectorizer(id) ON DELETE CASCADE,
    message  text,
    details  jsonb,
    recorded timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_vectorizer_errors_id_recorded
    ON ai.vectorizer_errors(id, recorded);

-------------------------------------------------------------------------------
-- Config helper functions
-------------------------------------------------------------------------------

-- loading
CREATE OR REPLACE FUNCTION ai.loading_column(
    column_name name,
    retries int4 DEFAULT 6
) RETURNS jsonb AS $$
    SELECT json_build_object(
        'implementation', 'column',
        'config_type', 'loading',
        'column_name', column_name,
        'retries', retries
    )::jsonb
$$ LANGUAGE sql IMMUTABLE;

-- embedding: ollama
CREATE OR REPLACE FUNCTION ai.embedding_ollama(
    model text,
    dimensions int4,
    base_url text DEFAULT NULL,
    options jsonb DEFAULT NULL,
    keep_alive text DEFAULT NULL
) RETURNS jsonb AS $$
    SELECT json_strip_nulls(json_build_object(
        'implementation', 'ollama',
        'config_type', 'embedding',
        'model', model,
        'dimensions', dimensions,
        'base_url', base_url,
        'options', options,
        'keep_alive', keep_alive
    ))::jsonb
$$ LANGUAGE sql IMMUTABLE;

-- embedding: openai
CREATE OR REPLACE FUNCTION ai.embedding_openai(
    model text,
    dimensions int4,
    chat_user text DEFAULT NULL,
    api_key_name text DEFAULT 'OPENAI_API_KEY',
    base_url text DEFAULT NULL
) RETURNS jsonb AS $$
    SELECT json_strip_nulls(json_build_object(
        'implementation', 'openai',
        'config_type', 'embedding',
        'model', model,
        'dimensions', dimensions,
        'user', chat_user,
        'api_key_name', api_key_name,
        'base_url', base_url
    ))::jsonb
$$ LANGUAGE sql IMMUTABLE;

-- chunking: none
CREATE OR REPLACE FUNCTION ai.chunking_none() RETURNS jsonb AS $$
    SELECT json_build_object(
        'implementation', 'none',
        'config_type', 'chunking'
    )::jsonb
$$ LANGUAGE sql IMMUTABLE;

-- chunking: recursive character text splitter
CREATE OR REPLACE FUNCTION ai.chunking_recursive_character_text_splitter(
    chunk_size int4 DEFAULT 800,
    chunk_overlap int4 DEFAULT 400,
    separators text[] DEFAULT ARRAY[E'\n\n', E'\n', '.', '?', '!', ' ', ''],
    is_separator_regex bool DEFAULT false
) RETURNS jsonb AS $$
    SELECT json_strip_nulls(json_build_object(
        'implementation', 'recursive_character_text_splitter',
        'config_type', 'chunking',
        'chunk_size', chunk_size,
        'chunk_overlap', chunk_overlap,
        'separators', separators,
        'is_separator_regex', is_separator_regex
    ))::jsonb
$$ LANGUAGE sql IMMUTABLE;

-- chunking: character text splitter
CREATE OR REPLACE FUNCTION ai.chunking_character_text_splitter(
    chunk_size int4 DEFAULT 800,
    chunk_overlap int4 DEFAULT 400,
    separator text DEFAULT E'\n\n',
    is_separator_regex bool DEFAULT false
) RETURNS jsonb AS $$
    SELECT json_strip_nulls(json_build_object(
        'implementation', 'character_text_splitter',
        'config_type', 'chunking',
        'chunk_size', chunk_size,
        'chunk_overlap', chunk_overlap,
        'separator', separator,
        'is_separator_regex', is_separator_regex
    ))::jsonb
$$ LANGUAGE sql IMMUTABLE;

-- processing
CREATE OR REPLACE FUNCTION ai.processing_default(
    batch_size int4 DEFAULT NULL,
    concurrency int4 DEFAULT NULL
) RETURNS jsonb AS $$
    SELECT json_strip_nulls(json_build_object(
        'implementation', 'default',
        'config_type', 'processing',
        'batch_size', batch_size,
        'concurrency', concurrency
    ))::jsonb
$$ LANGUAGE sql IMMUTABLE;

-- destination: table
CREATE OR REPLACE FUNCTION ai.destination_table(
    target_schema name DEFAULT NULL,
    target_table name DEFAULT NULL,
    view_schema name DEFAULT NULL,
    view_name name DEFAULT NULL
) RETURNS jsonb AS $$
    SELECT json_strip_nulls(json_build_object(
        'implementation', 'table',
        'config_type', 'destination',
        'target_schema', target_schema,
        'target_table', target_table,
        'view_schema', view_schema,
        'view_name', view_name
    ))::jsonb
$$ LANGUAGE sql IMMUTABLE;

-- formatting: python template
CREATE OR REPLACE FUNCTION ai.formatting_python_template(
    template text DEFAULT '$chunk'
) RETURNS jsonb AS $$
    SELECT json_build_object(
        'implementation', 'python_template',
        'config_type', 'formatting',
        'template', template
    )::jsonb
$$ LANGUAGE sql IMMUTABLE;

-- formatting: chunk_value
CREATE OR REPLACE FUNCTION ai.formatting_chunk_value() RETURNS jsonb AS $$
    SELECT json_build_object(
        'implementation', 'chunk_value',
        'config_type', 'formatting'
    )::jsonb
$$ LANGUAGE sql IMMUTABLE;

-------------------------------------------------------------------------------
-- Internal helper functions
-------------------------------------------------------------------------------

-- Extract primary key metadata from a source table
CREATE OR REPLACE FUNCTION ai._vectorizer_source_pk(
    source_table regclass
) RETURNS jsonb AS $$
    SELECT jsonb_agg(x)
    FROM (
        SELECT
            e.attnum,
            e.pknum,
            a.attname,
            format_type(y.oid, a.atttypmod) AS typname
        FROM pg_constraint k
        CROSS JOIN LATERAL unnest(k.conkey) WITH ORDINALITY e(attnum, pknum)
        INNER JOIN pg_attribute a
            ON (k.conrelid = a.attrelid AND e.attnum = a.attnum)
        INNER JOIN pg_type y ON (a.atttypid = y.oid)
        WHERE k.conrelid = source_table
          AND k.contype = 'p'
    ) x
$$ LANGUAGE sql STABLE;

-- Create the embedding store (target) table
CREATE OR REPLACE FUNCTION ai._vectorizer_create_target_table(
    source_pk jsonb,
    target_schema name,
    target_table name,
    dimensions int4
) RETURNS void AS $func$
DECLARE
    _pk_cols text;
    _sql text;
BEGIN
    SELECT string_agg(format('%I', x.attname), ', ' ORDER BY x.pknum)
    INTO STRICT _pk_cols
    FROM jsonb_to_recordset(source_pk) x(pknum int, attname name);

    SELECT format(
        $sql$
        CREATE TABLE IF NOT EXISTS %I.%I (
            embedding_uuid uuid NOT NULL PRIMARY KEY DEFAULT gen_random_uuid(),
            %s,
            chunk_seq int NOT NULL,
            chunk text NOT NULL,
            embedding vector(%s) NOT NULL,
            UNIQUE (%s, chunk_seq)
        )
        $sql$,
        target_schema, target_table,
        (
            SELECT string_agg(
                format('%I %s NOT NULL', x.attname, x.typname),
                E',\n            '
                ORDER BY x.attnum
            )
            FROM jsonb_to_recordset(source_pk)
                x(attnum int, attname name, typname text)
        ),
        dimensions,
        _pk_cols
    ) INTO STRICT _sql;

    EXECUTE _sql;

    -- optimize storage for the embedding column
    EXECUTE format(
        'ALTER TABLE %I.%I ALTER COLUMN embedding SET STORAGE main',
        target_schema, target_table
    );
END;
$func$ LANGUAGE plpgsql VOLATILE;

-- Create the join view between source and target tables
CREATE OR REPLACE FUNCTION ai._vectorizer_create_view(
    view_schema name,
    view_name name,
    source_schema name,
    source_table name,
    source_pk jsonb,
    target_schema name,
    target_table name
) RETURNS void AS $func$
DECLARE
    _join_cond text;
    _sql text;
BEGIN
    SELECT string_agg(
        format('s.%I = t.%I', x.attname, x.attname),
        ' AND '
        ORDER BY x.pknum
    )
    INTO STRICT _join_cond
    FROM jsonb_to_recordset(source_pk) x(pknum int, attname name);

    SELECT format(
        $sql$
        CREATE OR REPLACE VIEW %I.%I AS
        SELECT
            t.embedding_uuid,
            t.chunk_seq,
            t.chunk,
            t.embedding,
            s.*
        FROM %I.%I s
        INNER JOIN %I.%I t ON %s
        $sql$,
        view_schema, view_name,
        source_schema, source_table,
        target_schema, target_table,
        _join_cond
    ) INTO STRICT _sql;

    EXECUTE _sql;
END;
$func$ LANGUAGE plpgsql VOLATILE;

-- Create the work queue table
CREATE OR REPLACE FUNCTION ai._vectorizer_create_queue_table(
    queue_schema name,
    queue_table name,
    source_pk jsonb
) RETURNS void AS $func$
DECLARE
    _sql text;
BEGIN
    SELECT format(
        $sql$
        CREATE TABLE IF NOT EXISTS %I.%I (
            %s,
            queued_at timestamptz NOT NULL DEFAULT now()
        )
        $sql$,
        queue_schema, queue_table,
        (
            SELECT string_agg(
                format('%I %s NOT NULL', x.attname, x.typname),
                E',\n            '
                ORDER BY x.attnum
            )
            FROM jsonb_to_recordset(source_pk)
                x(attnum int, attname name, typname text)
        )
    ) INTO STRICT _sql;

    EXECUTE _sql;

    -- index on PK columns for efficient lookups (skip if table already exists)
    IF to_regclass(format('%I.%I', queue_schema, queue_table)) IS NOT NULL THEN
        -- table was just created or already existed; only create index if none exist
        IF NOT EXISTS (
            SELECT 1 FROM pg_indexes
            WHERE schemaname = queue_schema AND tablename = queue_table
        ) THEN
            SELECT format(
                'CREATE INDEX ON %I.%I (%s)',
                queue_schema, queue_table,
                (
                    SELECT string_agg(format('%I', x.attname), ', ' ORDER BY x.pknum)
                    FROM jsonb_to_recordset(source_pk) x(pknum int, attname name)
                )
            ) INTO STRICT _sql;
            EXECUTE _sql;
        END IF;
    END IF;
END;
$func$ LANGUAGE plpgsql VOLATILE;

-- Create the failed queue table
CREATE OR REPLACE FUNCTION ai._vectorizer_create_queue_failed_table(
    queue_schema name,
    failed_table name,
    source_pk jsonb
) RETURNS void AS $func$
DECLARE
    _sql text;
BEGIN
    SELECT format(
        $sql$
        CREATE TABLE IF NOT EXISTS %I.%I (
            %s,
            created_at timestamptz NOT NULL DEFAULT now(),
            failure_step text NOT NULL DEFAULT ''
        )
        $sql$,
        queue_schema, failed_table,
        (
            SELECT string_agg(
                format('%I %s NOT NULL', x.attname, x.typname),
                E',\n            '
                ORDER BY x.attnum
            )
            FROM jsonb_to_recordset(source_pk)
                x(attnum int, attname name, typname text)
        )
    ) INTO STRICT _sql;

    EXECUTE _sql;

    -- index on PK columns (skip if table already had indexes)
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = queue_schema AND tablename = failed_table
    ) THEN
        SELECT format(
            'CREATE INDEX ON %I.%I (%s)',
            queue_schema, failed_table,
            (
                SELECT string_agg(format('%I', x.attname), ', ' ORDER BY x.pknum)
                FROM jsonb_to_recordset(source_pk) x(pknum int, attname name)
            )
        ) INTO STRICT _sql;
        EXECUTE _sql;
    END IF;
END;
$func$ LANGUAGE plpgsql VOLATILE;

-- Build the trigger function body
CREATE OR REPLACE FUNCTION ai._vectorizer_build_trigger_definition(
    queue_schema name,
    queue_table name,
    target_schema name,
    target_table name,
    source_schema name,
    source_table name,
    source_pk jsonb
) RETURNS text AS $func$
DECLARE
    _pk_cols text;
    _new_pk_vals text;
    _old_pk_match_target text;
    _old_pk_match_queue text;
    _pk_changed_check text;
    _relevant_columns_check text;
    _body text;
BEGIN
    -- "id" column list
    SELECT string_agg(format('%I', x.attname), ', ' ORDER BY x.pknum)
    INTO STRICT _pk_cols
    FROM jsonb_to_recordset(source_pk) x(pknum int, attname name);

    -- NEW.id, NEW.id2, ...
    SELECT string_agg(format('NEW.%I', x.attname), ', ' ORDER BY x.pknum)
    INTO STRICT _new_pk_vals
    FROM jsonb_to_recordset(source_pk) x(pknum int, attname name);

    -- target_table WHERE id = OLD.id AND ...
    SELECT string_agg(
        format('%I = OLD.%I', x.attname, x.attname),
        ' AND ' ORDER BY x.pknum
    )
    INTO STRICT _old_pk_match_target
    FROM jsonb_to_recordset(source_pk) x(pknum int, attname name);

    -- same for queue table
    _old_pk_match_queue = _old_pk_match_target;

    -- OLD.id IS DISTINCT FROM NEW.id OR ...
    SELECT string_agg(
        format('OLD.%I IS DISTINCT FROM NEW.%I', x.attname, x.attname),
        ' OR ' ORDER BY x.pknum
    )
    INTO STRICT _pk_changed_check
    FROM jsonb_to_recordset(source_pk) x(pknum int, attname name);

    -- Check if any non-PK column changed (to avoid re-embedding when nothing changed)
    -- We compare OLD vs NEW for all columns except PK columns
    SELECT coalesce(
        string_agg(
            format('OLD.%I IS DISTINCT FROM NEW.%I', a.attname, a.attname),
            ' OR '
        ),
        'true'  -- if no non-PK columns exist, always re-embed
    )
    INTO _relevant_columns_check
    FROM pg_attribute a
    INNER JOIN pg_class c ON a.attrelid = c.oid
    INNER JOIN pg_namespace n ON c.relnamespace = n.oid
    WHERE n.nspname = source_schema
      AND c.relname = source_table
      AND a.attnum > 0
      AND NOT a.attisdropped
      AND a.attname NOT IN (
          SELECT x.attname FROM jsonb_to_recordset(source_pk) x(attname name)
      );

    _body = format($trig$
BEGIN
    IF (TG_LEVEL = 'ROW') THEN
        IF (TG_OP = 'DELETE') THEN
            DELETE FROM %I.%I WHERE %s;
            RETURN OLD;
        ELSIF (TG_OP = 'UPDATE') THEN
            IF %s THEN
                -- PK changed: remove old embeddings, queue new row
                DELETE FROM %I.%I WHERE %s;
                INSERT INTO %I.%I (%s) VALUES (%s);
            ELSIF %s THEN
                -- relevant column changed: queue for re-embedding
                INSERT INTO %I.%I (%s) VALUES (%s);
            END IF;
            RETURN NEW;
        ELSE
            -- INSERT: queue new row
            INSERT INTO %I.%I (%s) VALUES (%s);
            RETURN NEW;
        END IF;
    ELSIF (TG_LEVEL = 'STATEMENT') THEN
        IF (TG_OP = 'TRUNCATE') THEN
            TRUNCATE TABLE %I.%I;
            TRUNCATE TABLE %I.%I;
        END IF;
        RETURN NULL;
    END IF;
    RETURN NULL;
END;
$trig$,
        -- DELETE
        target_schema, target_table, _old_pk_match_target,
        -- UPDATE: PK changed check
        _pk_changed_check,
        -- UPDATE: delete old from target
        target_schema, target_table, _old_pk_match_target,
        -- UPDATE: insert into queue
        queue_schema, queue_table, _pk_cols, _new_pk_vals,
        -- UPDATE: relevant columns changed check
        _relevant_columns_check,
        -- UPDATE: insert into queue (relevant change)
        queue_schema, queue_table, _pk_cols, _new_pk_vals,
        -- INSERT: insert into queue
        queue_schema, queue_table, _pk_cols, _new_pk_vals,
        -- TRUNCATE: truncate target and queue
        target_schema, target_table,
        queue_schema, queue_table
    );

    RETURN _body;
END;
$func$ LANGUAGE plpgsql STABLE;

-- Create the trigger function and attach triggers to the source table
CREATE OR REPLACE FUNCTION ai._vectorizer_create_source_trigger(
    trigger_name name,
    queue_schema name,
    queue_table name,
    source_schema name,
    source_table name,
    target_schema name,
    target_table name,
    source_pk jsonb
) RETURNS void AS $func$
DECLARE
    _trigger_def text;
    _sql text;
BEGIN
    -- Build the trigger function body
    _trigger_def = ai._vectorizer_build_trigger_definition(
        queue_schema, queue_table,
        target_schema, target_table,
        source_schema, source_table,
        source_pk
    );

    -- Create the trigger function
    SELECT format(
        $sql$
        CREATE OR REPLACE FUNCTION %I.%I() RETURNS trigger
        AS $trg$%s$trg$
        LANGUAGE plpgsql VOLATILE SECURITY DEFINER
        SET search_path = pg_catalog, pg_temp
        $sql$,
        queue_schema, trigger_name,
        _trigger_def
    ) INTO STRICT _sql;

    EXECUTE _sql;

    -- Row-level trigger for INSERT/UPDATE/DELETE
    SELECT format(
        $sql$
        CREATE TRIGGER %I
        AFTER INSERT OR UPDATE OR DELETE
        ON %I.%I
        FOR EACH ROW EXECUTE FUNCTION %I.%I()
        $sql$,
        trigger_name,
        source_schema, source_table,
        queue_schema, trigger_name
    ) INTO STRICT _sql;

    EXECUTE _sql;

    -- Statement-level trigger for TRUNCATE
    SELECT format(
        $sql$
        CREATE TRIGGER %I
        AFTER TRUNCATE
        ON %I.%I
        FOR EACH STATEMENT EXECUTE FUNCTION %I.%I()
        $sql$,
        trigger_name || '_truncate',
        source_schema, source_table,
        queue_schema, trigger_name
    ) INTO STRICT _sql;

    EXECUTE _sql;
END;
$func$ LANGUAGE plpgsql VOLATILE;

-------------------------------------------------------------------------------
-- create_vectorizer: main entry point
-------------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION ai.create_vectorizer(
    source           regclass,
    loading          jsonb DEFAULT NULL,
    embedding        jsonb DEFAULT NULL,
    chunking         jsonb DEFAULT ai.chunking_recursive_character_text_splitter(),
    formatting       jsonb DEFAULT ai.formatting_python_template(),
    processing       jsonb DEFAULT ai.processing_default(),
    destination      jsonb DEFAULT ai.destination_table(),
    queue_schema     name DEFAULT NULL,
    queue_table      name DEFAULT NULL,
    enqueue_existing bool DEFAULT true,
    name             text DEFAULT NULL,
    if_not_exists    bool DEFAULT false
) RETURNS int4 AS $func$
DECLARE
    _source_table    name;
    _source_schema   name;
    _dimensions      int4;
    _source_pk       jsonb;
    _vectorizer_id   int4;
    _existing_id     int4;
    _trigger_name    name;
    _queue_failed_table name;
    _target_schema   name;
    _target_table    name;
    _view_schema     name;
    _view_name       name;
    _sql             text;
BEGIN
    -- Validate required params
    IF embedding IS NULL THEN
        RAISE EXCEPTION 'embedding configuration is required';
    END IF;
    IF loading IS NULL THEN
        RAISE EXCEPTION 'loading configuration is required';
    END IF;

    -- Resolve source table
    SELECT k.relname, n.nspname
    INTO STRICT _source_table, _source_schema
    FROM pg_class k
    INNER JOIN pg_namespace n ON k.relnamespace = n.oid
    WHERE k.oid = source;

    -- Extract dimensions from embedding config
    _dimensions = (embedding ->> 'dimensions')::int4;
    IF _dimensions IS NULL THEN
        RAISE EXCEPTION 'dimensions argument is required in embedding config';
    END IF;

    -- Get source PK
    SELECT ai._vectorizer_source_pk(source) INTO STRICT _source_pk;
    IF _source_pk IS NULL OR jsonb_array_length(_source_pk) = 0 THEN
        RAISE EXCEPTION 'source table must have a primary key constraint';
    END IF;

    -- Allocate ID and derive names
    _vectorizer_id = nextval('ai.vectorizer_id_seq');
    _trigger_name = concat('_vectorizer_src_trg_', _vectorizer_id);
    queue_schema = coalesce(queue_schema, 'ai');
    queue_table = coalesce(queue_table, concat('_vectorizer_q_', _vectorizer_id));
    _queue_failed_table = concat('_vectorizer_q_failed_', _vectorizer_id);

    -- Resolve destination defaults
    _target_schema = coalesce(destination ->> 'target_schema', _source_schema);
    _target_table = coalesce(
        destination ->> 'target_table',
        concat(_source_table, '_embedding_store')
    );
    _view_schema = coalesce(destination ->> 'view_schema', _source_schema);
    _view_name = coalesce(
        destination ->> 'view_name',
        concat(_source_table, '_embedding')
    );

    -- Update destination with resolved values
    destination = json_build_object(
        'implementation', 'table',
        'config_type', 'destination',
        'target_schema', _target_schema,
        'target_table', _target_table,
        'view_schema', _view_schema,
        'view_name', _view_name
    )::jsonb;

    -- Resolve name
    IF name IS NULL THEN
        name = format('%s_%s', _target_schema, _target_table);
    END IF;

    -- Check for existing vectorizer with same name
    SELECT id FROM ai.vectorizer
    WHERE ai.vectorizer.name = create_vectorizer.name
    INTO _existing_id;

    IF _existing_id IS NOT NULL THEN
        IF NOT if_not_exists THEN
            RAISE EXCEPTION 'a vectorizer named % already exists', name
            USING ERRCODE = 'duplicate_object';
        END IF;
        RAISE NOTICE 'a vectorizer named % already exists, skipping', name;
        RETURN _existing_id;
    END IF;

    -- Create destination table + view (IF NOT EXISTS for idempotency)
    PERFORM ai._vectorizer_create_target_table(
        _source_pk, _target_schema, _target_table, _dimensions
    );
    PERFORM ai._vectorizer_create_view(
        _view_schema, _view_name,
        _source_schema, _source_table,
        _source_pk,
        _target_schema, _target_table
    );

    -- Create queue tables
    PERFORM ai._vectorizer_create_queue_table(
        queue_schema, queue_table, _source_pk
    );
    PERFORM ai._vectorizer_create_queue_failed_table(
        queue_schema, _queue_failed_table, _source_pk
    );

    -- Create source trigger
    PERFORM ai._vectorizer_create_source_trigger(
        _trigger_name,
        queue_schema, queue_table,
        _source_schema, _source_table,
        _target_schema, _target_table,
        _source_pk
    );

    -- Insert vectorizer record
    INSERT INTO ai.vectorizer (
        id, source_schema, source_table, source_pk, trigger_name,
        queue_schema, queue_table, queue_failed_table, config, name
    ) VALUES (
        _vectorizer_id,
        _source_schema, _source_table, _source_pk, _trigger_name,
        queue_schema, queue_table, _queue_failed_table,
        jsonb_build_object(
            'version', '0.1.0',
            'loading', loading,
            'embedding', embedding,
            'chunking', chunking,
            'formatting', formatting,
            'processing', processing,
            'destination', destination
        ),
        create_vectorizer.name
    );

    -- Enqueue existing rows
    IF enqueue_existing THEN
        SELECT format(
            $sql$
            INSERT INTO %I.%I (%s)
            SELECT %s FROM %I.%I x
            $sql$,
            queue_schema, queue_table,
            (
                SELECT string_agg(format('%I', x.attname), ', ' ORDER BY x.attnum)
                FROM jsonb_to_recordset(_source_pk) x(attnum int, attname name)
            ),
            (
                SELECT string_agg(format('x.%I', x.attname), ', ' ORDER BY x.attnum)
                FROM jsonb_to_recordset(_source_pk) x(attnum int, attname name)
            ),
            _source_schema, _source_table
        ) INTO STRICT _sql;

        EXECUTE _sql;
    END IF;

    RETURN _vectorizer_id;
END;
$func$ LANGUAGE plpgsql VOLATILE;

-------------------------------------------------------------------------------
-- drop_vectorizer: cleanup
-------------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION ai.drop_vectorizer(
    vectorizer_id int4,
    drop_all bool DEFAULT true
) RETURNS void AS $func$
DECLARE
    _rec record;
    _target_schema name;
    _target_table name;
    _view_schema name;
    _view_name name;
BEGIN
    SELECT * INTO _rec FROM ai.vectorizer WHERE id = vectorizer_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'vectorizer % not found', vectorizer_id;
    END IF;

    -- Drop source triggers
    EXECUTE format(
        'DROP TRIGGER IF EXISTS %I ON %I.%I',
        _rec.trigger_name, _rec.source_schema, _rec.source_table
    );
    EXECUTE format(
        'DROP TRIGGER IF EXISTS %I ON %I.%I',
        _rec.trigger_name || '_truncate', _rec.source_schema, _rec.source_table
    );

    -- Drop trigger function
    EXECUTE format(
        'DROP FUNCTION IF EXISTS %I.%I()',
        _rec.queue_schema, _rec.trigger_name
    );

    IF drop_all THEN
        -- Drop view
        _view_schema = _rec.config -> 'destination' ->> 'view_schema';
        _view_name = _rec.config -> 'destination' ->> 'view_name';
        IF _view_schema IS NOT NULL AND _view_name IS NOT NULL THEN
            EXECUTE format('DROP VIEW IF EXISTS %I.%I', _view_schema, _view_name);
        END IF;

        -- Drop target table
        _target_schema = _rec.config -> 'destination' ->> 'target_schema';
        _target_table = _rec.config -> 'destination' ->> 'target_table';
        IF _target_schema IS NOT NULL AND _target_table IS NOT NULL THEN
            EXECUTE format('DROP TABLE IF EXISTS %I.%I', _target_schema, _target_table);
        END IF;

        -- Drop queue tables
        IF _rec.queue_schema IS NOT NULL AND _rec.queue_table IS NOT NULL THEN
            EXECUTE format('DROP TABLE IF EXISTS %I.%I', _rec.queue_schema, _rec.queue_table);
        END IF;
        IF _rec.queue_schema IS NOT NULL AND _rec.queue_failed_table IS NOT NULL THEN
            EXECUTE format('DROP TABLE IF EXISTS %I.%I', _rec.queue_schema, _rec.queue_failed_table);
        END IF;
    END IF;

    -- Remove vectorizer record (cascades to vectorizer_errors)
    DELETE FROM ai.vectorizer WHERE id = vectorizer_id;
END;
$func$ LANGUAGE plpgsql VOLATILE;

-------------------------------------------------------------------------------
-- Worker tracking tables
-------------------------------------------------------------------------------

-- One row per worker process (alive or dead)
CREATE TABLE IF NOT EXISTS ai.vectorizer_worker_process (
    id                          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    version                     text NOT NULL,
    started                     timestamptz NOT NULL DEFAULT now(),
    expected_heartbeat_interval interval NOT NULL,
    last_heartbeat              timestamptz NOT NULL DEFAULT now(),
    heartbeat_count             bigint NOT NULL DEFAULT 0,
    success_count               bigint NOT NULL DEFAULT 0,
    error_count                 bigint NOT NULL DEFAULT 0,
    last_error_at               timestamptz,
    last_error_message          text
);

CREATE INDEX IF NOT EXISTS idx_vectorizer_worker_process_last_heartbeat
    ON ai.vectorizer_worker_process (last_heartbeat);

-- Per-vectorizer progress (one row per vectorizer, shared across workers)
CREATE TABLE IF NOT EXISTS ai.vectorizer_worker_progress (
    vectorizer_id          int4 PRIMARY KEY NOT NULL
                           REFERENCES ai.vectorizer(id) ON DELETE CASCADE,
    success_count          bigint NOT NULL DEFAULT 0,
    error_count            bigint NOT NULL DEFAULT 0,
    last_success_at        timestamptz,
    last_success_process_id uuid,
    last_error_at          timestamptz,
    last_error_message     text,
    last_error_process_id  uuid,
    updated_at             timestamptz NOT NULL DEFAULT now()
);

-------------------------------------------------------------------------------
-- Worker tracking functions
-------------------------------------------------------------------------------

-- Register worker, return UUID
CREATE OR REPLACE FUNCTION ai._worker_start(
    version text,
    expected_heartbeat_interval interval
) RETURNS uuid AS $$
DECLARE
    worker_id uuid;
BEGIN
    INSERT INTO ai.vectorizer_worker_process (version, expected_heartbeat_interval)
    VALUES (version, expected_heartbeat_interval)
    RETURNING id INTO worker_id;
    RETURN worker_id;
END;
$$ LANGUAGE plpgsql SECURITY INVOKER
SET search_path TO pg_catalog, pg_temp;

-- Heartbeat: bump timestamp, increment heartbeat_count, accumulate deltas
CREATE OR REPLACE FUNCTION ai._worker_heartbeat(
    worker_id uuid,
    num_successes_since_last_heartbeat bigint,
    num_errors_since_last_heartbeat bigint,
    error_message text
) RETURNS void AS $$
DECLARE
    heartbeat_timestamp timestamptz = clock_timestamp();
BEGIN
    UPDATE ai.vectorizer_worker_process SET
        last_heartbeat = heartbeat_timestamp,
        heartbeat_count = heartbeat_count + 1,
        success_count = success_count + num_successes_since_last_heartbeat,
        error_count = error_count + num_errors_since_last_heartbeat,
        last_error_message = CASE WHEN error_message IS NOT NULL
            THEN error_message ELSE last_error_message END,
        last_error_at = CASE WHEN error_message IS NOT NULL
            THEN heartbeat_timestamp ELSE last_error_at END
    WHERE id = worker_id;
END;
$$ LANGUAGE plpgsql SECURITY INVOKER
SET search_path TO pg_catalog, pg_temp;

-- Per-vectorizer upsert: error_message NULL = success path, non-NULL = error path
CREATE OR REPLACE FUNCTION ai._worker_progress(
    worker_id uuid,
    worker_vectorizer_id int4,
    num_successes bigint,
    error_message text
) RETURNS void AS $$
DECLARE
    progress_timestamp timestamptz = clock_timestamp();
BEGIN
    INSERT INTO ai.vectorizer_worker_progress (
        vectorizer_id, success_count, error_count,
        last_success_at, last_success_process_id,
        last_error_at, last_error_message, last_error_process_id,
        updated_at
    ) VALUES (
        worker_vectorizer_id, num_successes,
        CASE WHEN error_message IS NULL THEN 0 ELSE 1 END,
        CASE WHEN error_message IS NULL THEN progress_timestamp END,
        CASE WHEN error_message IS NULL THEN worker_id END,
        CASE WHEN error_message IS NOT NULL THEN progress_timestamp END,
        error_message,
        CASE WHEN error_message IS NOT NULL THEN worker_id END,
        progress_timestamp
    )
    ON CONFLICT (vectorizer_id) DO UPDATE SET
        success_count = ai.vectorizer_worker_progress.success_count + EXCLUDED.success_count,
        error_count = ai.vectorizer_worker_progress.error_count + EXCLUDED.error_count,
        last_success_at = COALESCE(EXCLUDED.last_success_at,
            ai.vectorizer_worker_progress.last_success_at),
        last_success_process_id = COALESCE(EXCLUDED.last_success_process_id,
            ai.vectorizer_worker_progress.last_success_process_id),
        last_error_at = COALESCE(EXCLUDED.last_error_at,
            ai.vectorizer_worker_progress.last_error_at),
        last_error_message = COALESCE(EXCLUDED.last_error_message,
            ai.vectorizer_worker_progress.last_error_message),
        last_error_process_id = COALESCE(EXCLUDED.last_error_process_id,
            ai.vectorizer_worker_progress.last_error_process_id),
        updated_at = progress_timestamp;
END;
$$ LANGUAGE plpgsql SECURITY INVOKER
SET search_path TO pg_catalog, pg_temp;

-------------------------------------------------------------------------------
-- vectorizer_queue_pending: check queue depth
-------------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION ai.vectorizer_queue_pending(
    vectorizer_id int4
) RETURNS int8 AS $func$
DECLARE
    _rec record;
    _count int8;
BEGIN
    SELECT queue_schema, queue_table
    INTO _rec
    FROM ai.vectorizer
    WHERE id = vectorizer_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'vectorizer % not found', vectorizer_id;
    END IF;

    EXECUTE format(
        'SELECT count(*) FROM %I.%I',
        _rec.queue_schema, _rec.queue_table
    ) INTO _count;

    RETURN _count;
END;
$func$ LANGUAGE plpgsql STABLE;
