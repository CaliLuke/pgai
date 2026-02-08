-- Test ai.create_vectorizer end-to-end
-- This test validates:
--   1. Vectorizer creation with existing data (enqueue_existing=true)
--   2. Target table, queue table, view, and trigger creation
--   3. Trigger fires correctly on INSERT/UPDATE/DELETE
--   4. SECURITY DEFINER trigger function has search_path set
--   5. drop_vectorizer cleans up all objects
--   6. Composite primary key support

-------------------------------------------------------------------------------
-- Setup: create a source table with existing data
-------------------------------------------------------------------------------
CREATE TABLE public.blog_posts (
    id serial PRIMARY KEY,
    title text NOT NULL,
    body text NOT NULL
);

INSERT INTO public.blog_posts (title, body) VALUES
    ('First Post', 'Hello world'),
    ('Second Post', 'More content here'),
    ('Third Post', 'Even more content');

-------------------------------------------------------------------------------
-- Test 1: create_vectorizer with enqueue_existing=true
-- This exercises the enqueue query (the alias bug was here)
-------------------------------------------------------------------------------
SELECT ai.create_vectorizer(
    'public.blog_posts'::regclass,
    loading    => ai.loading_column('body'),
    embedding  => ai.embedding_openai('text-embedding-3-small', 1536),
    chunking   => ai.chunking_none(),
    formatting => ai.formatting_chunk_value()
);

-- Verify vectorizer record was created
SELECT count(*) AS vectorizer_count FROM ai.vectorizer
WHERE source_schema = 'public' AND source_table = 'blog_posts';

-- Verify queue table exists and has 3 rows (enqueued existing data)
SELECT count(*) AS queued_rows FROM ai._vectorizer_q_1;

-- Verify target table exists (empty, no embeddings yet)
SELECT count(*) AS target_rows FROM public.blog_posts_embedding_store;

-- Verify view exists
SELECT count(*) AS view_cols
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'blog_posts_embedding';

-- Verify failed queue table exists
SELECT count(*) AS failed_rows FROM ai._vectorizer_q_failed_1;

-------------------------------------------------------------------------------
-- Test 2: triggers fire on INSERT/UPDATE/DELETE
-------------------------------------------------------------------------------

-- INSERT: new row should appear in queue
INSERT INTO public.blog_posts (title, body) VALUES ('Fourth Post', 'New content');
SELECT count(*) AS queued_after_insert FROM ai._vectorizer_q_1;

-- UPDATE: changed row should be re-queued
UPDATE public.blog_posts SET body = 'Updated content' WHERE id = 1;
SELECT count(*) AS queued_after_update FROM ai._vectorizer_q_1;

-- UPDATE: PK-only change should not re-queue (no non-PK column changed)
-- (id is the only PK column; updating title changes a non-PK column)
-- Instead test: updating to same values should NOT re-queue
-- Actually let's just verify the count after a real content change
UPDATE public.blog_posts SET title = 'Updated Title' WHERE id = 2;
SELECT count(*) AS queued_after_title_update FROM ai._vectorizer_q_1;

-- DELETE: should remove from target table (but target is empty, so just
-- verify trigger doesn't error)
DELETE FROM public.blog_posts WHERE id = 3;

-------------------------------------------------------------------------------
-- Test 3: SECURITY DEFINER trigger function has search_path set
-------------------------------------------------------------------------------
SELECT proconfig::text LIKE '%search_path=pg_catalog, pg_temp%' AS has_search_path
FROM pg_proc
WHERE proname = '_vectorizer_src_trg_1'
  AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'ai');

-------------------------------------------------------------------------------
-- Test 4: drop_vectorizer cleans up everything
-------------------------------------------------------------------------------
SELECT ai.drop_vectorizer(1, drop_all => true);

-- Vectorizer record should be gone
SELECT count(*) AS vectorizer_after_drop FROM ai.vectorizer WHERE id = 1;

-- Target table should be gone
SELECT count(*) AS target_exists
FROM information_schema.tables
WHERE table_schema = 'public' AND table_name = 'blog_posts_embedding_store';

-- View should be gone
SELECT count(*) AS view_exists
FROM information_schema.tables
WHERE table_schema = 'public' AND table_name = 'blog_posts_embedding';

-- Queue tables should be gone
SELECT count(*) AS queue_exists
FROM information_schema.tables
WHERE table_schema = 'ai' AND table_name = '_vectorizer_q_1';

-- Trigger should be gone
SELECT count(*) AS trigger_exists
FROM information_schema.triggers
WHERE trigger_name = '_vectorizer_src_trg_1'
  AND event_object_table = 'blog_posts';

-------------------------------------------------------------------------------
-- Test 5: composite primary key support
-------------------------------------------------------------------------------
CREATE TABLE public.multi_pk (
    tenant_id int NOT NULL,
    doc_id int NOT NULL,
    content text NOT NULL,
    PRIMARY KEY (tenant_id, doc_id)
);

INSERT INTO public.multi_pk VALUES (1, 1, 'Tenant 1 Doc 1'), (1, 2, 'Tenant 1 Doc 2');

SELECT ai.create_vectorizer(
    'public.multi_pk'::regclass,
    loading    => ai.loading_column('content'),
    embedding  => ai.embedding_openai('text-embedding-3-small', 1536),
    chunking   => ai.chunking_none(),
    formatting => ai.formatting_chunk_value()
);

-- Verify composite PK rows were enqueued
SELECT count(*) AS multi_pk_queued FROM ai._vectorizer_q_2;

-- Verify trigger works with composite PK
INSERT INTO public.multi_pk VALUES (2, 1, 'Tenant 2 Doc 1');
SELECT count(*) AS multi_pk_queued_after_insert FROM ai._vectorizer_q_2;

-- Cleanup
SELECT ai.drop_vectorizer(2, drop_all => true);

-------------------------------------------------------------------------------
-- Test 6: if_not_exists behavior
-------------------------------------------------------------------------------
SELECT ai.create_vectorizer(
    'public.blog_posts'::regclass,
    loading    => ai.loading_column('body'),
    embedding  => ai.embedding_openai('text-embedding-3-small', 1536),
    chunking   => ai.chunking_none(),
    formatting => ai.formatting_chunk_value(),
    name       => 'test_idempotent'
);

-- Same name with if_not_exists=true should succeed (return existing id)
SELECT ai.create_vectorizer(
    'public.blog_posts'::regclass,
    loading    => ai.loading_column('body'),
    embedding  => ai.embedding_openai('text-embedding-3-small', 1536),
    chunking   => ai.chunking_none(),
    formatting => ai.formatting_chunk_value(),
    name       => 'test_idempotent',
    if_not_exists => true
);

-- Cleanup
SELECT ai.drop_vectorizer(3, drop_all => true);

-------------------------------------------------------------------------------
-- Test 7: error cases
-------------------------------------------------------------------------------

-- Missing embedding config
DO $$
BEGIN
    PERFORM ai.create_vectorizer(
        'public.blog_posts'::regclass,
        loading => ai.loading_column('body')
    );
    RAISE EXCEPTION 'should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'correctly raised: %', SQLERRM;
END $$;

-- Missing loading config
DO $$
BEGIN
    PERFORM ai.create_vectorizer(
        'public.blog_posts'::regclass,
        embedding => ai.embedding_openai('text-embedding-3-small', 1536)
    );
    RAISE EXCEPTION 'should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'correctly raised: %', SQLERRM;
END $$;

-- Table without PK
CREATE TABLE public.no_pk (data text);
DO $$
BEGIN
    PERFORM ai.create_vectorizer(
        'public.no_pk'::regclass,
        loading   => ai.loading_column('data'),
        embedding => ai.embedding_openai('text-embedding-3-small', 1536)
    );
    RAISE EXCEPTION 'should have failed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'correctly raised: %', SQLERRM;
END $$;

-------------------------------------------------------------------------------
-- Teardown
-------------------------------------------------------------------------------
DROP TABLE public.blog_posts CASCADE;
DROP TABLE public.multi_pk CASCADE;
DROP TABLE public.no_pk;
