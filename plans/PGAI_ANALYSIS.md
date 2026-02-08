# pgai: Practical Analysis

`pgai` transforms PostgreSQL into a high-performance engine for AI applications by bridging the gap between data and LLMs. It consists of a PostgreSQL extension and a Python library/worker.

## Key Practical Capabilities

### 1. "AI-Powered Indexes" (Vectorizers)
Instead of manually managing embeddings, you declare a **vectorizer** on a table.
*   **Automatic Sync:** When you insert or update data in your table, a background worker automatically detects the change, generates a new embedding via an LLM provider (OpenAI, Ollama, Anthropic, etc.), and updates the storage table.
*   **Resilience:** It handles retries, rate limits, and model failures automatically, so your main application doesn't crash if an AI API is down.

### 2. AI Directly in SQL
You can call LLMs directly within your SQL queries to process data at scale:
*   **Classification:** `SELECT ai.openai_classify('sentiment-model', comment) FROM feedback;`
*   **Summarization:** Generate summaries for thousands of rows with a single `UPDATE` or `SELECT` statement.
*   **Data Enrichment:** Extract entities or structured data from raw text columns using SQL.

### 3. RAG (Retrieval-Augmented Generation) in a Single Query
You can implement an entire RAG pipeline—searching for context and generating an answer—within a single SQL function. This keeps your logic close to the data and simplifies your application code.

### 4. Text-to-SQL (Semantic Catalog)
It helps build applications where users can ask questions in natural language.
*   **Metadata Management:** Automatically generates and stores descriptions of your database schema.
*   **Improved Accuracy:** Provides LLMs with context (schema, business facts, SQL examples) to generate accurate SQL queries from natural language questions.

### 5. Utility Functions
*   **Hugging Face Integration:** Load datasets directly from Hugging Face into PostgreSQL with one command.
*   **Chunking:** Built-in SQL functions to split long text into smaller chunks for better embedding performance.

## Conclusion
`pgai` allows you to build **production-ready AI applications** without leaving the PostgreSQL ecosystem or building complex external "MLOps" pipelines.
