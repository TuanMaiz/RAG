# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Retrieval-Augmented Generation (RAG) system competing in the mtRAG benchmark (TACL 2025 - Multi-Turn Conversational RAG). Uses LangChain, OpenAI embeddings/LLM, and Qdrant vector store.

## Common Commands

**Run the application:**
```bash
uv run python main.py
```

**Start Qdrant (Docker):**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Reset/clear the vector store:**
```bash
uv run python drop_qdrant.py
```

## Environment

Create `.env` from `.env.example`:
- `OPENAI_API_KEY` - Required for embeddings and LLM
- `QDRANT_URL` - Default: `http://localhost:6333`
- `QDRANT_COLLECTION` - Default: `rag_documents`

## Architecture

### Data Flow
1. **Loading**: `JSONLLoader` reads JSONL files from `dataset/` into LangChain Documents
2. **Splitting**: `RecursiveCharacterTextSplitter` chunks (1000 chars, 200 overlap)
3. **Indexing**: Chunks stored in Qdrant via `store_doc()`
4. **Retrieval**: `similarity_search(k=4)` fetches relevant chunks
5. **Generation**: RAG agent uses `@dynamic_prompt` middleware to inject retrieved context

### Key Components

**`workflow/generation.py`**: Core RAG logic. The `prompt_with_context()` function uses LangChain's `@dynamic_prompt` decorator to intercept queries, retrieve docs, and inject them as system messages. The `query()` function creates a RAG agent and invokes it.

**`workflow/indexing.py`**: `load_doc()` recursively finds all `*.jsonl` files under `dataset/`, loads them, and splits. `store_doc()` batches writes (1000 docs) with progress logging.

**`vector_stores/qdrant.py`**: Singleton `client` and `embeddings` (OpenAI `text-embedding-3-large`). `get_vector_store()` auto-creates collection with COSINE distance if missing.

**`loaders/document_loader.py`**: `JSONLLoader` class parses JSONL files, expects a `text` key by default, stores all other keys as metadata.

**`main.py`**: CLI with chat mode (option 1) and document processing (option 2).

### mtRAG Benchmark Context

The system is being developed for the mtRAG benchmark. Key findings from the paper:
- Query rewriting is critical for multi-turn (R@5: 0.47 → 0.52)
- ~25% of questions are unanswerable (IDK detection needed)
- Paper uses k=5 for retrieval (current code uses k=4)
- Evaluation uses FANC metrics: Faithfulness, Appropriateness, Naturalness, Completeness

Current implementation roadmap is documented in `PROJECT_STATUS.md`.

## Current Stats

- 255,850 raw documents → 409,470 chunks after splitting
- Dataset: `clapnq.jsonl/` (French Revolution docs), `cloud.jsonl/` (IBM Cloud CDN)
