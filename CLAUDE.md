# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Retrieval-Augmented Generation (RAG) system competing in the mtRAG benchmark (TACL 2025 - Multi-Turn Conversational RAG). Uses LangChain, OpenRouter/OpenAI embeddings, and Qdrant vector store with **hybrid (dense + sparse) retrieval**.

## Common Commands

**Run the application:**
```bash
uv run python main.py
# Option 1: Chat mode | Option 2: Process/index documents
```

**Start Qdrant (Docker):**
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

**Reset/clear the vector store:**
```bash
uv run python drop_qdrant.py
```

**Run mtRAG evaluation:**
```bash
uv run python evaluation.py \
  evaluation_dataset/human/RAG.jsonl \
  predictions/predictions.jsonl
```

**Check Qdrant collection status:**
```bash
curl -s http://localhost:6333/collections/rag_documents | python -m json.tool
```

## Environment

Create `.env` from `.env.example`:
```
OPENAI_API_KEY=sk-or-v1-...          # OpenRouter or OpenAI key
OPENAI_BASE_URL=https://openrouter.ai/api/v1  # Or https://api.openai.com/v1
OPENAI_EMBEDDING_MODEL=openai/text-embedding-3-small
OPENAI_LLM_MODEL=openai/gpt-4o-mini
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_documents
```

**Note**: The system supports OpenRouter as a drop-in replacement for OpenAI. Set `OPENAI_BASE_URL` accordingly.

## Architecture

### Hybrid Retrieval Data Flow
```
1. JSONL files (dataset/) → JSONLLoader → LangChain Documents
2. Documents → RecursiveCharacterTextSplitter → Chunks (1000 chars, 200 overlap)
3. Chunks → [Dense Embeddings + Sparse BM25] → Qdrant (hybrid collection)
4. Query → [Rewrite (if needed)] → [Duplicate] → Hybrid Search (dense + sparse)
5. Retrieved docs → LLM → Response (or IDK message)
```

### Key Components

**`vector_stores/qdrant.py`**: Singleton Qdrant client and OpenAI embeddings. `create_hybrid_collection()` creates a collection with both dense (named "dense") and sparse (named "sparse") vectors. Sparse vectors use Qdrant's IDF modifier for BM25-style scoring.

**`workflow/hybrid_retrieval.py`**: Hybrid search implementation. `tokenize_bm25()` creates simple sparse vectors from text. `hybrid_search()` uses Qdrant's prefetch + fusion API to combine dense and sparse results via Reciprocal Rank Fusion (RRF).

**`workflow/indexing.py`**: `load_doc()` recursively finds `*.jsonl` files and splits them. `store_doc()` supports **resume functionality** - detects existing collection, continues from last indexed point, saves progress to `.indexing_progress.json`. Use `resume=False` to force reindex.

**`workflow/generation.py`**: Multi-turn RAG with:
- `should_rewrite()` - LLM judge for context-dependent queries
- `rewrite_query()` - Rewrites query to be standalone
- `duplicate_query()` - Returns `"query query"` for better retrieval
- `retrieve_with_scores()` - Uses hybrid retrieval via `hybrid_retrieve()`
- IDK detection with configurable score threshold (0.5)

**`workflow/memory.py`**: `ConversationMemory` class tracks conversation history with sliding window (default 5 turns).

**`loaders/document_loader.py`**: `JSONLLoader` parses JSONL files expecting a `text` key, stores all other keys as metadata.

### Hybrid Collection Schema

Qdrant collection uses named vectors:
- `dense`: OpenAI embeddings (1536 dims for text-embedding-3-small, 3072 for large)
- `sparse`: BM25-style sparse vectors with IDF modifier

Points are upserted with both vector types in a single call for hybrid search.

### mtRAG Benchmark Context

Key findings from the paper:
- **Query rewriting critical**: R@5 drops 0.89 → 0.47 without it
- **IDK detection**: ~25% of questions are unanswerable
- **k=5 retrieval**: Paper uses k=5 (system configured for this)
- **FANC metrics**: Faithfulness, Appropriateness, Naturalness, Completeness

Current implementation status in `PROJECT_STATUS.md`.

## Indexing Notes

- **Resume enabled by default**: `store_doc()` detects existing collection and continues
- **Progress file**: `.indexing_progress.json` saved after each batch
- **Batch size**: Default 350, adjustable via parameter
- **Auto-retry**: Embedding API failures trigger 10s wait + retry
- **Sparse vectors**: Computed locally via simple tokenization (no external model)

## Datasets

Located in `dataset/`:
- `clapnq.jsonl` (162 MB) - French Revolution
- `cloud.jsonl` (126 MB) - IBM Cloud CDN
- `fiqa.jsonl` (50 MB) - Financial QA
- `govt.jsonl` (108 MB) - Government documents

Total: ~622k chunks after splitting.
