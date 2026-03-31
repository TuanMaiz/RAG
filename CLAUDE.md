# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Retrieval-Augmented Generation (RAG) system competing in the mtRAG benchmark (TACL 2025 - Multi-Turn Conversational RAG). Uses LangChain, OpenRouter/OpenAI embeddings, Qdrant vector store with **hybrid (dense + sparse) retrieval**, and **Neo4j knowledge graph** for entity-based retrieval enhancement.

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

**Start Neo4j (Docker):**

```bash
docker run -d -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
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
# OpenAI / OpenRouter
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_EMBEDDING_MODEL=openai/text-embedding-3-small
OPENAI_LLM_MODEL=openai/gpt-4o-mini

# Qdrant Vector Store
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_documents

# Neo4j Knowledge Graph
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
ENABLE_KG=true

# Logging
LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
LOG_FILE=                   # Optional: path to log file
```

**Note**: The system supports OpenRouter as a drop-in replacement for OpenAI. Set `OPENAI_BASE_URL` accordingly.

## Architecture

### Complete RAG with Knowledge Graph Data Flow

```
1. JSONL files (dataset/) → JSONLLoader → LangChain Documents
2. Documents → RecursiveCharacterTextSplitter → Chunks (1000 chars, 200 overlap)

Indexing (parallel):
├─ Chunks → [Dense + Sparse vectors] → Qdrant (with text in payload)
└─ Chunks → LLM Entity Extraction → Neo4j (Entity nodes + MENTIONS relationships)

Retrieval:
├─ Query → [Rewrite] → [Duplicate] → Hybrid Search (dense + sparse) → Docs with TEXT
└─ Query → Entity Extraction → Graph Search (Neo4j) → Document IDs
                                          ↓
                                  Fetch TEXT from Qdrant
                                          ↓
                                  Context Fusion (merge + deduplicate)
                                          ↓
                                  Format with [1], [2] citations
                                          ↓
                                  LLM → Response + References section
```

### Key Design: KG as Index

- **Qdrant stores**: Vectors + **actual text in payload** (`page_content`)
- **Neo4j stores**: Entity → Document links (index only)
- **Citations work**: KG finds document_ids → Fetch text from Qdrant → LLM gets actual text

### Key Components

**`vector_stores/qdrant.py`**: Singleton Qdrant client and OpenAI embeddings. `create_hybrid_collection()` creates a collection with both dense (named "dense") and sparse (named "sparse") vectors. Sparse vectors use Qdrant's IDF modifier for BM25-style scoring.

**`workflow/hybrid_retrieval.py`**: Hybrid search implementation. `tokenize_bm25()` creates simple sparse vectors from text. `hybrid_search()` uses Qdrant's prefetch + fusion API to combine dense and sparse results via Reciprocal Rank Fusion (RRF).

**`workflow/indexing.py`**: `load_doc()` recursively finds `*.jsonl` files and splits them. `store_doc()` supports **resume functionality** - detects existing collection, continues from last indexed point, saves progress to `.indexing_progress.json`. Use `resume=False` to force reindex.

**`workflow/generation.py`**: Multi-turn RAG with:

- `should_rewrite()` - LLM judge for context-dependent queries
- `rewrite_query()` - Rewrites query to be standalone
- `duplicate_query()` - Returns `"query query"` for better retrieval
- `retrieve_with_scores()` - Uses hybrid retrieval via `hybrid_retrieve()`
- `retrieve_with_kg()` - Combines vector + graph search with context fusion
- `query()` - Main entry point, returns response with **automatic references section**
- IDK detection with configurable score threshold (0.5)
- Structured logging with box-drawing characters for readability

**`workflow/memory.py`**: `ConversationMemory` class tracks conversation history with sliding window (default 5 turns).

**`loaders/document_loader.py`**: `JSONLLoader` parses JSONL files expecting a `text` key, stores all other keys as metadata.

**`utils/logging_config.py`**: Centralized logging configuration with colored console output and optional file logging.

### Knowledge Graph Components

**`graph_stores/neo4j_client.py`**: Neo4j connection and basic CRUD operations.

- `get_driver()` - Get connection singleton
- `execute_query()` - Run Cypher queries
- `merge_node()` - Create/update nodes
- `create_relationship()` - Link nodes

**`workflow/kg_extraction.py`**: LLM-based entity and relationship extraction.

- `extract_entities()` - Extract entities from text
- `extract_relationships()` - Extract relationships between entities
- `extract_graph_data()` - Combined extraction
- `extract_query_entities()` - Extract entities from user queries

**`workflow/graph_retrieval.py`**: Graph-based document retrieval.

- `graph_search()` - Main search, returns document IDs
- `_find_documents_by_entity()` - Direct entity matching
- `_expand_entities()` - Relationship traversal for entity expansion

**`workflow/context_fusion.py`**: Merge vector and graph retrieval results.

- `fetch_texts_from_qdrant()` - Fetch actual text by document IDs
- `fuse_results()` - Merge and deduplicate vector + graph results
- `format_for_llm()` - Format with [1], [2] citation markers

### Hybrid Collection Schema

Qdrant collection uses named vectors:

- `dense`: OpenAI embeddings (1536 dims for text-embedding-3-small, 3072 for large)
- `sparse`: BM25-style sparse vectors with IDF modifier

Points are upserted with both vector types in a single call for hybrid search.

### Neo4j Schema

```cypher
// Entity node (LLM-extracted types)
(:Entity {
  name: "Napoleon Bonaparte",
  type: "PERSON",           // PERSON, ORG, LOC, EVENT, CONCEPT, PRODUCT, DATE
  domain: "clapnq"           // Dataset source: clapnq, cloud, fiqa, govt
})

// Document node (links to Qdrant point IDs)
(:Document {
  id: "123",                 // Matches Qdrant point ID
  domain: "clapnq"
})

// Relationships
(:Document)-[:MENTIONS]->(:Entity)
(:Entity)-[:RELATED_TO]->(:Entity)
```

### mtRAG Benchmark Context

Key findings from the paper:

- **Query rewriting critical**: R@5 drops 0.89 → 0.47 without it
- **IDK detection**: ~25% of questions are unanswerable
- **k=5 retrieval**: Paper uses k=5 (system configured for this)
- **FANC metrics**: Faithfulness, Appropriateness, Naturalness, Completeness

Current implementation status in `PROJECT_STATUS.md`.

## Indexing Notes

### Standard Indexer (`workflow/indexing.py`)

- **Resume enabled by default**: `store_doc()` detects existing collection and continues
- **Progress file**: `.indexing_progress.json` saved after each batch
- **Batch size**: Default 350, adjustable via parameter
- **Auto-retry**: Embedding API failures trigger 10s wait + retry
- **Sparse vectors**: Computed locally via simple tokenization (no external model)
- **KG indexing**: When `ENABLE_KG=true`, extracts entities via LLM and stores to Neo4j during indexing
- **Graceful degradation**: If Neo4j is down, Qdrant indexing continues without error

### Streaming Indexer (`workflow/threading_indexing.py`)

**Experimental** - Significantly faster indexing through parallelism and optimized batch sizing:

- **Optimized LLM batch size**: Calculated from model context window (default: ~500 docs/call vs 10)
- **Parallel extract workers**: Configurable thread pool (default: 8 workers)
- **Memory-based backpressure**: Spill-to-disk queue (90% spill, 60% restore hysteresis)
- **Dual progress bars**: Separate bars for extraction and upsert
- **Retry logic**: Exponential backoff for Neo4j/Qdrant failures (3 attempts)

**Usage:**
```bash
uv run python main.py
# Choose option 2 for streaming mode
```

**Configuration:**
```bash
# Memory Management
INDEXING_MAX_MEMORY_GB=4        # Max queue memory before spill
SPILL_THRESHOLD_PCT=90         # Spill at 90% capacity
RESTORE_THRESHOLD_PCT=60       # Restore at 60% capacity
SPILL_PATH=./indexing_spill    # Shelve file location

# Deduplication
GLOBAL_DEDUP_INTERVAL=10       # Batches between global dedup

# Optimized Batch Size (auto-calculated from model context)
KG_BATCH_SIZE=500              # Docs per LLM call
MODEL_TOKEN_LIMIT=128000       # Context window (GPT-4o-mini: 128K)
AVG_DOC_CHARS=500              # Average chunk size
SAFETY_MARGIN=0.8              # Use 80% of max tokens
```

**Performance:** 600k docs from ~500 hours (sequential) to ~2 hours (streaming)

## Datasets

Located in `dataset/`:

- `clapnq.jsonl` (162 MB) - French Revolution
- `cloud.jsonl` (126 MB) - IBM Cloud CDN
- `fiqa.jsonl` (50 MB) - Financial QA
- `govt.jsonl` (108 MB) - Government documents

Total: ~622k chunks after splitting.

## Logging

The system uses structured logging with colored console output (`utils/logging_config.py`).

**Log levels**: Set `LOG_LEVEL` in `.env` (default: INFO)

- `INFO` - Clean retrieval summaries during chat
- `DEBUG` - Full context dumps, query details, entity extraction

**Example output (INFO level)**:

```
└─ Mode: Vector + Graph search
└─ Retrieved: 10 docs | Max score: 0.842
```

**Example output (DEBUG level)**:

```
├─ Query: "Who was Napoleon?" (standalone)
├─ Graph: found 5 docs
┌─ Context provided to LLM (10 docs)
│ [1] Napoleon Bonaparte was born on August 15, 1769...
│ [2] The French Revolution began in 1789...
└─ End context
```

**Response format**: Answers include automatic references section with full chunk text for all citations used.

## Testing

**Quick KG integration test:**

```bash
uv run python test_kg.py
```

**Run KG comparison evaluation:**

```bash
uv run python compare_kg_evaluation.py
# Select dataset to compare with/without KG
```

**Check Neo4j data:**

```bash
# Count entities
uv run python -c "
from graph_stores.neo4j_client import execute_query
result = execute_query('MATCH (e:Entity) RETURN count(e) as count')
print(f'Entities: {result[0][\"count\"]}')
"
```

**Test individual components:**

```bash
# Entity extraction
uv run python -c "
from workflow.kg_extraction import extract_graph_data
print(extract_graph_data('Napoleon led the French army.', 'clapnq'))
"

# Graph retrieval
uv run python -c "
from workflow.graph_retrieval import graph_search
print(graph_search('Who was Napoleon?', domain='clapnq', k=5))
"

# Context fusion
uv run python -c "
from workflow.context_fusion import fetch_texts_from_qdrant
print(fetch_texts_from_qdrant([1]))
"
```

## TODO:

- Implement: Towards Practical GraphRAG: Efficient Knowledge Graph Construction and Hybrid Retrieval at Scale as module, has env to switch to this beside LLM extract approach
