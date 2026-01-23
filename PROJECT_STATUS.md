# RAG System - Project Status

## Overview
A Retrieval-Augmented Generation (RAG) system using LangChain, OpenAI, and Qdrant.
**Goal**: Compete in the mtRAG benchmark (TACL 2025 - Multi-Turn Conversational RAG)

## Benchmark: mtRAG (TACL 2025)

**Paper**: "mtRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems"

### Key Findings from mtRAG Paper

| Issue | Impact | Current Status |
|-------|--------|----------------|
| **Query Rewriting** | R@5 drops from 0.89 → 0.47 without it. With rewriting: 0.52 (+11%) | ❌ Not implemented |
| **IDK Detection** | ~25% of questions are unanswerable. Models hallucinate badly | ❌ Not implemented |
| **Conversation Memory** | Required for multi-turn context tracking | ❌ Not implemented |
| **Retrieval k** | Paper uses k=5, we use k=4 | ⚠️ Needs update |
| **Better Retrieval** | Elser + BGE outperforms basic embedding search | ⚠️ Basic only |
| **Long-form Answers** | FANC evaluation: Faithfulness, Appropriateness, Naturalness, Completeness | ⚠️ Needs tuning |

### Critical Quote from Paper
> "Query rewriting is essential for multi-turn RAG. Without it, later turns with non-standalone
> questions (e.g., 'When did that happen?') suffer dramatic performance degradation."

### Unanswerable Questions
- ~25% of mtRAG questions are unanswerable from retrieved context
- Models must explicitly say "I don't know" rather than hallucinate
- Current system: Always generates a response (hallucination risk)

## Project Structure

```
project/
├── main.py                    # CLI entry point with menu
├── drop_qdrant.py             # Script to clear Qdrant collection
├── loaders/
│   └── document_loader.py     # JSONLLoader for JSONL datasets
├── vector_stores/
│   └── qdrant.py              # Qdrant client & embeddings setup
├── workflow/
│   ├── indexing.py            # load_doc(), store_doc()
│   └── generation.py          # RAG query with context injection
├── dataset/
│   ├── clapnq.jsonl/          # French Revolution docs
│   └── cloud.jsonl/           # IBM Cloud CDN docs
└── .env.example               # Environment variables template
```

## Features

### 1. Document Loading (`loaders/document_loader.py`)
- `JSONLLoader` class - loads JSONL files into LangChain Documents
- `load_dataset()` - load by dataset name

### 2. Indexing (`workflow/indexing.py`)
- `load_doc()` - loads all JSONL files from `dataset/` and splits them
  - Chunk size: 1000 chars, overlap: 200 chars
- `store_doc()` - stores documents in Qdrant with batch logging (batch_size=1000)

### 3. Vector Store (`vector_stores/qdrant.py`)
- Connects to Qdrant at `http://localhost:6333` (Docker)
- Uses `text-embedding-3-large` embeddings
- Collection: `rag_documents`
- `get_vector_store()` - returns QdrantVectorStore instance

### 4. Generation (`workflow/generation.py`)
- `prompt_with_context()` - injects retrieved docs into prompt
- `query(query, model)` - RAG query interface (retrieves k=4 docs)

### 5. CLI (`main.py`)
```
=== RAG System ===
1. Chat with model    # Interactive RAG chat
2. Process documents  # Index & store
3. Exit
```

## Environment Variables (.env)
```
OPENAI_API_KEY=sk-...
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=rag_documents
```

## Running

### Start Qdrant (Docker)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Run the app
```bash
uv run python main.py
```

### Reset data
```bash
uv run python drop_qdrant.py
```

## Current Stats
- **255,850** raw documents
- **409,470** chunks after splitting

## Dependencies
- langchain[openai] >= 1.2.6
- langchain-qdrant >= 1.1.0
- langchain-openai >= 1.1.7
- langchain-text-splitters >= 1.1.0
- python-dotenv >= 1.0.0
- Python >= 3.13

---

## Implementation Roadmap (mtRAG Preparation)

### Phase 1: Query Rewriting (HIGHEST PRIORITY)
**Why**: R@5 improves from 0.47 → 0.52 (+11%) on multi-turn queries

**Tasks**:
- [ ] Create `rewrite_query()` function using LLM
- [ ] Add `ConversationMemory` class to track history
- [ ] Update `query()` to pass history and use rewritten query for retrieval
- [ ] Log original vs rewritten queries for debugging

**Files to modify**:
- `workflow/generation.py` - Add rewrite logic, ConversationMemory class

### Phase 2: IDK Detection
**Why**: ~25% of mtRAG questions are unanswerable; prevents hallucination

**Tasks**:
- [ ] Add IDK judge prompt/classifier
- [ ] Check if answer is in retrieved context before generating
- [ ] Return "I don't know" instead of hallucinating

**Files to modify**:
- `workflow/generation.py` - Add IDK detection before response generation

### Phase 3: Increase Retrieval k
**Why**: mtRAG paper uses k=5, we currently use k=4

**Tasks**:
- [ ] Change `similarity_search(k=4)` to `similarity_search(k=5)`

**Files to modify**:
- `workflow/generation.py` - Update k parameter

### Phase 4: Better Retrieval (Optional)
**Why**: Elser + BGE hybrid search outperforms basic embedding search

**Tasks**:
- [ ] Evaluate hybrid retrieval (dense + sparse)
- [ ] Add re-ranking stage
- [ ] Compare BM25 vs Elser for sparse retrieval

### Phase 5: FANC Evaluation Tuning
**Why**: mtRAG evaluates on Faithfulness, Appropriateness, Naturalness, Completeness

**Tasks**:
- [ ] Update generation prompt for longer, more complete answers
- [ ] Add citations/references to retrieved context
- [ ] Evaluate on FANC metrics
