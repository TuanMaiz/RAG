# RAG System - Project Status

## Overview
A Retrieval-Augmented Generation (RAG) system using LangChain, OpenAI, and Qdrant.
**Goal**: Compete in the mtRAG benchmark (TACL 2025 - Multi-Turn Conversational RAG)

## Benchmark: mtRAG (TACL 2025)

**Paper**: "mtRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems"

### Key Findings from mtRAG Paper

| Issue | Impact | Current Status |
|-------|--------|----------------|
| **Query Rewriting** | R@5 drops from 0.89 → 0.47 without it. With rewriting: 0.52 (+11%) | ✅ **Implemented** |
| **IDK Detection** | ~25% of questions are unanswerable. Models hallucinate badly | ✅ **Implemented** |
| **Conversation Memory** | Required for multi-turn context tracking | ✅ **Implemented** |
| **Query Duplication** | Duplicating query improves retrieval weighting | ✅ **Implemented** |
| **Retrieval k** | Paper uses k=5 | ✅ **Updated to k=5** |
| **Long-form Answers** | FANC evaluation: Faithfulness, Appropriateness, Naturalness, Completeness | ⚠️ Needs tuning |

### Critical Quote from Paper
> "Query rewriting is essential for multi-turn RAG. Without it, later turns with non-standalone
> questions (e.g., 'When did that happen?') suffer dramatic performance degradation."

---

## Project Structure

```
project/
├── main.py                    # CLI entry point with menu
├── evaluation.py              # mtRAG evaluation script
├── format_checker.py          # mtRAG format validation
├── drop_qdrant.py             # Script to clear Qdrant collection
├── loaders/
│   └── document_loader.py     # JSONLLoader for JSONL datasets
├── vector_stores/
│   └── qdrant.py              # Qdrant client & embeddings setup
├── workflow/
│   ├── indexing.py            # load_doc(), store_doc()
│   ├── generation.py          # RAG with query rewriting + IDK detection
│   └── memory.py              # ConversationMemory class
├── evaluation_dataset/
│   ├── human/                 # mtRAG test questions
│   │   └── RAG.jsonl          # 842 tasks (205 IBM Cloud)
│   └── cloud.jsonl/           # IBM Cloud corpus (101MB)
├── predictions/
│   └── ibmcloud_predictions.jsonl  # Our predictions (205 tasks)
└── .env.example               # Environment variables template
```

---

## Features Implemented

### 1. Document Loading (`loaders/document_loader.py`)
- `JSONLLoader` class - loads JSONL files into LangChain Documents
- Handles `document_id`, `text`, `url`, `domain` fields

### 2. Indexing (`workflow/indexing.py`)
- `load_doc()` - loads all JSONL files and splits them
- Chunk size: 1000 chars, overlap: 200 chars
- `store_doc()` - stores documents in Qdrant with batch logging (batch_size=1000)

### 3. Vector Store (`vector_stores/qdrant.py`)
- Connects to Qdrant at `http://localhost:6333` (Docker)
- Uses `text-embedding-3-large` embeddings
- Collection: `rag_documents`

### 4. Generation (`workflow/generation.py`)
- ✅ **Query Rewriting**: LLM-based rewrite for multi-turn context
- ✅ **IDK Detection**: Score threshold (0.5) for unanswerable questions
- ✅ **Query Duplication**: `"query query"` for better retrieval
- ✅ **Conversation Memory**: Tracks full history, uses last 5 for rewriting
- ✅ **k=5 Retrieval**: Matches mtRAG paper setting

### 5. Memory (`workflow/memory.py`)
- `ConversationMemory` class with window_size=5
- Stores full conversation history
- `get_context()` formats history for rewrite prompt

### 6. Evaluation (`evaluation.py`)
- Reads mtRAG RAG.jsonl format
- Filters for IBM Cloud questions (matching indexed corpus)
- Generates predictions in mtRAG format
- Output includes `contexts` (with document_id, score, text) and `predictions`

---

## Environment Variables (.env)
```
OPENAI_API_KEY=sk-...
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=rag_documents
```

---

## Running

### Start Qdrant
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

### Interactive Chat
```bash
uv run python main.py
# Select 1. Chat with model
```

### Run Evaluation
```bash
uv run python evaluation.py \
  evaluation_dataset/human/RAG.jsonl \
  predictions/ibmcloud_predictions.jsonl
```

### Format Check (mtRAG)
```bash
python format_checker.py \
  --input_file evaluation_dataset/human/RAG.jsonl \
  --prediction_file predictions/ibmcloud_predictions.jsonl \
  --mode rag_taskc
```

---

## Current Stats

### Indexed Data
| Dataset | Documents | Chunks |
|---------|-----------|--------|
| IBM Cloud (mtRAG corpus) | 101 MB file | ~???,??? chunks |

### Evaluation Results
| Metric | Value |
|--------|-------|
| Tasks processed | 205 (IBM Cloud only) |
| Format | mtRAG compliant |
| Time taken | ~1.5 hours |

---

## Format Checker Issues (⚠️)

**Problem**: Input has 842 tasks, but we only predicted 205 (IBM Cloud subset).

```
Mismatch in number of instances: input=842, output=205
637 missing task_id(s)
```

**Root Cause**: RAG.jsonl contains questions from **multiple datasets**:
- IBM Cloud (ibmcld_xxx) → 205 tasks ✅
- Clapnq (French Revolution) → Not indexed ❌
- Govt, Fiqa → Not indexed ❌

**Solution Options**:
1. **Create subset input** - Evaluate only IBM Cloud tasks (current state)
2. **Index all corpora** - Download and index all mtRAG datasets for full 842-task evaluation

---

## Completed Phases ✅

### Phase 1: Query Rewriting ✅
- ✅ Created `rewrite_query()` function using gpt-4o-mini
- ✅ Added `ConversationMemory` class with window_size=5
- ✅ Added `should_rewrite()` LLM judge
- ✅ Updated retrieval to use rewritten query
- ✅ Debug logging: `[Query Rewritten]` or `[Query Standalone]`

### Phase 2: IDK Detection ✅
- ✅ Added score threshold (0.5) using `similarity_search_with_score()`
- ✅ IDK message: "The available documents don't contain information to answer..."
- ✅ Debug logging: `[IDK: max_score < threshold]`
- ✅ Saves IDK responses to memory for conversation continuity

### Phase 3: k=5 Retrieval ✅
- ✅ Updated from k=4 to k=5

---

## Current Work (2025-02-05)

**Focus**: Phase 6 - Full Evaluation & FANC Tuning

**Recent Progress:**
- ✅ Fixed `evaluation.py` to use Qdrant `query_points` API with named vectors
- ✅ Completed evaluation on Cloud dataset (205 tasks)
- ✅ Hybrid retrieval (dense + sparse BM25) fully operational

**Code Changes:**
- `evaluation.py`: Updated `query_points` usage:
  ```python
  results = client.query_points(
      collection_name=COLLECTION_NAME,
      query=dense_vector,  # list[float]
      using="dense",       # named vector selector
      limit=5,
  )
  # Access results via .points attribute
  for result in results.points: ...
  ```

---

## Remaining Tasks

### Phase 5: Better Retrieval ✅ COMPLETE
**Why**: Hybrid retrieval (dense + sparse) outperforms basic embedding search

**Completed:**
- [x] Analyze current retrieval implementation
- [x] Design hybrid retrieval approach
- [x] Implement sparse retrieval layer
- [x] Implement resume functionality for indexing
- [x] Complete indexing of all datasets (622,231 documents)
- [x] Fix indexing timeout issues
- [x] Fix evaluation script for hybrid retrieval

**Result:**
- All 4 datasets indexed: clapnq, cloud, fiqa, govt
- Hybrid collection with dense (1536 dims) + sparse (BM25) vectors
- Resume functionality working
- Evaluation script compatible with named vectors

### Next Steps

| Priority | Task | Description |
|----------|------|-------------|
| **High** | Run evaluation on full 842 tasks | All datasets now indexed |
| **High** | Compare hybrid vs dense-only metrics | Measure retrieval improvement |
| **Medium** | Tune FANC evaluation | Optimize for Faithfulness, Appropriateness, Naturalness, Completeness |

### Phase 6: FANC Evaluation Tuning 🔄 IN PROGRESS
**Why**: mtRAG evaluates on Faithfulness, Appropriateness, Naturalness, Completeness

**Tasks**:
- [x] Run evaluation.py on Cloud dataset (205 tasks)
- [ ] Run evaluation.py on full RAG.jsonl (all 842 tasks)
- [ ] Compare results with baseline
- [ ] Update generation prompt for longer, more complete answers
- [ ] Add citations/references to retrieved context

---

## Dependencies
- langchain[openai] >= 1.2.6
- langchain-qdrant >= 1.1.0
- langchain-openai >= 1.1.7
- langchain-text-splitters >= 1.1.0
- python-dotenv >= 1.0.0
- tqdm >= 4.65.0
- Python >= 3.13
