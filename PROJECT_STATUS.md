# RAG System - Project Status

## Overview
A Retrieval-Augmented Generation (RAG) system using LangChain, OpenAI, Qdrant, and **Neo4j Knowledge Graph**.
**Goal**: Compete in the mtRAG benchmark (TACL 2025 - Multi-Turn Conversational RAG)

## Knowledge Graph Progress

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Neo4j Foundation - Client setup, basic CRUD |
| Phase 2 | ✅ Complete | Entity Extraction - LLM-based extraction |
| Phase 3 | ✅ Complete | Graph Storage - Store entities during indexing |
| Phase 4 | ✅ Complete | Graph Retrieval - Query by entity, expand relationships |
| Phase 5 | ✅ Complete | Context Fusion - Merge vector + graph results |
| Phase 6 | ✅ Complete | Integration - KG integrated into generation.py and evaluation.py |
| Phase 7 | ✅ Complete | Testing & Evaluation - Test scripts, comparison tools |

**🎉 ALL 7 PHASES COMPLETE!**

---

## Benchmark: mtRAG (TACL 2025)

**Paper**: "mtRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems"

### Key Findings from mtRAG Paper

| Issue | Impact | Current Status |
|-------|--------|----------------|
| **Query Rewriting** | R@5 drops from 0.89 → 0.47 without it. With rewriting: 0.52 (+11%) | ✅ **Implemented** |
| **IDK Detection** | ~25% of questions are unanswerable. Models hallucinate badly | ✅ **Implemented** |
| **Conversation Memory** | Required for multi-turn context tracking | ✅ **Implemented** |
| **Query Duplication** | Duplicating query improves retrieval weighting | ✅ **Implemented** |
| **Hybrid Retrieval** | Dense + sparse (BM25) with RRF fusion | ✅ **Implemented** |
| **Retrieval k** | Paper uses k=5 | ✅ **Updated to k=5** |
| **Long-form Answers** | FANC evaluation: Faithfulness, Appropriateness, Naturalness, Completeness | ✅ **Tuned** |

---

## Project Structure

```
project/
├── main.py                      # CLI entry point with menu
├── evaluation.py                # mtRAG evaluation script
├── format_checker.py            # mtRAG format validation
├── drop_qdrant.py               # Script to clear Qdrant collection
├── test_kg.py                   # NEW: Quick KG integration test
├── compare_kg_evaluation.py     # NEW: KG comparison evaluation
├── loaders/
│   └── document_loader.py        # JSONLLoader for JSONL datasets
├── vector_stores/
│   └── qdrant.py                 # Qdrant client & embeddings setup
├── graph_stores/                 # NEW: Knowledge graph storage
│   └── neo4j_client.py           # Neo4j client (basic CRUD)
├── workflow/
│   ├── indexing.py              # load_doc(), store_doc() + KG indexing
│   ├── generation.py            # RAG with query rewriting + KG retrieval
│   ├── memory.py                # ConversationMemory class
│   ├── prompts.py               # Modularized system prompts + KG extraction
│   ├── hybrid_retrieval.py      # Dense + sparse BM25 with RRF
│   ├── kg_extraction.py         # NEW: LLM-based entity/relationship extraction
│   ├── graph_retrieval.py       # NEW: Graph search by entity
│   └── context_fusion.py        # NEW: Merge vector + graph results
├── evaluation_dataset/
│   └── human/
│       └── RAG.jsonl            # 842 tasks (all datasets)
├── dataset/                      # Training data
│   ├── clapnq.jsonl             # French Revolution (162 MB)
│   ├── cloud.jsonl              # IBM Cloud CDN (126 MB)
│   ├── fiqa.jsonl               # Financial QA (50 MB)
│   └── govt.jsonl               # Government documents (108 MB)
├── predictions/
│   └── predictions.jsonl        # Our predictions
└── .env.example                 # Environment variables template
```

---

## Knowledge Graph Integration (NEW) 🚧

### Architecture
```
Query → Hybrid Search (Qdrant) → Chunks with TEXT
     ↓
     AND
     ↓
     → Graph Search (Neo4j) → Entity → Document IDs → Fetch TEXT from Qdrant
     ↓
     Context Fusion → LLM → Response with citations [1], [2]...
```

**Key Design**: KG acts as an **index** - stores entity → document links. Actual text stays in Qdrant payload.

### Completed Phases

#### Phase 1: Neo4j Foundation ✅
- [x] Install neo4j Python driver (neo4j>=5.0.0)
- [x] Create `graph_stores/neo4j_client.py` with basic CRUD operations
- [x] Set up Neo4j with Docker
- [x] Test connection and operations

**Files Created:**
- `graph_stores/__init__.py`
- `graph_stores/neo4j_client.py`

**Available Functions:**
```python
from graph_stores.neo4j_client import get_driver, execute_query, merge_node, create_relationship

# Execute any Cypher query
results = execute_query("MATCH (n) RETURN n LIMIT 10")

# Create/merge nodes
doc = merge_node("Document", {"id": "doc_1", "title": "My Doc"})
entity = merge_node("Entity", {"name": "Napoleon", "type": "PERSON", "domain": "clapnq"})

# Create relationships
create_relationship("Document", {"id": "doc_1"}, "Entity", {"name": "Napoleon"}, "MENTIONS")
```

#### Phase 2: Entity Extraction ✅
- [x] Add KG extraction prompts to `workflow/prompts.py`
- [x] Create `workflow/kg_extraction.py` with LLM-based extraction
- [x] Test extraction on sample documents

**Files Modified:**
- `workflow/prompts.py` - Added ENTITY_EXTRACTION_PROMPT, RELATIONSHIP_EXTRACTION_PROMPT, QUERY_ENTITY_PROMPT

**Files Created:**
- `workflow/kg_extraction.py`

**Extraction Example:**
```python
from workflow.kg_extraction import extract_entities, extract_graph_data

# Extract entities
entities = extract_entities("Napoleon led the French army.", domain="clapnq")
# Returns: [{"name": "Napoleon", "type": "PERSON", "domain": "clapnq"}, ...]

# Extract both entities and relationships
data = extract_graph_data(text, domain="clapnq")
# Returns: {"entities": [...], "relationships": [...]}
```

#### Phase 3: Graph Storage ✅
- [x] Add `extract_domain()` to extract domain from file paths
- [x] Add `store_entities_to_graph()` to indexing pipeline
- [x] Integrate KG storage into `store_doc()` function
- [x] Add `ENABLE_KG` environment variable flag

**Files Modified:**
- `workflow/indexing.py` - Added KG storage during indexing

**Indexing Flow with KG:**
```python
# During indexing (workflow/indexing.py)
if ENABLE_KG:
    domain = extract_domain(doc.metadata.get("source", ""))
    store_entities_to_graph(doc_id, doc.page_content, domain)
```

#### Phase 4: Graph Retrieval ✅
- [x] Create `workflow/graph_retrieval.py`
- [x] Implement `graph_search()` - Main search function
- [x] Implement `_find_documents_by_entity()` - Direct entity matching
- [x] Implement `_expand_entities()` - Relationship traversal
- [x] Implement `get_entity_context()` - Entity details with related entities
- [x] Add domain inference from entity keywords

**Files Created:**
- `workflow/graph_retrieval.py`

**Retrieval Example:**
```python
from workflow.graph_retrieval import graph_search

# Search for relevant documents
doc_ids = graph_search(
    query="Tell me about the French Revolution",
    domain="clapnq",  # Optional, auto-inferred if None
    k=5,
    expand=True  # Use relationship traversal
)
# Returns: [1, 45, 123, ...] - document IDs for Qdrant lookup
```

#### Phase 5: Context Fusion ✅
- [x] Create `workflow/context_fusion.py`
- [x] Implement `fetch_texts_from_qdrant()` - Fetch actual text by document IDs
- [x] Implement `fuse_results()` - Merge and deduplicate vector + graph results
- [x] Implement `fuse_with_scores()` - Score-based fusion with re-ranking
- [x] Implement `format_for_llm()` - Format with [1], [2] citations

**Files Created:**
- `workflow/context_fusion.py`

**Fusion Example:**
```python
from workflow.context_fusion import fuse_results, fetch_texts_from_qdrant
from workflow.graph_retrieval import graph_search

# Get vector docs (from hybrid search)
vector_docs = hybrid_search(query, k=3)

# Get graph doc IDs
graph_ids = graph_search(query, domain="clapnq", k=3)

# Fetch actual text from Qdrant for graph results
graph_docs = fetch_texts_from_qdrant(graph_ids)

# Merge and deduplicate
fused = fuse_results(vector_docs, graph_ids, k=5)

# Format for LLM with citations
formatted = format_for_llm(fused)
# "[1] Doc 1 content...\n\n[2] Doc 2 content..."
```

#### Phase 6: Integration ✅
- [x] Add `retrieve_with_kg()` to `workflow/generation.py`
- [x] Update `query()` function to use KG retrieval when enabled
- [x] Update `evaluation.py` to use KG-enhanced retrieval
- [x] Add graceful fallback when KG disabled

**Files Modified:**
- `workflow/generation.py` - Added KG retrieval integration
- `evaluation.py` - Updated to use KG retrieval
- `workflow/hybrid_retrieval.py` - Fixed FusionQuery weights bug

**Integration Flow:**
```python
# In workflow/generation.py:
if ENABLE_KG:
    docs, max_score = retrieve_with_kg(query, history, k=5)
    # Uses both vector and graph search, fuses results
else:
    docs, max_score = retrieve_with_scores(query, history, k=5)
    # Uses hybrid vector search only
```

#### Phase 7: Testing & Evaluation ✅
- [x] Create `test_kg.py` - Quick integration test
- [x] Create `compare_kg_evaluation.py` - Side-by-side comparison tool
- [x] Test KG toggle (ENABLE_KG=true/false)
- [x] Verify graceful fallback

**Files Created:**
- `test_kg.py` - Quick integration test
- `compare_kg_evaluation.py` - Full comparison evaluation script

**Testing Commands:**
```bash
# Quick integration test
uv run python test_kg.py

# Full comparison evaluation
uv run python compare_kg_evaluation.py
```

**Comparison Metrics:**
- Same vs Different responses
- Answerability (who answered more questions)
- Average contexts retrieved
- Processing time comparison

### Remaining KG Phases

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 3 | ✅ DONE | Graph Storage - Store extracted entities in Neo4j, link to documents |
| Phase 4 | ✅ DONE | Graph Retrieval - Query KG to find relevant documents |
| Phase 5 | ✅ DONE | Context Fusion - Merge vector + graph results, fetch text from Qdrant |
| Phase 6 | ⏳ TODO | Integration - Add KG to indexing and generation pipelines |
| Phase 7 | ⏳ TODO | Testing - End-to-end testing and evaluation |

---

## Features Implemented

### 1. Document Loading (`loaders/document_loader.py`)
- `JSONLLoader` class - loads JSONL files into LangChain Documents
- Handles `document_id`, `text`, `url`, `domain` fields

### 2. Indexing (`workflow/indexing.py`)
- `load_doc()` - loads all JSONL files and splits them
- Chunk size: 1000 chars, overlap: 200 chars
- `store_doc()` - stores documents in Qdrant with batch processing
- **Resume functionality** - continues from last indexed point

### 3. Vector Store (`vector_stores/qdrant.py`)
- Connects to Qdrant at `http://localhost:6333` (Docker)
- Uses `text-embedding-3-small` embeddings (1536 dims)
- Hybrid collection: dense + sparse (BM25 with IDF)
- RRF weights: dense=0.7, sparse=0.3

### 4. Hybrid Retrieval (`workflow/hybrid_retrieval.py`)
- ✅ **Dense + Sparse BM25** with RRF fusion
- ✅ **Weights**: [0.7, 0.3] for dense vs sparse
- ✅ **k=5** retrieval

### 5. Generation (`workflow/generation.py`)
- ✅ **Query Rewriting**: LLM-based rewrite for multi-turn context
- ✅ **IDK Detection**: Score threshold (0.5) for unanswerable questions
- ✅ **Query Duplication**: `"query query"` for better retrieval
- ✅ **Conversation Memory**: Tracks full history, uses last 5 for rewriting
- ✅ **Modularized Prompts**: All prompts in `workflow/prompts.py`

### 6. Memory (`workflow/memory.py`)
- `ConversationMemory` class with window_size=5
- Stores full conversation history

### 7. Evaluation (`evaluation.py`)
- Reads mtRAG RAG.jsonl format
- **Now uses HYBRID retrieval** (was dense-only)
- Generates predictions in mtRAG format
- Supports dataset selection (clapnq, cloud, fiqa, govt, all)

### 8. Prompts (`workflow/prompts.py`)
- ✅ **Modularized** - All prompts in one file
- ✅ **FANC-optimized** - Citations, completeness, natural tone
- ✅ **KG extraction prompts** - For entity/relationship extraction

---

## Environment Variables (.env)

```bash
# OpenAI / OpenRouter
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_EMBEDDING_MODEL=openai/text-embedding-3-small
OPENAI_LLM_MODEL=openai/gpt-4o-mini

# Qdrant Vector Store
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_documents
DENSE_WEIGHT=0.7

# Neo4j Knowledge Graph (NEW)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
ENABLE_KG=true
```

---

## Running

### Start Qdrant
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

### Start Neo4j (NEW)
```bash
docker run -d -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### Interactive Chat
```bash
uv run python main.py
# Select 1. Chat with model
```

### Run Evaluation
```bash
uv run python evaluation.py
# Select dataset (1-5, 5 = all 842 tasks)
```

### Format Check (mtRAG)
```bash
python format_checker.py \
  --input_file evaluation_dataset/human/RAG.jsonl \
  --prediction_file predictions/predictions.jsonl \
  --mode rag_taskc
```

---

## Current Stats

### Indexed Data
| Dataset | Documents | Chunks | Status |
|---------|-----------|--------|--------|
| Clapnq (French Revolution) | 162 MB | ~183,408 | ✅ Indexed |
| Cloud (IBM Cloud) | 126 MB | ~72,442 | ✅ Indexed |
| Fiqa (Financial QA) | 50 MB | ~61,022 | ✅ Indexed |
| Govt (Government) | 108 MB | ~49,607 | ✅ Indexed |
| **Total** | **446 MB** | **~366,479** | ✅ **All indexed** |

---

## Dependencies
```toml
[project]
dependencies = [
    "bs4>=0.0.2",
    "langchain[openai]>=1.2.6",
    "langchain-community>=0.4.1",
    "langchain-text-splitters>=1.1.0",
    "langchain-openai>=1.1.7",
    "langchain-core>=1.2.7",
    "langchain-qdrant>=1.1.0",
    "neo4j>=5.0.0",          # NEW: Knowledge graph
    "python-dotenv>=1.0.0",
    "rich>=14.3.1",
    "tqdm>=4.65.0",
]
requires-python = ">=3.13"
```

---

## KG Implementation Plan

### Neo4j Schema
```cypher
// Entity node (LLM-extracted types)
(:Entity {
  name: "Napoleon Bonaparte",
  type: "PERSON",           // PERSON, ORG, LOC, EVENT, CONCEPT, etc.
  domain: "clapnq"           // Dataset source
})

// Document node (links to Qdrant)
(:Document {
  id: "doc_123",            // Links to Qdrant point ID
  source: "clapnq.jsonl"
})

// Relationships
(:Document)-[:MENTIONS]->(:Entity)
(:Entity)-[:RELATED_TO]->(:Entity)
```

### Key Insight: Text Recall
- **Qdrant stores**: Vectors + **actual text in payload** (`page_content`)
- **Neo4j stores**: Entity → Document links (index only)
- **Citations work**: KG finds document_ids → Fetch text from Qdrant → LLM gets actual text

---

## Remaining Tasks

### Knowledge Graph ✅ ALL PHASES COMPLETE

All 7 phases of Knowledge Graph integration have been completed:

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Neo4j Foundation |
| Phase 2 | ✅ Complete | Entity Extraction |
| Phase 3 | ✅ Complete | Graph Storage |
| Phase 4 | ✅ Complete | Graph Retrieval |
| Phase 5 | ✅ Complete | Context Fusion |
| Phase 6 | ✅ Complete | Integration |
| Phase 7 | ✅ Complete | Testing & Evaluation |

### Other Improvements
| Priority | Task | Description |
|----------|------|-------------|
| **Medium** | Tune prompts | Optimize FANC metrics further |
| **Low** | Add citations | Enhanced citation format in prompts |

---

## Recent Commits
```
d27a906 improving indexing
f63a50b add improvement and dataset for evaluation
da18326 add claude.md
16cd7f3 add generation and indexing
```

---

## Quick Reference

### Test Neo4j Connection
```bash
uv run python -c "from graph_stores.neo4j_client import get_driver; print(get_driver())"
```

### Test Entity Extraction
```bash
uv run python -c "
from workflow.kg_extraction import extract_graph_data
text = 'Napoleon led the French army.'
print(extract_graph_data(text, 'clapnq'))
"
```

### Test Graph Retrieval
```bash
uv run python -c "
from workflow.graph_retrieval import graph_search
doc_ids = graph_search('Tell me about the French Revolution', domain='clapnq', k=5)
print(f'Found documents: {doc_ids}')
"
```

### Test Context Fusion
```bash
uv run python -c "
from workflow.context_fusion import fetch_texts_from_qdrant, format_for_llm
from workflow.graph_retrieval import graph_search

# Graph search returns doc IDs
doc_ids = graph_search('French Revolution', domain='clapnq', k=3)
# Fetch actual text from Qdrant
docs = fetch_texts_from_qdrant(doc_ids)
# Format with citations
print(format_for_llm(docs))
"
```

### View Neo4j in Browser
Visit: http://localhost:7474
Query: `MATCH (n) RETURN n LIMIT 25`

### Quick Integration Test
```bash
uv run python test_kg.py
```

### Run KG Comparison Evaluation
```bash
uv run python compare_kg_evaluation.py
# Select dataset (1-5)
# Generates predictions with and without KG
# Compares results side-by-side
```

### Test Full KG-Integrated RAG
```bash
uv run python -c "
from workflow.generation import retrieve_with_kg
docs, max_score = retrieve_with_kg('Tell me about Napoleon', history=[], k=5)
print(f'Found {len(docs)} documents with max_score={max_score}')
for doc in docs:
    print(f'  - {doc.page_content[:50]}...')
"
```
