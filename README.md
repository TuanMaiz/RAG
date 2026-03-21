# mtRAG: Multi-Turn Conversational RAG System

A Retrieval-Augmented Generation (RAG) system competing in the [mtRAG benchmark](https://github.com/TAR3tHub/mtRAG) (TACL 2025 - Multi-Turn Conversational RAG).

## Features

- **Hybrid Retrieval**: Dense (OpenAI embeddings) + Sparse (BM25) with RRF fusion
- **Knowledge Graph**: Neo4j-based entity extraction and graph-enhanced retrieval
- **Query Rewriting**: LLM-based query rewriting for multi-turn context
- **IDK Detection**: Confidence threshold for unanswerable questions
- **Conversation Memory**: Tracks conversation history for context-aware responses

## Architecture

```
Query → [Rewrite] → [Duplicate] → Hybrid Search (Qdrant)
                    ↓
                Graph Search (Neo4j) → Document IDs
                    ↓
                Context Fusion (merge + deduplicate)
                    ↓
                LLM → Response with [1], [2] citations
```

## Quick Start

### Prerequisites

- Python 3.13+
- Docker (for Qdrant and Neo4j)

### Installation

```bash
# Install dependencies
uv pip install -e .

# Or with pip
pip install -e .
```

### Environment Variables

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Edit `.env`:
```bash
# OpenAI / OpenRouter
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_EMBEDDING_MODEL=openai/text-embedding-3-small
OPENAI_LLM_MODEL=openai/gpt-4o-mini

# Qdrant Vector Store
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_documents
DENSE_WEIGHT=0.7

# Neo4j Knowledge Graph
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
ENABLE_KG=true
```

### Start Services

```bash
# Start Qdrant (vector store)
docker run -d -p 6333:6333 qdrant/qdrant

# Start Neo4j (knowledge graph)
docker run -d -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### Run the Application

```bash
uv run python main.py
```

Select:
1. **Chat with model** - Interactive chat mode
2. **Process documents** - Index documents into vector store
3. **Exit**

## Project Structure

```
project/
├── main.py                      # CLI entry point
├── evaluation.py                # mtRAG evaluation script
├── format_checker.py            # mtRAG format validation
├── drop_qdrant.py               # Clear Qdrant collection
├── compare_kg_evaluation.py     # KG vs non-KG comparison
│
├── model/                       # LLM initialization
│   └── __init__.py              # get_llm() function
│
├── loaders/                     # Document loading
│   └── document_loader.py       # JSONLLoader
│
├── vector_stores/               # Vector storage
│   └── qdrant.py                # Qdrant client + embeddings
│
├── graph_stores/                # Knowledge graph storage
│   └── neo4j_client.py          # Neo4j client
│
├── workflow/                    # Core RAG workflows
│   ├── indexing.py              # Document indexing
│   ├── generation.py            # RAG generation
│   ├── memory.py                # Conversation memory
│   ├── prompts.py               # System prompts
│   ├── hybrid_retrieval.py      # Dense + sparse search
│   ├── kg_extraction.py         # Entity extraction
│   ├── graph_retrieval.py       # Graph search
│   └── context_fusion.py        # Merge vector + graph results
│
├── dataset/                     # Training data
│   ├── clapnq.jsonl             # French Revolution
│   ├── cloud.jsonl              # IBM Cloud CDN
│   ├── fiqa.jsonl               # Financial QA
│   └── govt.jsonl               # Government documents
│
├── evaluation_dataset/          # mtRAG benchmark
│   └── human/RAG.jsonl          # 842 tasks
│
└── predictions/                 # Output predictions
    └── predictions.jsonl
```

## Running Evaluation

```bash
uv run python evaluation.py
```

Select dataset:
1. Clapnq (French Revolution)
2. Cloud (IBM Cloud)
3. Fiqa (Financial QA)
4. Govt (Government)
5. ALL datasets

### Format Check (mtRAG)

```bash
python format_checker.py \
  --input_file evaluation_dataset/human/RAG.jsonl \
  --prediction_file predictions/predictions.jsonl \
  --mode rag_taskc
```

## Utilities

### Reset Vector Store

```bash
uv run python drop_qdrant.py
```

### Quick KG Test

```bash
uv run python test_kg.py
```

### Compare KG vs Non-KG

```bash
uv run python compare_kg_evaluation.py
```

## Knowledge Graph Integration

The system uses Neo4j to store entities and their relationships:

1. **Entity Extraction**: LLM extracts entities (PERSON, ORG, LOC, EVENT, etc.)
2. **Graph Storage**: Entities linked to documents via MENTIONS relationships
3. **Graph Retrieval**: Query by entity, expand through relationships
4. **Context Fusion**: Merge vector + graph results with deduplication

### Neo4j Schema

```cypher
(:Entity {name, type, domain})
(:Document {id, domain})
(:Document)-[:MENTIONS]->(:Entity)
(:Entity)-[:RELATED_TO]->(:Entity)
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI/OpenRouter API key |
| `OPENAI_BASE_URL` | - | API base URL |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model |
| `OPENAI_LLM_MODEL` | `gpt-4o-mini` | LLM model |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant URL |
| `QDRANT_COLLECTION` | `rag_documents` | Collection name |
| `DENSE_WEIGHT` | `0.7` | Dense retrieval weight |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `ENABLE_KG` | `true` | Enable knowledge graph |

## License

MIT License
