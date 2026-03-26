# KG Pipeline Plan — MTRAG

## Overview

```
MTRAG Dataset (4 domains)
        │
        ▼
┌─────────────────────┐
│   INDEXING PIPELINE │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Neo4j + Vectors   │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   RAG PIPELINE      │  ← Subtask C
└─────────────────────┘
        │
        ▼
     Answer
```

---

## 1.1 Preprocessing

### Input structure (MTRAG jsonl)

```python
{
  "id": "...",
  "title": "French Revolution",  # optional
  "text": "Furet emphasises that..."
}
```

### Step 1: Parse + Classify each line

```python
MIN_TOKENS = 150
MAX_TOKENS = 512

def classify(line: dict) -> Literal["short", "normal", "long"]:
    tokens = count_tokens(line["text"])
    if tokens < MIN_TOKENS:
        return "short"
    elif tokens > MAX_TOKENS:
        return "long"
    return "normal"
```

### Step 2: Domain grouping

Since title is optional, you need a fallback:

```python
def get_domain(line: dict) -> str:
    if "title" in line:
        return line["title"]
    return line.get("id", "unknown").split("-")[0]
```

Group all lines by domain before any chunking:

```python
from collections import defaultdict

domain_groups: dict[str, list[dict]] = defaultdict(list)
for line in jsonl:
    domain_groups[get_domain(line)].append(line)
```

### Step 3: Per-domain processing

For each domain group, process lines in order:

```python
for domain, lines in domain_groups.items():
    buffer = []
    buffer_tokens = 0

    for line in lines:
        kind = classify(line)

        if kind == "short":
            buffer.append(line)
            buffer_tokens += count_tokens(line["text"])

            if buffer_tokens >= MIN_TOKENS:
                yield make_chunk(buffer, domain)  # flush
                buffer, buffer_tokens = [], 0

        elif kind == "normal":
            if buffer:                            # flush pending buffer first
                yield make_chunk(buffer, domain)
                buffer, buffer_tokens = [], 0
            yield make_chunk([line], domain)

        elif kind == "long":
            if buffer:                            # flush pending buffer first
                yield make_chunk(buffer, domain)
                buffer, buffer_tokens = [], 0
            for chunk in split_long(line, MAX_TOKENS):
                yield chunk

    if buffer:                                    # flush remaining
        yield make_chunk(buffer, domain)
```

### Step 4: make_chunk + split_long

```python
def make_chunk(lines: list[dict], domain: str) -> dict:
    parts = []
    for line in lines:
        if "title" in line:
            parts.append(f"[{line['title']}] {line['text']}")
        else:
            parts.append(line["text"])

    return {
        "chunk_id": generate_id(),
        "text": "\n".join(parts),
        "domain": domain,
        "title": lines[0].get("title", domain),
        "source_ids": [l["id"] for l in lines]
    }

def split_long(line: dict, max_tokens: int, overlap: int = 50) -> list[dict]:
    words = line["text"].split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_tokens
        chunk_text = " ".join(words[start:end])

        if "title" in line:
            chunk_text = f"[{line['title']}] {chunk_text}"

        chunks.append({
            "chunk_id": generate_id(),
            "text": chunk_text,
            "domain": get_domain(line),
            "title": line.get("title", ""),
            "source_ids": [line["id"]]
        })
        start = end - overlap  # sliding window

    return chunks
```

### Output

Every path produces the same chunk schema:

```python
{
    "chunk_id": "chunk-xxxxxx",
    "text": "[French Revolution] Furet emphasises...",
    "domain": "French Revolution",
    "title": "French Revolution",
    "source_ids": ["doc-001", "doc-002"]  # may be multiple if merged
}
```

### Edge cases

| Case | Handling |
|---|---|
| No title | Use id prefix as domain |
| Buffer never reaches MIN_TOKENS | Flush at end of domain group |
| Single sentence, no title | Merge with next line in same domain |
| Long line with no title | Split + prepend domain as context |

---

## 1.2 Entity Extraction

### Input

Each chunk from 1.1:

```python
{
    "chunk_id": "chunk-xxxxxx",
    "text": "[French Revolution] Furet emphasises...",
    "domain": "French Revolution",
    "title": "French Revolution",
    "source_ids": ["doc-001"]
}
```

### Step 1: Prompt Design

Two prompts, split for OpenAI prompt caching:

```python
# Built ONCE per domain (static, cache hit across chunks)
system_prompt = """
You are an expert knowledge graph extractor.
Extract entities and relationships from the given text.

Entity types: {entity_types}

## Coreference Resolution
When the same entity is referred to by different names or pronouns,
always use the most complete and specific identifier throughout.
Example: "he", "the emperor", "Napoleon" → always use "Napoleon Bonaparte"

## Output format (use structured JSON)
Return a JSON object with:
- entities: list of {{name, type, description}}
- relationships: list of {{source, target, type, description}}

## Rules
- Only extract from the provided text
- Be specific, avoid generic entities
- Use consistent, complete names for entities
- Description should be concise but informative
- Use camelCase for relationship types (e.g., WROTE_ABOUT, BORN_IN)
""".format(entity_types=ENTITY_TYPES[domain])

# Built PER chunk (dynamic, changes every call)
user_prompt = """
Text: {text}

Extract all entities and relationships.
""".format(text=chunk["text"])
```

Domain-aware entity types:

```python
ENTITY_TYPES = {
    "French Revolution": ["Person", "Event", "Location", "Concept", "Date"],
    "IBM Cloud":         ["Product", "Technology", "Organization", "Feature"],
    "default":           ["Person", "Organization", "Location", "Concept", "Event"]
}
```

### Step 2: Cache Check (before LLM call)

```python
async def use_llm_func_with_cache(
    chunk_id: str,
    user_prompt: str,
    system_prompt: str,
    cache: dict
) -> tuple[str, datetime]:

    cache_key = hash(user_prompt)  # system_prompt excluded (cached by OpenAI)

    if cache_key in cache:
        return cache[cache_key]["result"], cache[cache_key]["timestamp"]

    result = await call_llm(system_prompt, user_prompt)
    cache[cache_key] = {"result": result, "timestamp": datetime.now()}
    return result, datetime.now()
```

Two levels of caching:
- **OpenAI-level**: system prompt prefix cached across requests automatically
- **App-level**: full response cached by `hash(user_prompt)` to avoid re-calling for same chunk

### Step 3: Initial Extraction

```python
result, timestamp = await use_llm_func_with_cache(
    chunk_id=chunk["chunk_id"],
    user_prompt=user_prompt,
    system_prompt=system_prompt,
    cache=llm_cache
)

entities, relationships = parse_extraction_result(result)
```

### Step 4: Gleaning Pass

```python
GLEANING_PROMPT = """
Some entities or relationships may have been missed.
Re-read the text and extract anything not already captured.

Text: {text}
Already extracted:
{previous_result}
"""

if entity_extract_max_gleaning > 0:
    history = [
        {"role": "user",      "content": user_prompt},
        {"role": "assistant", "content": result}
    ]
    gleaning_user_prompt = GLEANING_PROMPT.format(
        text=chunk["text"],
        previous_result=result
    )
    full_context = system_prompt + json.dumps(history) + gleaning_user_prompt

    if count_tokens(full_context) > MAX_INPUT_TOKENS:
        logger.warning(f"Gleaning skipped for {chunk_id}: token limit exceeded")
    else:
        glean_result, _ = await use_llm_func_with_cache(...)
        glean_entities, glean_relationships = parse_extraction_result(glean_result)

        # merge gleaning into initial (longer description wins)
        entities, relationships = merge_results(
            (entities, relationships),
            (glean_entities, glean_relationships)
        )
```

### Step 5: Structured Output + Normalize

Instead of fragile string parsing, use OpenAI structured output (function calling) to get typed objects directly:

```python
from pydantic import BaseModel

class Entity(BaseModel):
    name: str
    type: str
    description: str

class Relationship(BaseModel):
    source: str
    target: str
    type: str
    description: str

class KnowledgeGraph(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]

# LLM returns KnowledgeGraph directly — no string parsing needed
structured_llm = llm.with_structured_output(KnowledgeGraph)
result: KnowledgeGraph = await structured_llm.ainvoke([
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": user_prompt}
])

# Normalize names after extraction
def normalize(name: str) -> str:
    return name.lower().strip()

entities = [
    {**e.model_dump(), "name": normalize(e.name)}
    for e in result.entities
]
relationships = [
    {**r.model_dump(), "source": normalize(r.source), "target": normalize(r.target)}
    for r in result.relationships
]
```

> **Why**: Raw string parsing with `line.startswith("ENTITIES:")` breaks on any LLM formatting variation. Structured output guarantees a valid typed object every time — no try/except parsing logic needed.

### Output per chunk

```python
{
    "chunk_id": "chunk-xxxxxx",
    "domain": "French Revolution",
    "timestamp": "2024-01-01T00:00:00",
    "entities": [
        {"name": "furet", "type": "Person", "description": "French historian..."},
    ],
    "relationships": [
        {"source": "furet", "target": "french revolution",
         "type": "WROTE_ABOUT", "description": "..."}
    ]
}
```

### Full flow summary

```
chunk
  │
  ├─ build system_prompt (domain-aware, static, with coreference instruction)
  ├─ build user_prompt (chunk text, dynamic)
  │
  ├─ [cache check] → hit? return cached result
  │                → miss? call LLM
  │
  ├─ structured output → KnowledgeGraph (typed, no parsing)
  │
  ├─ normalize names (lowercase + strip)
  │
  ├─ [gleaning]
  │   ├─ token guard
  │   ├─ call LLM with history
  │   ├─ structured output → KnowledgeGraph
  │   └─ merge (longer description wins)
  │
  └─ return {chunk_id, entities, relationships}
```

---

## 1.3 In-memory Dedup

### The Problem

After extracting from all chunks, you'll have duplicates:

```python
# chunk-001:
{"name": "napoleon bonaparte", "type": "Person", "description": "French general"}

# chunk-045:
{"name": "napoleon bonaparte", "type": "Person", "description": "French general and emperor who led many campaigns across Europe"}

# chunk-099:
{"name": "napoleon", "type": "Person", "description": "Emperor of France"}
```

Three problems:
1. **Same entity, different descriptions** → keep best
2. **Same entity, different name variants** → need to unify
3. **Same relationship, described differently** → keep best

### Step 1: Collect all chunk results

```python
tasks = [extract(chunk) for chunk in chunks]
all_chunks_results = await asyncio.gather(*tasks)
```

### Step 2: Name normalization (before dedup)

```python
NAME_ALIASES = {
    "napoleon":   "napoleon bonaparte",
    "napoleon i": "napoleon bonaparte",
    "the emperor": "napoleon bonaparte",  # domain-specific
}

DOMAIN_ALIASES = {
    "French Revolution": {
        "the emperor": "napoleon bonaparte",
        "the king":    "louis xvi",
    }
}

def resolve_alias(name: str, domain: str) -> str:
    domain_aliases = DOMAIN_ALIASES.get(domain, {})
    if name in domain_aliases:
        return domain_aliases[name]
    return NAME_ALIASES.get(name, name)
```

Two levels of aliases:
- **Global** — common across all domains
- **Domain-specific** — "the emperor" only means Napoleon in French Revolution context

### Step 3: Merge nodes in memory

```python
merged_nodes: dict[tuple, dict] = {}
# key: (normalized_name, type)

for chunk_result in all_chunks_results:
    domain = chunk_result["domain"]

    for entity in chunk_result["entities"]:
        name = resolve_alias(entity["name"], domain)
        key = (name, entity["type"])

        if key not in merged_nodes:
            merged_nodes[key] = {
                "name": name,
                "type": entity["type"],
                "description": entity["description"],
                "domain": domain,
                "seen_in_chunks": [chunk_result["chunk_id"]]
            }
        else:
            existing = merged_nodes[key]

            # longer description wins
            if len(entity["description"]) > len(existing["description"]):
                existing["description"] = entity["description"]

            existing["seen_in_chunks"].append(chunk_result["chunk_id"])
```

### Step 4: Merge edges in memory

```python
merged_edges: dict[tuple, dict] = {}
# key: (source, target, type)

for chunk_result in all_chunks_results:
    domain = chunk_result["domain"]

    for rel in chunk_result["relationships"]:
        src = resolve_alias(rel["source"], domain)
        tgt = resolve_alias(rel["target"], domain)
        key = (src, tgt, rel["type"])

        if key not in merged_edges:
            merged_edges[key] = {
                "source": src,
                "target": tgt,
                "type": rel["type"],
                "description": rel["description"],
                "domain": domain,
                "seen_in_chunks": [chunk_result["chunk_id"]]
            }
        else:
            existing = merged_edges[key]

            if len(rel["description"]) > len(existing["description"]):
                existing["description"] = rel["description"]

            existing["seen_in_chunks"].append(chunk_result["chunk_id"])
```

### Step 5: Orphan edge guard

```python
valid_names = {name for (name, _) in merged_nodes.keys()}

merged_edges = {
    key: edge for key, edge in merged_edges.items()
    if edge["source"] in valid_names and edge["target"] in valid_names
}
```

### Output

```python
{
    "nodes": merged_nodes,  # (name, type) → entity dict
    "edges": merged_edges   # (src, tgt, type) → relationship dict
}
```

### Full flow summary

```
all chunk results
  │
  ├─ [name normalization]
  │   ├─ lowercase + strip (done in 1.2)
  │   └─ alias resolution (global + domain-specific)
  │
  ├─ [merge nodes]
  │   ├─ key: (name, type)
  │   ├─ longer description wins
  │   └─ track seen_in_chunks
  │
  ├─ [merge edges]
  │   ├─ key: (source, target, type)
  │   ├─ longer description wins
  │   └─ track seen_in_chunks
  │
  └─ [orphan edge guard]
      └─ drop edges with unknown source/target

  → (merged_nodes, merged_edges) → Neo4j upsert
```

### Where this can be improved (thesis)

| Current | Improvement |
|---|---|
| Alias dict is manual | Embedding similarity to auto-detect aliases |
| Longer description wins | LLM-based description merging |
| Type used in key | Cross-type entity resolution |
| Coreference in prompt (best effort) | Dedicated coreference resolution model pre-pass |

> **Note**: Coreference resolution in the system prompt (from LangChain blog) reduces alias noise before extraction, meaning fewer entries hit the alias dict in Step 2. It doesn't eliminate the need for alias resolution but reduces its burden significantly.