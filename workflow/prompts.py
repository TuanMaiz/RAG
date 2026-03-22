"""System prompts for RAG generation, query rewriting, and knowledge graph extraction.

These prompts are optimized for mtRAG FANC metrics:
- Faithfulness: Citations required
- Appropriateness: Conversational context awareness
- Naturalness: Human-like responses
- Completeness: Thorough answers

Inspired by LightRAG (HKUDS) for better entity/relationship extraction.
"""

# ============================================================================
# Knowledge Graph Extraction Prompts (LightRAG-inspired)
# ============================================================================

# Delimiters for structured output (avoids JSON parsing issues)
TUPLE_DELIMITER = "<|#|>"
COMPLETION_DELIMITER = "<|COMPLETE|>"

# Entity extraction from text - uses delimited format (not JSON)
ENTITY_EXTRACTION_PROMPT = """Extract entities and relationships from the text.

**Output Format:**
- Entity: `entity{delimiter}name{delimiter}type{delimiter}description`
- Relation: `relation{delimiter}source{delimiter}target{delimiter}keywords{delimiter}description`

**Rules:**
- Output entities first, then relations
- Max 10 entities, max 10 relations (be selective)
- Use title case, third person, no pronouns
- End with: `{completion_delimiter}`

**Text:**
```
{text}
```

**Output:**
"""

# Relationship extraction prompt (now uses same delimited format)
RELATIONSHIP_EXTRACTION_PROMPT = """Extract relationships between the entities.

**Entities:** {entities}

**Format:** `relation{delimiter}source{delimiter}target{delimiter}keywords{delimiter}description`

**Text:**
```
{text}
```

**Output:**
"""

# Entity extraction from user query (for graph search)
QUERY_ENTITY_PROMPT = """Extract entities from this query.

**Format:** `entity{delimiter}name{delimiter}type{delimiter}description`

**Query:** {query}

**Output:**
"""

# ============================================================================
# Existing RAG Prompts
# ============================================================================

# IDK Detection (when retrieval score is below threshold)
IDK_MESSAGE = "I couldn't find relevant information in the available documents to answer this. Could you try rephrasing, or ask about a related topic covered in the documentation?"

# Query Rewrite Judge (decides if query needs context from previous turns)
REWRITE_JUDGE_PROMPT = """Determine if the query below requires context from the chat history to be fully understood.

Chat history:
{history_text}

Query: {query}

Reply with a single word: yes or no"""

# Query Rewrite (rewrites query to be standalone for retrieval)
QUERY_REWRITE_PROMPT = """You are a query rewriter for a retrieval system. Rewrite the current query to be standalone and clear.

Chat history:
{history_text}

Current query: {query}

Rewrite the query to:
- Replace pronouns (he, she, it, they, this, that) with the actual entities
- Make it fully understandable without the chat history
- Keep it concise and natural

Do NOT include any prefix like 'Rewritten query:' or explanation. Return the query text only."""

# Main Generation System Prompt (optimized for FANC metrics)
RAG_SYSTEM_PROMPT = """You are a knowledgeable assistant. Answer questions using ONLY the numbered context below.

## Context
{docs_content}

## Rules
- Answer thoroughly and completely — cover all relevant points from the context
- Cite inline using [1], [2], etc. immediately after each claim, e.g. "X is true [1]."
- If multiple sources support a claim, cite all: [1][3]
- Use a clear, natural tone — avoid bullet overuse, prefer prose
- If the context lacks enough information, explicitly state what is and isn't covered
- Never fabricate information not present in the context

## Question
{question}

## Answer"""


# Helper function to format prompts with delimiters
def format_entity_prompt(text: str, examples: str = "") -> str:
    """Format entity extraction prompt with delimiters."""
    return ENTITY_EXTRACTION_PROMPT.format(
        text=text,
        delimiter=TUPLE_DELIMITER,
        completion_delimiter=COMPLETION_DELIMITER,
    )


def format_relationship_prompt(text: str, entities: list) -> str:
    """Format relationship extraction prompt."""
    # Format entities for display
    entity_list = ", ".join([f"{e['name']} ({e.get('type', 'OTHER')})" for e in entities])

    return RELATIONSHIP_EXTRACTION_PROMPT.format(
        text=text,
        entities=entity_list,
        delimiter=TUPLE_DELIMITER,
        completion_delimiter=COMPLETION_DELIMITER,
    )


def format_query_prompt(query: str) -> str:
    """Format query entity extraction prompt."""
    return QUERY_ENTITY_PROMPT.format(
        query=query,
        delimiter=TUPLE_DELIMITER,
        completion_delimiter=COMPLETION_DELIMITER,
    )
