"""System prompts for RAG generation, query rewriting, and knowledge graph extraction.

These prompts are optimized for mtRAG FANC metrics:
- Faithfulness: Citations required
- Appropriateness: Conversational context awareness
- Naturalness: Human-like responses
- Completeness: Thorough answers

Inspired by LightRAG (HKUDS) for better entity/relationship extraction.
"""

import os

# ============================================================================
# Knowledge Graph Extraction Prompts (LightRAG-inspired)
# ============================================================================

# Delimiters for structured output (avoids JSON parsing issues)
# DEPRECATED: Kept for backward compatibility. Use structured output instead.
TUPLE_DELIMITER = "<|#|>"
COMPLETION_DELIMITER = "<|COMPLETE|>"

# Entity types - configurable via environment variable
# Default types inspired by LightRAG and common KG schemas
DEFAULT_ENTITY_TYPES = [
    "PERSON",       # People, including fictional
    "ORGANIZATION", # Companies, NGOs, governments
    "LOCATION",     # Places, cities, countries, buildings
    "EVENT",        # Historical events, meetings, wars
    "CONCEPT",      # Ideas, theories, methodologies
    "PRODUCT",      # Software, hardware, services
    "DATE",         # Specific dates, time periods
    "OTHER",        # Catch-all for entities that don't fit above
]

ENTITY_TYPES = os.getenv("KG_ENTITY_TYPES", ",".join(DEFAULT_ENTITY_TYPES)).split(",")

# Domain-aware entity types for better extraction quality
DOMAIN_ENTITY_TYPES = {
    "french revolution": ["Person", "Event", "Location", "Concept", "Date", "Organization"],
    "frenchrevolution": ["Person", "Event", "Location", "Concept", "Date", "Organization"],
    "ibm cloud": ["Product", "Technology", "Organization", "Feature", "Concept"],
    "ibmcloud": ["Product", "Technology", "Organization", "Feature", "Concept"],
    "cloud": ["Product", "Technology", "Organization", "Feature", "Concept"],
    "financial": ["Person", "Organization", "Concept", "Date", "Product", "Market"],
    "fiqa": ["Person", "Organization", "Concept", "Date", "Product", "Market"],
    "government": ["Organization", "Person", "Location", "Concept", "Date", "Policy"],
    "govt": ["Organization", "Person", "Location", "Concept", "Date", "Policy"],
}


def get_entity_types_for_domain(domain: str | None) -> list[str]:
    """Get entity types for a specific domain, with fuzzy matching.

    Args:
        domain: Domain name (e.g., "French Revolution", "IBM Cloud")

    Returns:
        List of entity types for the domain, or DEFAULT_ENTITY_TYPES if no match
    """
    if not domain:
        return ENTITY_TYPES

    domain_lower = domain.lower().strip().replace(" ", "").replace("-", "")

    # Try exact match first
    for key, types in DOMAIN_ENTITY_TYPES.items():
        if domain_lower == key.lower():
            return types

    # Try substring match
    for key, types in DOMAIN_ENTITY_TYPES.items():
        if key.lower() in domain_lower or domain_lower in key.lower():
            return types

    # Fallback to default
    return ENTITY_TYPES

# Few-shot examples for entity extraction
ENTITY_EXTRACTION_EXAMPLES = """
**Example 1:**
Text: "Napoleon Bonaparte led the French army to victory at the Battle of Austerlitz in 1805."
entity{tuple_delimiter}Napoleon Bonaparte{tuple_delimiter}PERSON{tuple_delimiter}Emperor of France who led the French army
entity{tuple_delimiter}French army{tuple_delimiter}ORGANIZATION{tuple_delimiter}Military force of France
entity{tuple_delimiter}Battle of Austerlitz{tuple_delimiter}EVENT{tuple_delimiter}Military battle in 1805
entity{tuple_delimiter}1805{tuple_delimiter}DATE{tuple_delimiter}Year of the Battle of Austerlitz
relation{tuple_delimiter}Napoleon Bonaparte{tuple_delimiter}French army{tuple_delimiter}led, commanded{tuple_delimiter}Napoleon was the commander of the French army
relation{tuple_delimiter}Napoleon Bonaparte{tuple_delimiter}Battle of Austerlitz{tuple_delimiter}won, achieved victory{tuple_delimiter}Napoleon led the French army to victory at Austerlitz

**Example 2:**
Text: "IBM Cloud provides content delivery network services through its CDN infrastructure."
entity{tuple_delimiter}IBM Cloud{tuple_delimiter}ORGANIZATION{tuple_delimiter}Cloud computing division of IBM
entity{tuple_delimiter}content delivery network{tuple_delimiter}CONCEPT{tuple_delimiter}Distributed network of servers that delivers content
entity{tuple_delimiter}CDN infrastructure{tuple_delimiter}PRODUCT{tuple_delimiter}Infrastructure for content delivery
relation{tuple_delimiter}IBM Cloud{tuple_delimiter}CDN infrastructure{tuple_delimiter}provides, offers{tuple_delimiter}IBM Cloud provides CDN infrastructure services
relation{tuple_delimiter}content delivery network{tuple_delimiter}CDN infrastructure{tuple_delimiter}enables, implements{tuple_delimiter}CDN infrastructure implements content delivery network concept
{completion_delimiter}
"""

# Main entity extraction prompt (LightRAG-style)
ENTITY_EXTRACTION_PROMPT = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1. **Entity Extraction & Output:**
   * **Identification:** Identify clearly defined and meaningful entities in the input text.
   * **Entity Details:** For each identified entity, extract the following information:
     - `entity_name`: The name of the entity. Use title case for consistent naming.
     - `entity_type`: Categorize the entity using one of the following types: `{entity_types}`.
       If none of the provided entity types apply, classify it as `OTHER`.
     - `entity_description`: A concise description based *solely* on information in the input text.
   * **Output Format - Entities:** Output 3 fields for each entity, delimited by `{tuple_delimiter}`, on a single line.
     Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2. **Relationship Extraction & Output:**
   * **Identification:** Identify direct, clearly stated relationships between extracted entities.
   * **Relationship Details:** For each relationship, extract:
     - `source_entity`: Name of the source entity (use consistent naming from entity extraction).
     - `target_entity`: Name of the target entity (use consistent naming from entity extraction).
     - `relationship_keywords`: High-level keywords summarizing the relationship nature. Separate multiple keywords with comma `,`.
     - `relationship_description`: Concise explanation of the relationship.
   * **Output Format - Relationships:** Output 4 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line.
     Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3. **Delimiter Usage:**
   * The `{tuple_delimiter}` is a complete marker and **must not be filled with content**. Use it strictly as a field separator.

4. **Relationship Direction & Duplication:**
   * Treat relationships as **undirected** unless explicitly stated otherwise.
   * Avoid duplicate relationships (A→B is same as B→A for undirected).

5. **Output Order:**
   * Output all entities first, followed by all relationships.
   * Prioritize relationships most significant to the core meaning of the text.

6. **Context & Objectivity:**
   * Write all entity names and descriptions in **third person**.
   * Avoid pronouns such as `this article`, `I`, `you`, `he/she`.
   * Use explicit names instead.

7. **Completion Signal:** Output the literal string `{completion_delimiter}` after all entities and relationships have been extracted.

---Examples---
{examples}

---Text---
```
{text}
```

---Output---
"""

# Query entity extraction prompt (for graph search during retrieval)
QUERY_ENTITY_PROMPT = """Extract entities from this user query.

**Output Format:**
- Entity: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

**Entity Types:** Use one of: `{entity_types}`, or `OTHER` if none apply.

**Rules:**
- Max 5 entities (be selective - only key entities)
- Use title case, third person
- End with: `{completion_delimiter}`

**Query:** {query}

**Output:**
"""

# Batch entity extraction prompt - multiple documents in one LLM call
BATCH_ENTITY_EXTRACTION_PROMPT = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from multiple documents.

---Instructions---
For each document below, extract entities and relationships.

**Output Format (per document):**
Prepend each line with the document ID:
`doc_id:{id}{tuple_delimiter}entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`
`doc_id:{id}{tuple_delimiter}relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}keywords{tuple_delimiter}description`

**Entity Types:** Use one of: `{entity_types}`, or `OTHER` if none apply.

**Rules:**
- Output entities first, then relationships for each document
- Max 5 entities, max 5 relations per document
- Use title case, third person, no pronouns
- Include doc_id prefix in EVERY output line
- End with: `{completion_delimiter}`

---Documents---
{documents}

---Output---
"""

# ============================================================================
# Structured Output Prompts (New - replaces delimited format)
# ============================================================================

# System prompt for structured entity extraction
STRUCTURED_ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are an expert knowledge graph extractor.
Extract entities and relationships from the given text.

## Entity Types
{entity_types}

## Coreference Resolution
When the same entity is referred to by different names or pronouns,
always use the most complete and specific identifier throughout.
Examples:
- "he", "the emperor", "Napoleon" -> always use "Napoleon Bonaparte"
- "the king", "Louis XVI" -> always use "Louis XVI"
- "it", "the cloud", "IBM Cloud" -> always use "IBM Cloud"

## Output Format
Return a JSON object with:
- entities: list of {{name, type, description}}
- relationships: list of {{source, target, type, description}}

## Rules
- Only extract from the provided text
- Be specific, avoid generic entities
- Use consistent, complete names for entities (apply coreference resolution)
- Description should be concise but informative
- Use camelCase for relationship types (e.g., WROTE_ABOUT, BORN_IN, LED)
- Treat relationships as undirected unless explicitly stated otherwise
- Write entity names and descriptions in third person
"""

# User prompt for structured entity extraction
STRUCTURED_ENTITY_EXTRACTION_USER_PROMPT = """Text: {text}

Extract all entities and relationships."""

# Gleaning prompt - second pass to catch missed entities
GLEANING_PROMPT = """Some entities or relationships may have been missed.
Re-read the text and extract anything not already captured.

Text: {text}

Already extracted:
{previous_result}

Extract any additional entities or relationships not already captured.
Use the same output format."""

# ============================================================================
# RAG Generation Prompts
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
- Answer thoroughly and completely - cover all relevant points from the context
- Cite inline using [1], [2], etc. immediately after each claim, e.g. "X is true [1]."
- If multiple sources support a claim, cite all: [1][3]
- Use a clear, natural tone - avoid bullet overuse, prefer prose
- If the context lacks enough information, explicitly state what is and is not covered
- Never fabricate information not present in the context

## Question
{question}

## Answer"""

# ============================================================================
# Helper Functions
# ============================================================================

def format_entity_prompt(text: str, examples: str = ENTITY_EXTRACTION_EXAMPLES) -> str:
    """Format entity extraction prompt with delimiters and entity types."""
    return ENTITY_EXTRACTION_PROMPT.format(
        text=text,
        tuple_delimiter=TUPLE_DELIMITER,
        completion_delimiter=COMPLETION_DELIMITER,
        entity_types=", ".join(ENTITY_TYPES),
        examples=examples,
    )


def format_query_prompt(query: str) -> str:
    """Format query entity extraction prompt."""
    return QUERY_ENTITY_PROMPT.format(
        query=query,
        tuple_delimiter=TUPLE_DELIMITER,
        completion_delimiter=COMPLETION_DELIMITER,
        entity_types=", ".join(ENTITY_TYPES),
    )


def format_batch_prompt(texts_with_ids: list[tuple[int, str]]) -> str:
    """Format batch entity extraction prompt.

    Args:
        texts_with_ids: List of (doc_id, text) tuples

    Returns:
        Formatted prompt with multiple documents
    """
    # Format documents block
    docs_block = ""
    for doc_id, text in texts_with_ids:
        # Truncate very long texts
        truncated = text[:1000] + "..." if len(text) > 1000 else text
        docs_block += f"Document {doc_id}:\n{truncated}\n\n"

    return BATCH_ENTITY_EXTRACTION_PROMPT.format(
        documents=docs_block,
        tuple_delimiter=TUPLE_DELIMITER,
        completion_delimiter=COMPLETION_DELIMITER,
        entity_types=", ".join(ENTITY_TYPES),
    )


# ============================================================================
# Structured Output Helper Functions
# ============================================================================

def format_structured_entity_prompt(
    text: str,
    domain: str | None = None,
) -> tuple[str, str]:
    """Format prompts for structured entity extraction.

    Args:
        text: The text to extract entities from
        domain: Optional domain for domain-aware entity types

    Returns:
        Tuple of (system_prompt, user_prompt) for structured output
    """
    entity_types = get_entity_types_for_domain(domain)

    system_prompt = STRUCTURED_ENTITY_EXTRACTION_SYSTEM_PROMPT.format(
        entity_types=", ".join(entity_types),
    )

    user_prompt = STRUCTURED_ENTITY_EXTRACTION_USER_PROMPT.format(
        text=text[:2000],  # Truncate for context window
    )

    return system_prompt, user_prompt


def format_gleaning_prompt(
    text: str,
    previous_result: dict,
) -> str:
    """Format gleaning prompt for second extraction pass.

    Args:
        text: Original text to extract from
        previous_result: Previous extraction result with entities/relationships

    Returns:
        Formatted gleaning prompt
    """
    # Show sample of previously extracted entities
    entities_sample = previous_result.get("entities", [])[:3]
    sample_str = ", ".join([e.get("name", "unknown") for e in entities_sample])

    return GLEANING_PROMPT.format(
        text=text[:2000],
        previous_result=f"Entities: {sample_str}..." if entities_sample else "No entities found yet.",
    )
