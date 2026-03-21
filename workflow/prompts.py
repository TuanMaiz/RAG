"""System prompts for RAG generation, query rewriting, and knowledge graph extraction.

These prompts are optimized for mtRAG FANC metrics:
- Faithfulness: Citations required
- Appropriateness: Conversational context awareness
- Naturalness: Human-like responses
- Completeness: Thorough answers
"""

# ============================================================================
# Knowledge Graph Extraction Prompts
# ============================================================================

# Entity extraction from text
ENTITY_EXTRACTION_PROMPT = """Extract key entities from the text below.

Analyze the text and identify important entities such as:
- People (PERSON)
- Organizations (ORG)
- Locations (LOC)
- Events (EVENT)
- Concepts (CONCEPT)
- Products/Services (PRODUCT)
- Dates/Time periods (DATE)

Text: {text}

Return ONLY valid JSON with this exact format:
{{"entities": [{{"name": "Entity Name", "type": "PERSON|ORG|LOC|EVENT|CONCEPT|PRODUCT|DATE"}}]}}

Rules:
- Extract only significant, specific entities
- Use the most specific type possible
- Return valid JSON only, no explanation
"""

# Relationship extraction between entities
RELATIONSHIP_EXTRACTION_PROMPT = """Extract relationships between the entities in the text below.

Entities found: {entities}

Text: {text}

Identify how these entities relate to each other. Common relationship types:
- PARTICIPATED_IN (person → event)
- LED (person → organization/group)
- LOCATED_IN (entity → location)
- HAPPENED_IN (event → location/date)
- RELATED_TO (general association)
- PART_OF (entity → larger entity)
- CAUSED (entity → event/outcome)

Return ONLY valid JSON with this exact format:
{{"relationships": [{{"source": "Entity1", "target": "Entity2", "type": "RELATIONSHIP_TYPE"}}]}}

Rules:
- Only extract explicit relationships stated in the text
- Use exact entity names from the provided list
- Return valid JSON only, no explanation
"""

# Entity extraction from user query
QUERY_ENTITY_PROMPT = """Extract key entities from this user query.

Query: {query}

Return ONLY valid JSON with this exact format:
{{"entities": [{{"name": "Entity Name", "type": "PERSON|ORG|LOC|EVENT|CONCEPT|PRODUCT"}}]}}

Focus on entities that would be useful for searching a knowledge graph.
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
