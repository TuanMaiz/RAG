"""Knowledge Graph entity and relationship extraction using LLM.

Supports two extraction modes:
1. Structured output (new): Uses Pydantic models with OpenAI's structured output
2. Delimited format (legacy): LightRAG-style delimited parsing

The structured output mode is recommended as it eliminates parsing errors.
"""

import os
import time
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from utils.logging_config import get_logger
from workflow.kg_cache import ExtractionCache
from workflow.kg_models import Entity, KnowledgeGraph, Relationship
from workflow.prompts import (
    COMPLETION_DELIMITER,
    TUPLE_DELIMITER,
    format_batch_prompt,
    format_entity_prompt,
    format_gleaning_prompt,
    format_query_prompt,
    format_structured_entity_prompt,
    get_entity_types_for_domain,
)

load_dotenv()

logger = get_logger(__name__)

# Feature flag: Use structured output instead of delimited parsing
USE_STRUCTURED_OUTPUT: bool = os.getenv("KG_USE_STRUCTURED_OUTPUT", "true").lower() == "true"

# Feature flag: Enable gleaning pass
ENABLE_GLEANING: bool = os.getenv("KG_ENABLE_GLEANING", "true").lower() == "true"

# Gleaning max tokens - skip gleaning if context exceeds this
GLEANING_MAX_TOKENS: int = int(os.getenv("KG_GLEANING_MAX_TOKENS", "2000"))

# LLM for extraction (can use cheaper/faster model)
_extraction_llm = ChatOpenAI(
    model=os.getenv("OPENAI_KG_MODEL", os.getenv("OPENAI_LLM_MODEL", "openai/gpt-4o-mini")),
    temperature=0,
    max_tokens=4096,
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# Structured output LLM - wraps the base LLM with structured output
# Use getattr for safer access in case with_structured_output is not available
structured_llm: Any = None
if USE_STRUCTURED_OUTPUT:
    try:
        structured_llm = _extraction_llm.with_structured_output(KnowledgeGraph)
    except (AttributeError, TypeError) as e:
        logger.warning("Structured output not available: %s. Falling back to delimited format.", e)
        USE_STRUCTURED_OUTPUT = False
        structured_llm = None


# ============================================================================
# Legacy: Delimited Parsing (kept for backward compatibility)
# ============================================================================

def _parse_delimited_output(content: str) -> dict[str, Any]:
    """Parse LightRAG-style delimited output.

    Expected format:
    entity{delimiter}name{delimiter}type{delimiter}description
    relation{delimiter}source{delimiter}target{delimiter}keywords{delimiter}description
    ...
    {completion_delimiter}
    """
    entities = []
    relationships = []

    lines = content.strip().split('\n')
    completion_found = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if COMPLETION_DELIMITER in line:
            completion_found = True
            break

        parts = line.split(TUPLE_DELIMITER)

        if len(parts) < 2:
            continue

        entry_type = parts[0].strip()

        if entry_type == "entity" and len(parts) >= 4:
            entities.append({
                "name": parts[1].strip(),
                "type": parts[2].strip().upper(),
                "description": parts[3].strip() if len(parts) > 3 else "",
            })
        elif entry_type == "relation" and len(parts) >= 5:
            relationships.append({
                "source": parts[1].strip(),
                "target": parts[2].strip(),
                "type": parts[3].strip(),
                "description": parts[4].strip() if len(parts) > 4 else "",
            })

    return {
        "entities": entities,
        "relationships": relationships,
        "complete": completion_found,
    }


def _parse_batch_output(content: str, doc_ids: list[int]) -> dict[int, dict[str, Any]]:
    """Parse batch extraction output with doc_id prefixes."""
    results = {doc_id: {"entities": [], "relationships": []} for doc_id in doc_ids}

    lines = content.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if COMPLETION_DELIMITER in line:
            break

        doc_match = None
        for doc_id in doc_ids:
            prefix = f"doc_id:{doc_id}{TUPLE_DELIMITER}"
            if line.startswith(prefix):
                doc_match = doc_id
                line = line[len(prefix):]
                break

        if doc_match is None:
            continue

        parts = line.split(TUPLE_DELIMITER)

        if len(parts) < 3:
            continue

        entry_type = parts[0].strip()

        if entry_type == "entity" and len(parts) >= 4:
            results[doc_match]["entities"].append({
                "name": parts[1].strip(),
                "type": parts[2].strip().upper(),
                "description": parts[3].strip() if len(parts) > 3 else "",
            })
        elif entry_type == "relation" and len(parts) >= 5:
            results[doc_match]["relationships"].append({
                "source": parts[1].strip(),
                "target": parts[2].strip(),
                "type": parts[3].strip(),
                "description": parts[4].strip() if len(parts) > 4 else "",
            })

    return results


# ============================================================================
# Structured Output Extraction (New)
# ============================================================================

def _normalize_entity(entity: Entity) -> dict[str, str]:
    """Normalize an entity from Pydantic model to dict."""
    return {
        "name": entity.name.lower().strip(),
        "type": entity.type.upper(),
        "description": entity.description,
    }


def _normalize_relationship(rel: Relationship) -> dict[str, str]:
    """Normalize a relationship from Pydantic model to dict."""
    return {
        "source": rel.source.lower().strip(),
        "target": rel.target.lower().strip(),
        "type": rel.type,
        "description": rel.description,
    }


def _merge_extraction_results(
    initial: dict[str, Any],
    gleaned: dict[str, Any],
) -> dict[str, Any]:
    """Merge initial and gleaning extraction results.

    Args:
        initial: First extraction result
        gleaned: Second extraction result (gleaning pass)

    Returns:
        Merged result with unique entities and relationships
    """
    # Track by (name, type) for entities, (source, target, type) for relationships
    seen_entities = {}
    seen_relationships = {}

    # Add initial results
    for entity in initial.get("entities", []):
        key = (entity["name"], entity["type"])
        seen_entities[key] = entity

    for rel in initial.get("relationships", []):
        key = (rel["source"], rel["target"], rel["type"])
        seen_relationships[key] = rel

    # Add gleaned results (longer description wins)
    for entity in gleaned.get("entities", []):
        key = (entity["name"], entity["type"])
        if key in seen_entities:
            # Keep longer description
            if len(entity.get("description", "")) > len(seen_entities[key].get("description", "")):
                seen_entities[key] = entity
        else:
            seen_entities[key] = entity

    for rel in gleaned.get("relationships", []):
        key = (rel["source"], rel["target"], rel["type"])
        if key in seen_relationships:
            if len(rel.get("description", "")) > len(seen_relationships[key].get("description", "")):
                seen_relationships[key] = rel
        else:
            seen_relationships[key] = rel

    return {
        "entities": list(seen_entities.values()),
        "relationships": list(seen_relationships.values()),
    }


# ============================================================================
# Main Extraction Functions
# ============================================================================

def extract_entities(
    text: str,
    domain: str | None = None,
    enable_gleaning: bool | None = None,
    cache: ExtractionCache | None = None,
) -> dict[str, Any]:
    """Extract entities and relationships from text.

    Args:
        text: Text to extract from
        domain: Optional domain for domain-aware entity types
        enable_gleaning: Override default gleaning setting
        cache: Optional cache for storing results

    Returns:
        Dictionary with {"entities": [...], "relationships": [...]}
    """
    if not text or len(text.strip()) < 10:
        return {"entities": [], "relationships": []}

    # Use default gleaning setting if not overridden
    if enable_gleaning is None:
        enable_gleaning = ENABLE_GLEANING

    # Structured output path
    if USE_STRUCTURED_OUTPUT:
        return _extract_structured(text, domain, enable_gleaning, cache)
    else:
        # Legacy delimited format path
        return _extract_delimited(text, domain, enable_gleaning, cache)


def _extract_structured(
    text: str,
    domain: str | None,
    enable_gleaning: bool,
    cache: ExtractionCache | None,
) -> dict[str, Any]:
    """Extract using structured output (Pydantic models)."""
    if structured_llm is None:
        logger.warning("Structured output LLM not available, falling back to delimited")
        return _extract_delimited(text, domain, enable_gleaning, cache)

    # Build prompts
    system_prompt, user_prompt = format_structured_entity_prompt(text, domain)
    cache_key = cache.make_key(text, system_prompt) if cache else None

    # Check cache
    if cache and cache_key:
        cached = cache.get(cache_key)
        if cached:
            return cached

    try:
        # Initial extraction
        result = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        # Normalize entities and relationships
        entities = [_normalize_entity(e) for e in result.entities]
        relationships = [_normalize_relationship(r) for r in result.relationships]

        initial_result = {"entities": entities, "relationships": relationships}

        # Gleaning pass
        if enable_gleaning:
            gleaned_result = _gleaning_pass_structured(
                text, system_prompt, initial_result
            )
            initial_result = _merge_extraction_results(initial_result, gleaned_result)

        # Cache result
        if cache and cache_key:
            cache.set(cache_key, initial_result, time.time())

        return initial_result

    except Exception as e:
        logger.warning("Structured extraction failed: %s", e)
        return {"entities": [], "relationships": []}


def _gleaning_pass_structured(
    text: str,
    system_prompt: str,
    initial_result: dict[str, Any],
) -> dict[str, Any]:
    """Perform gleaning pass using structured output."""
    if structured_llm is None:
        return {"entities": [], "relationships": []}

    try:
        gleaning_prompt = format_gleaning_prompt(text, initial_result)

        result = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": gleaning_prompt}
        ])

        return {
            "entities": [_normalize_entity(e) for e in result.entities],
            "relationships": [_normalize_relationship(r) for r in result.relationships],
        }
    except Exception as e:
        logger.warning("Gleaning pass failed: %s", e)
        return {"entities": [], "relationships": []}


def _extract_delimited(
    text: str,
    domain: str | None,
    enable_gleaning: bool,
    cache: ExtractionCache | None,
) -> dict[str, Any]:
    """Extract using legacy delimited format."""
    prompt = format_entity_prompt(text)
    cache_key = cache.make_key(text, prompt) if cache else None

    # Check cache
    if cache and cache_key:
        cached = cache.get(cache_key)
        if cached:
            return cached

    try:
        response = _extraction_llm.invoke(prompt)
        content = response.content.strip()
        result = _parse_delimited_output(content)

        output = {
            "entities": result.get("entities", []),
            "relationships": result.get("relationships", []),
        }

        # Cache result
        if cache and cache_key:
            cache.set(cache_key, output, time.time())

        return output

    except Exception as e:
        logger.warning("Delimited extraction failed: %s", e)
        return {"entities": [], "relationships": []}


# ============================================================================
# Convenience Functions
# ============================================================================

def extract_graph_data(
    text: str,
    domain: str | None = None,
) -> dict[str, Any]:
    """Extract both entities and relationships from text.

    Args:
        text: Text to extract from
        domain: Optional domain for domain-aware entity types

    Returns:
        Dictionary with {"entities": [...], "relationships": [...]}
    """
    return extract_entities(text, domain=domain)


def extract_query_entities(query: str) -> list[dict[str, str]]:
    """Extract entities from a user query.

    Args:
        query: User's question

    Returns:
        List of entities found in the query
    """
    if not query:
        return []

    try:
        if USE_STRUCTURED_OUTPUT and structured_llm is not None:
            # Use structured output for queries too
            system_prompt, user_prompt = format_structured_entity_prompt(query, None)
            result = structured_llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            return [_normalize_entity(e) for e in result.entities]
        else:
            prompt = format_query_prompt(query)
            response = _extraction_llm.invoke(prompt)
            content = response.content.strip()
            result = _parse_delimited_output(content)
            return result.get("entities", [])

    except Exception as e:
        logger.warning("Query entity extraction failed: %s", e)
        return []


def extract_graph_data_batch(
    texts_with_ids: list[tuple[int, str]] | list[tuple[int, str, str]],
    batch_size: int | None = None,
    domain: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Extract entities and relationships from multiple texts.

    Note: Gleaning is disabled for batch extraction for performance.

    Args:
        texts_with_ids: List of (chunk_id, text) or (chunk_id, text, title) tuples
        batch_size: Number of texts to process per LLM call (default from env: 10)
        domain: Optional domain for domain-aware entity types

    Returns:
        Dictionary mapping chunk_id to {"entities": [...], "relationships": [...]}
    """
    if batch_size is None:
        batch_size = int(os.getenv("KG_BATCH_SIZE", "10"))

    # Normalize to (doc_id, text) tuples
    normalized = []
    for item in texts_with_ids:
        if len(item) >= 2:
            doc_id = item[0]
            text = item[1]
            normalized.append((doc_id, text))

    results = {}
    doc_ids = [doc_id for doc_id, _ in normalized]

    # Initialize empty results for all docs
    for doc_id in doc_ids:
        results[doc_id] = {"entities": [], "relationships": []}

    # Process each text individually for structured output
    # (Batch processing with structured output is more complex)
    if USE_STRUCTURED_OUTPUT:
        for doc_id, text in normalized:
            try:
                extracted = extract_entities(text, domain=domain, enable_gleaning=False)
                results[doc_id] = extracted
            except Exception as e:
                logger.warning("Extraction failed for doc %d: %s", doc_id, e)
    else:
        # Use legacy batch processing
        for i in range(0, len(normalized), batch_size):
            batch = normalized[i : i + batch_size]

            try:
                prompt = format_batch_prompt(batch)
                response = _extraction_llm.invoke(prompt)
                content = response.content.strip()

                batch_results = _parse_batch_output(content, [doc_id for doc_id, _ in batch])

                for doc_id, data in batch_results.items():
                    results[doc_id] = data

            except Exception as e:
                logger.warning("Batch extraction failed (docs %d-%d): %s", i, i + len(batch) - 1, e)

    return results


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    Napoleon Bonaparte led the French army during the French Revolution.
    He was born in Corsica and later became Emperor of France.
    The Battle of Waterloo in 1815 marked his final defeat.
    """

    print(f"Testing entity extraction (structured output: {USE_STRUCTURED_OUTPUT})...")
    result = extract_graph_data(sample_text, domain="french revolution")

    print(f"\nFound {len(result['entities'])} entities:")
    for entity in result["entities"]:
        print(f"  - {entity['name']} ({entity['type']})")
        if entity.get('description'):
            print(f"    {entity['description']}")

    print(f"\nFound {len(result['relationships'])} relationships:")
    for rel in result["relationships"]:
        print(f"  - {rel['source']} -> {rel['target']} ({rel['type']})")
