"""Knowledge Graph entity and relationship extraction using LLM.

This module extracts entities and relationships from text using LLM prompts.
Uses LightRAG-inspired delimited format to avoid JSON parsing issues.
"""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from utils.logging_config import get_logger
from workflow.prompts import (
    TUPLE_DELIMITER,
    COMPLETION_DELIMITER,
    format_entity_prompt,
    format_relationship_prompt,
    format_query_prompt,
)

load_dotenv()

logger = get_logger(__name__)

# LLM for extraction (can use cheaper/faster model)
# max_tokens=4096 to handle longer entity lists
_extraction_llm = ChatOpenAI(
    model=os.getenv("OPENAI_KG_MODEL", os.getenv("OPENAI_LLM_MODEL", "openai/gpt-4o-mini")),
    temperature=0,
    max_tokens=4096,
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


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

        # Check for completion signal
        if COMPLETION_DELIMITER in line:
            completion_found = True
            break

        # Split by delimiter
        parts = line.split(TUPLE_DELIMITER)

        if len(parts) < 2:
            continue

        entry_type = parts[0].strip()

        if entry_type == "entity" and len(parts) >= 4:
            # entity{delimiter}name{delimiter}type{delimiter}description
            entities.append({
                "name": parts[1].strip(),
                "type": parts[2].strip().upper(),
                "description": parts[3].strip() if len(parts) > 3 else "",
            })
        elif entry_type == "relation" and len(parts) >= 5:
            # relation{delimiter}source{delimiter}target{delimiter}keywords{delimiter}description
            relationships.append({
                "source": parts[1].strip(),
                "target": parts[2].strip(),
                "type": parts[3].strip(),  # keywords stored as type
                "description": parts[4].strip() if len(parts) > 4 else "",
            })

    return {
        "entities": entities,
        "relationships": relationships,
        "complete": completion_found,
    }


def extract_entities(text: str, domain: str = None) -> list[dict[str, str]]:
    """Extract entities from text using LLM.

    Args:
        text: Text to extract entities from
        domain: Optional domain hint (clapnq, cloud, fiqa, govt)

    Returns:
        List of entities with name, type, and description
    """
    if not text or len(text.strip()) < 10:
        return []

    prompt = format_entity_prompt(text)

    try:
        response = _extraction_llm.invoke(prompt)
        content = response.content.strip()

        # Parse delimited output
        result = _parse_delimited_output(content)
        entities = result.get("entities", [])

        # Add domain if provided
        for entity in entities:
            if domain:
                entity["domain"] = domain

        if not result.get("complete"):
            logger.warning("Entity extraction may be incomplete (no completion delimiter)")

        return entities

    except Exception as e:
        logger.warning("Entity extraction failed: %s", e)
        return []


def extract_relationships(
    text: str,
    entities: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Extract relationships between entities using LLM.

    Args:
        text: Text to extract relationships from
        entities: List of entities found in the text

    Returns:
        List of relationships with source, target, and type
    """
    if not entities or len(entities) < 2:
        return []

    prompt = format_relationship_prompt(text, entities)

    try:
        response = _extraction_llm.invoke(prompt)
        content = response.content.strip()

        # Parse delimited output
        result = _parse_delimited_output(content)
        relationships = result.get("relationships", [])

        if not result.get("complete"):
            logger.warning("Relationship extraction may be incomplete (no completion delimiter)")

        return relationships

    except Exception as e:
        logger.warning("Relationship extraction failed: %s", e)
        return []


def extract_graph_data(
    text: str,
    domain: str = None,
) -> dict[str, Any]:
    """Extract both entities and relationships from text.

    Args:
        text: Text to extract from
        domain: Optional domain hint

    Returns:
        Dictionary with entities and relationships
    """
    entities = extract_entities(text, domain)

    if not entities:
        return {"entities": [], "relationships": []}

    relationships = extract_relationships(text, entities)

    return {
        "entities": entities,
        "relationships": relationships,
    }


def extract_query_entities(query: str) -> list[dict[str, str]]:
    """Extract entities from a user query.

    Args:
        query: User's question

    Returns:
        List of entities found in the query
    """
    if not query:
        return []

    prompt = format_query_prompt(query)

    try:
        response = _extraction_llm.invoke(prompt)
        content = response.content.strip()

        # Parse delimited output
        result = _parse_delimited_output(content)
        entities = result.get("entities", [])

        return entities

    except Exception as e:
        logger.warning("Query entity extraction failed: %s", e)
        return []


def extract_entities_batch(
    texts: list[str],
    domain: str = None,
) -> list[list[dict[str, str]]]:
    """Extract entities from multiple texts.

    Args:
        texts: List of texts to process
        domain: Optional domain hint

    Returns:
        List of entity lists (one per input text)
    """
    results = []
    for text in texts:
        entities = extract_entities(text, domain)
        results.append(entities)
    return results


# For testing
if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    Napoleon Bonaparte led the French army during the French Revolution.
    He was born in Corsica and later became Emperor of France.
    The Battle of Waterloo in 1815 marked his final defeat.
    """

    print("Testing entity extraction...")
    result = extract_graph_data(sample_text, domain="clapnq")

    print(f"\nFound {len(result['entities'])} entities:")
    for entity in result["entities"]:
        print(f"  - {entity['name']} ({entity['type']})")
        if entity.get('description'):
            print(f"    {entity['description']}")

    print(f"\nFound {len(result['relationships'])} relationships:")
    for rel in result["relationships"]:
        print(f"  - {rel['source']} -> {rel['target']} ({rel['type']})")
