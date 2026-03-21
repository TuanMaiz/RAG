"""Knowledge Graph entity and relationship extraction using LLM.

This module extracts entities and relationships from text using LLM prompts.
"""

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from workflow.prompts import (
    ENTITY_EXTRACTION_PROMPT,
    RELATIONSHIP_EXTRACTION_PROMPT,
    QUERY_ENTITY_PROMPT,
)

load_dotenv()

# LLM for extraction (can use cheaper/faster model)
_extraction_llm = ChatOpenAI(
    model=os.getenv("OPENAI_LLM_MODEL", "openai/gpt-4o-mini"),
    temperature=0,
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


def extract_entities(text: str, domain: str = None) -> list[dict[str, str]]:
    """Extract entities from text using LLM.

    Args:
        text: Text to extract entities from
        domain: Optional domain hint (clapnq, cloud, fiqa, govt)

    Returns:
        List of entities with name and type
    """
    if not text or len(text.strip()) < 10:
        return []

    prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)

    try:
        response = _extraction_llm.invoke(prompt)
        content = response.content.strip()

        # Parse JSON response
        result = json.loads(content)
        entities = result.get("entities", [])

        # Add domain if provided
        if domain:
            for entity in entities:
                entity["domain"] = domain

        return entities

    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse entity extraction JSON: {e}")
        print(f"Response was: {content[:200]}")
        return []
    except Exception as e:
        print(f"Warning: Entity extraction failed: {e}")
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

    # Format entities for the prompt
    entity_list = ", ".join([f"{e['name']} ({e.get('type', 'UNKNOWN')})" for e in entities])

    prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
        entities=entity_list,
        text=text,
    )

    try:
        response = _extraction_llm.invoke(prompt)
        content = response.content.strip()

        # Parse JSON response
        result = json.loads(content)
        return result.get("relationships", [])

    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse relationship extraction JSON: {e}")
        print(f"Response was: {content[:200]}")
        return []
    except Exception as e:
        print(f"Warning: Relationship extraction failed: {e}")
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

    prompt = QUERY_ENTITY_PROMPT.format(query=query)

    try:
        response = _extraction_llm.invoke(prompt)
        content = response.content.strip()

        result = json.loads(content)
        return result.get("entities", [])

    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse query entity extraction JSON: {e}")
        return []
    except Exception as e:
        print(f"Warning: Query entity extraction failed: {e}")
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

    print(f"\nFound {len(result['relationships'])} relationships:")
    for rel in result["relationships"]:
        print(f"  - {rel['source']} -> {rel['target']} ({rel['type']})")
