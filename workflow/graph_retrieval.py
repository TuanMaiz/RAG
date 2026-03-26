"""Graph retrieval using Neo4j knowledge graph.

This module queries the knowledge graph to find relevant chunks
based on entities extracted from user queries.
"""

import os
from typing import Any

from dotenv import load_dotenv

from graph_stores.neo4j_client import execute_query
from workflow.kg_extraction import extract_query_entities

load_dotenv()

# KG retrieval flag
ENABLE_KG = os.getenv("ENABLE_KG", "true").lower() == "true"

# Schema version: 1 = old (name, title), 2 = new (name, type)
SCHEMA_VERSION = os.getenv("KG_SCHEMA_VERSION", "1")


def graph_search(
    query: str,
    title: str | None = None,
    k: int = 5,
    expand: bool = True,
    max_hops: int = 1,
) -> list[int]:
    """Search knowledge graph for relevant chunks.

    Extracts entities from the query and finds chunks that mention
    those entities or related entities (with expansion).

    Args:
        query: User's question
        title: Optional title filter (e.g., "French Revolution")
        k: Maximum number of chunk IDs to return
        expand: Whether to traverse relationships for entity expansion
        max_hops: Maximum relationship hops for expansion

    Returns:
        List of chunk IDs (integers) that match Qdrant point IDs
    """
    if not ENABLE_KG:
        return []

    # 1. Extract entities from query
    entities = extract_query_entities(query)
    if not entities:
        return []

    chunk_ids = set()
    limit_per_entity = max(k, 10)  # Get more than k to allow for expansion

    # 2. For each entity, find chunks
    for entity in entities:
        entity_name = entity["name"]

        # Direct mentions
        direct_chunks = _find_chunks_by_entity(
            entity_name, title, limit=limit_per_entity
        )
        chunk_ids.update(direct_chunks)

        # 3. Expand to related entities if title is specified
        if expand and title:
            related = _expand_entities(entity_name, title, max_hops)
            for related_entity in related[:3]:  # Limit expansion to top 3
                related_chunks = _find_chunks_by_entity(
                    related_entity, title, limit=limit_per_entity
                )
                chunk_ids.update(related_chunks)

        if len(chunk_ids) >= k * 2:  # Get enough candidates
            break

    # 4. Return top k as list
    return list(chunk_ids)[:k]


def _find_chunks_by_entity(
    entity_name: str,
    title: str | None = None,
    limit: int = 10,
) -> list[int]:
    """Find chunks that mention a specific entity.

    Queries Neo4j for Chunk nodes connected to an Entity node
    via the MENTIONS relationship.

    Args:
        entity_name: Name of the entity to search for
        title: Optional title filter (used for domain scoping in schema v1)
        limit: Maximum number of chunks to return

    Returns:
        List of chunk IDs (integers)
    """
    # Schema v1: (name, title) as key
    if SCHEMA_VERSION == "1":
        query = "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {name: $name})"
        params = {"name": entity_name}

        if title:
            query += " WHERE e.title = $title"
            params["title"] = title
    else:
        # Schema v2: (name, type) as key, title becomes optional domain filter
        query = "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {name: $name})"
        params = {"name": entity_name}

        if title:
            # Optional: filter by domain or aliases
            query += " WHERE e.domain = $title OR $title IN e.aliases"
            params["title"] = title

    query += " RETURN c.id"

    if limit:
        query += f" LIMIT {limit}"

    try:
        results = execute_query(query, params)
        return [int(r["c.id"]) for r in results if r.get("c.id") is not None]
    except Exception as e:
        print(f"Warning: Graph query failed for entity '{entity_name}': {e}")
        return []


def _expand_entities(
    entity_name: str,
    title: str,
    max_hops: int = 2,
) -> list[str]:
    """Find related entities through relationship traversal.

    Expands the search by finding entities connected to the given entity
    via any relationship type, up to max_hops away.

    Args:
        entity_name: Name of the starting entity
        title: Title/domain to constrain the search
        max_hops: Maximum relationship hops to traverse

    Returns:
        List of related entity names
    """
    # Schema v1: (name, title) as key
    if SCHEMA_VERSION == "1":
        query = f"""
        MATCH (start:Entity {{name: $name, title: $title}})-[*1..{max_hops}]-(related:Entity)
        WHERE related.title = $title AND related.name <> $name
        RETURN DISTINCT related.name
        LIMIT 5
        """
    else:
        # Schema v2: (name, type) as key, optional domain filter
        query = f"""
        MATCH (start:Entity {{name: $name}})-[*1..{max_hops}]-(related:Entity)
        WHERE related.name <> $name
        """
        if title:
            query += " AND (related.domain = $title OR $title IN related.aliases)"

        query += """
        RETURN DISTINCT related.name
        LIMIT 5
        """

    params = {"name": entity_name, "title": title}

    try:
        results = execute_query(query, params)
        return [r["related.name"] for r in results if r.get("related.name") is not None]
    except Exception as e:
        print(f"Warning: Entity expansion failed for '{entity_name}': {e}")
        return []


def get_entity_context(
    entity_name: str,
    title: str,
) -> dict[str, Any]:
    """Get detailed context about an entity from the knowledge graph.

    Retrieves the entity node along with its related entities and
    the chunks that mention it.

    Args:
        entity_name: Name of the entity
        title: Title/domain to constrain the search

    Returns:
        Dictionary with entity info, related entities, and chunk IDs
    """
    # Schema v1: (name, title) as key
    if SCHEMA_VERSION == "1":
        query = """
        MATCH (e:Entity {name: $name, title: $title})
        OPTIONAL MATCH (e)-[r]-(related:Entity)
        OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
        RETURN e.name as name, e.type as type,
               collect(DISTINCT related.name)[0..5] as related_entities,
               collect(DISTINCT c.id) as chunk_ids
        """
        params = {"name": entity_name, "title": title}
    else:
        # Schema v2: (name, type) as key, title is optional domain filter
        query = """
        MATCH (e:Entity {name: $name})
        OPTIONAL MATCH (e)-[r]-(related:Entity)
        OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
        """
        if title:
            query += " WHERE e.domain = $title OR $title IN e.aliases"

        query += """
        RETURN e.name as name, e.type as type,
               collect(DISTINCT related.name)[0..5] as related_entities,
               collect(DISTINCT c.id) as chunk_ids
        """
        params = {"name": entity_name, "title": title}

    try:
        results = execute_query(query, params)
        if results:
            return {
                "name": results[0].get("name"),
                "type": results[0].get("type"),
                "related_entities": results[0].get("related_entities", []),
                "chunk_ids": results[0].get("chunk_ids", []),
            }
    except Exception as e:
        print(f"Warning: Failed to get entity context for '{entity_name}': {e}")

    return {}


# For testing
if __name__ == "__main__":
    # Test graph search
    print("Testing graph retrieval...")

    # Test: Search for Napoleon
    test_query = "Who was Napoleon Bonaparte?"
    print(f"\nQuery: {test_query}")
    chunk_ids = graph_search(test_query, title="French Revolution", k=5)
    print(f"Found {len(chunk_ids)} chunks: {chunk_ids}")

    # Test: Entity expansion
    print("\nTesting entity expansion...")
    related = _expand_entities("Napoleon", "French Revolution", max_hops=2)
    print(f"Entities related to 'Napoleon': {related}")

    # Test: Entity context
    print("\nTesting entity context retrieval...")
    context = get_entity_context("Napoleon", "French Revolution")
    print(f"Context for 'Napoleon': {context}")
