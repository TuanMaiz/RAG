"""Graph retrieval using Neo4j knowledge graph.

This module queries the knowledge graph to find relevant documents
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


def graph_search(
    query: str,
    domain: str | None = None,
    k: int = 5,
    expand: bool = True,
    max_hops: int = 1,
) -> list[int]:
    """Search knowledge graph for relevant documents.

    Extracts entities from the query and finds documents that mention
    those entities or related entities (with expansion).

    Args:
        query: User's question
        domain: Optional domain filter (clapnq, cloud, fiqa, govt)
        k: Maximum number of documents to return
        expand: Whether to traverse relationships for entity expansion
        max_hops: Maximum relationship hops for expansion

    Returns:
        List of document IDs (integers) that match Qdrant point IDs
    """
    if not ENABLE_KG:
        return []

    # 1. Extract entities from query
    entities = extract_query_entities(query)
    if not entities:
        return []

    # Infer domain from first entity if not specified
    # Note: extract_query_entities doesn't add domain, so we rely on user input
    if domain is None:
        domain = _infer_domain_from_entities(entities)

    doc_ids = set()
    limit_per_entity = max(k, 10)  # Get more than k to allow for expansion

    # 2. For each entity, find documents
    for entity in entities:
        entity_name = entity["name"]

        # Direct mentions
        direct_docs = _find_documents_by_entity(
            entity_name, domain, limit=limit_per_entity
        )
        doc_ids.update(direct_docs)

        # 3. Expand to related entities if enabled
        if expand and domain:
            related = _expand_entities(entity_name, domain, max_hops)
            for related_entity in related[:3]:  # Limit expansion to top 3
                related_docs = _find_documents_by_entity(
                    related_entity, domain, limit=limit_per_entity
                )
                doc_ids.update(related_docs)

        if len(doc_ids) >= k * 2:  # Get enough candidates
            break

    # 4. Return top k as list
    return list(doc_ids)[:k]


def _infer_domain_from_entities(
    entities: list[dict[str, str]],
) -> str | None:
    """Infer domain from entity names using heuristics.

    Args:
        entities: List of extracted entities

    Returns:
        Inferred domain or None
    """
    entity_names = " ".join([e["name"].lower() for e in entities])

    # Simple keyword-based domain inference
    if any(keyword in entity_names for keyword in [
        "napoleon", "french revolution", "corsica", "waterloo", "france",
        "revolution", "battle", "empire", "1789", "1799"
    ]):
        return "clapnq"
    elif any(keyword in entity_names for keyword in [
        "ibm", "cloud", "cdn", "akamai", "delivery", "origin",
        "apple", "steve jobs", "cupertino", "california"
    ]):
        return "cloud"
    elif any(keyword in entity_names for keyword in [
        "stock", "financial", "market", "investment", "trading",
        "dividend", "portfolio", "investor", "sentiment", "conditions"
    ]):
        return "fiqa"
    elif any(keyword in entity_names for keyword in [
        "government", "agency", "service", "policy", "federal"
    ]):
        return "govt"

    return None


def _find_documents_by_entity(
    entity_name: str,
    domain: str | None = None,
    limit: int = 10,
) -> list[int]:
    """Find documents that mention a specific entity.

    Queries Neo4j for Document nodes connected to an Entity node
    via the MENTIONS relationship.

    Args:
        entity_name: Name of the entity to search for
        domain: Optional domain filter
        limit: Maximum number of documents to return

    Returns:
        List of document IDs (integers)
    """
    query = """
    MATCH (d:Document)-[:MENTIONS]->(e:Entity {name: $name})
    """

    params = {"name": entity_name}

    if domain:
        query += " WHERE e.domain = $domain"
        params["domain"] = domain

    query += " RETURN d.id"

    if limit:
        query += f" LIMIT {limit}"

    try:
        results = execute_query(query, params)
        return [int(r["d.id"]) for r in results if r.get("d.id") is not None]
    except Exception as e:
        print(f"Warning: Graph query failed for entity '{entity_name}': {e}")
        return []


def _expand_entities(
    entity_name: str,
    domain: str,
    max_hops: int = 2,
) -> list[str]:
    """Find related entities through relationship traversal.

    Expands the search by finding entities connected to the given entity
    via any relationship type, up to max_hops away.

    Args:
        entity_name: Name of the starting entity
        domain: Domain to constrain the search
        max_hops: Maximum relationship hops to traverse

    Returns:
        List of related entity names
    """
    query = f"""
    MATCH (start:Entity {{name: $name, domain: $domain}})-[*1..{max_hops}]-(related:Entity)
    WHERE related.domain = $domain AND related.name <> $name
    RETURN DISTINCT related.name
    LIMIT 5
    """

    params = {"name": entity_name, "domain": domain}

    try:
        results = execute_query(query, params)
        return [r["related.name"] for r in results if r.get("related.name") is not None]
    except Exception as e:
        print(f"Warning: Entity expansion failed for '{entity_name}': {e}")
        return []


def get_entity_context(
    entity_name: str,
    domain: str,
) -> dict[str, Any]:
    """Get detailed context about an entity from the knowledge graph.

    Retrieves the entity node along with its related entities and
    the documents that mention it.

    Args:
        entity_name: Name of the entity
        domain: Domain to constrain the search

    Returns:
        Dictionary with entity info, related entities, and document IDs
    """
    query = """
    MATCH (e:Entity {name: $name, domain: $domain})
    OPTIONAL MATCH (e)-[r]-(related:Entity)
    OPTIONAL MATCH (d:Document)-[:MENTIONS]->(e)
    RETURN e.name as name, e.type as type,
           collect(DISTINCT related.name)[0..5] as related_entities,
           collect(DISTINCT d.id) as document_ids
    """

    params = {"name": entity_name, "domain": domain}

    try:
        results = execute_query(query, params)
        if results:
            return {
                "name": results[0].get("name"),
                "type": results[0].get("type"),
                "related_entities": results[0].get("related_entities", []),
                "document_ids": results[0].get("document_ids", []),
            }
    except Exception as e:
        print(f"Warning: Failed to get entity context for '{entity_name}': {e}")

    return {}


# For testing
if __name__ == "__main__":
    # Test graph search
    print("Testing graph retrieval...")

    # Test 1: Search for Napoleon (should return clapnq documents)
    test_query = "Who was Napoleon Bonaparte?"
    print(f"\nQuery: {test_query}")
    doc_ids = graph_search(test_query, domain="clapnq", k=5)
    print(f"Found {len(doc_ids)} documents: {doc_ids}")

    # Test 2: Search with expansion
    print("\nTesting entity expansion...")
    related = _expand_entities("Napoleon", "clapnq", max_hops=2)
    print(f"Entities related to 'Napoleon': {related}")

    # Test 3: Get entity context
    print("\nTesting entity context retrieval...")
    context = get_entity_context("Napoleon", "clapnq")
    print(f"Context for 'Napoleon': {context}")
