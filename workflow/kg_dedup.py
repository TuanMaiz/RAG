"""In-memory entity and relationship deduplication before Neo4j storage.

Implements the deduplication strategy from documents/kg_pipeline_plan.md:
- Global and domain-specific alias resolution
- Entity merging by (name, type) key
- Relationship merging by (source, target, type) key
- Orphan edge filtering
- Bulk Neo4j upsert
"""

import os
from collections import defaultdict
from typing import Any

# Global aliases (domain-independent)
GLOBAL_ALIASES = {
    "napoleon": "napoleon bonaparte",
    "napoleon i": "napoleon bonaparte",
    "bonaparte": "napoleon bonaparte",
}

# Domain-specific aliases
DOMAIN_ALIASES = {
    "french revolution": {
        "the emperor": "napoleon bonaparte",
        "the king": "louis xvi",
        "the convention": "national convention",
    },
    "frenchrevolution": {
        "the emperor": "napoleon bonaparte",
        "the king": "louis xvi",
    },
    "ibm cloud": {
        "ibm": "international business machines",
        "the cloud": "ibm cloud",
    },
}


def resolve_alias(name: str, domain: str | None = None) -> str:
    """Resolve an entity name to its canonical form using aliases.

    Args:
        name: The entity name to resolve
        domain: Optional domain for domain-specific aliases

    Returns:
        The canonical name (or original if no alias found)
    """
    name_normalized = name.lower().strip()

    # Check domain-specific aliases first (more specific)
    if domain:
        domain_normalized = domain.lower().strip().replace(" ", "").replace("-", "")
        for key, aliases in DOMAIN_ALIASES.items():
            key_normalized = key.lower().replace(" ", "")
            if key_normalized in domain_normalized or domain_normalized in key_normalized:
                if name_normalized in aliases:
                    return aliases[name_normalized]

    # Check global aliases
    return GLOBAL_ALIASES.get(name_normalized, name)


def merge_entities_across_chunks(
    chunk_results: list[dict[str, Any]],
    domain: str | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Merge entities from multiple chunks, resolving aliases.

    Args:
        chunk_results: List of {chunk_id, entities: [...], relationships: [...]}
        domain: Optional domain for alias resolution

    Returns:
        Dict mapping (normalized_name, type) -> merged_entity dict
        where merged_entity has: name, type, description, seen_in_chunks, aliases
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    for chunk_result in chunk_results:
        chunk_id = chunk_result.get("chunk_id", "")

        for entity in chunk_result.get("entities", []):
            # Resolve alias to canonical name
            raw_name = entity.get("name", "").lower().strip()
            canonical_name = resolve_alias(raw_name, domain)
            entity_type = entity.get("type", "OTHER").upper()

            key = (canonical_name, entity_type)

            if key not in merged:
                merged[key] = {
                    "name": canonical_name,
                    "type": entity_type,
                    "description": entity.get("description", ""),
                    "seen_in_chunks": [chunk_id],
                    "aliases": set(),
                    "domain": domain or "unknown",
                }
            else:
                existing = merged[key]

                # Keep longer description
                if len(entity.get("description", "")) > len(existing.get("description", "")):
                    existing["description"] = entity.get("description", "")

                # Track chunks
                existing["seen_in_chunks"].append(chunk_id)

                # Track aliases (if raw name differs from canonical)
                if raw_name != canonical_name:
                    existing["aliases"].add(raw_name)

    # Convert sets to lists for JSON serialization
    for entity in merged.values():
        entity["aliases"] = list(entity["aliases"])
        entity["seen_in_chunks"] = list(set(entity["seen_in_chunks"]))  # Dedupe

    return merged


def merge_relationships_across_chunks(
    chunk_results: list[dict[str, Any]],
    domain: str | None = None,
    valid_entities: set[tuple[str, str]] | None = None,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Merge relationships from multiple chunks, resolving aliases.

    Args:
        chunk_results: List of {chunk_id, entities: [...], relationships: [...]}
        domain: Optional domain for alias resolution
        valid_entities: Set of (name, type) tuples for valid entities (orphan guard)

    Returns:
        Dict mapping (source, target, type) -> merged_relationship dict
    """
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}

    for chunk_result in chunk_results:
        chunk_id = chunk_result.get("chunk_id", "")

        for rel in chunk_result.get("relationships", []):
            # Resolve aliases for both source and target
            raw_source = rel.get("source", "").lower().strip()
            raw_target = rel.get("target", "").lower().strip()
            canonical_source = resolve_alias(raw_source, domain)
            canonical_target = resolve_alias(raw_target, domain)
            rel_type = rel.get("type", "RELATED_TO")

            key = (canonical_source, canonical_target, rel_type)

            # Orphan guard: skip if source/target not in valid entities
            if valid_entities:
                source_type = _infer_entity_type(canonical_source, chunk_result.get("entities", []))
                target_type = _infer_entity_type(canonical_target, chunk_result.get("entities", []))
                if (canonical_source, source_type) not in valid_entities:
                    continue
                if (canonical_target, target_type) not in valid_entities:
                    continue

            if key not in merged:
                merged[key] = {
                    "source": canonical_source,
                    "target": canonical_target,
                    "type": rel_type,
                    "description": rel.get("description", ""),
                    "seen_in_chunks": [chunk_id],
                    "domain": domain or "unknown",
                }
            else:
                existing = merged[key]

                # Keep longer description
                if len(rel.get("description", "")) > len(existing.get("description", "")):
                    existing["description"] = rel.get("description", "")

                existing["seen_in_chunks"].append(chunk_id)

    # Dedupe chunk lists
    for rel in merged.values():
        rel["seen_in_chunks"] = list(set(rel["seen_in_chunks"]))

    return merged


def _infer_entity_type(name: str, entities: list[dict]) -> str:
    """Infer entity type from a list of entities."""
    for e in entities:
        if e.get("name", "").lower().strip() == name.lower().strip():
            return e.get("type", "OTHER")
    return "OTHER"


def bulk_upsert_to_neo4j(
    merged_entities: dict[tuple[str, str], dict[str, Any]],
    merged_relationships: dict[tuple[str, str, str], dict[str, Any]],
    domain: str | None = None,
) -> bool:
    """Bulk upsert merged entities and relationships to Neo4j.

    Args:
        merged_entities: Dict from merge_entities_across_chunks
        merged_relationships: Dict from merge_relationships_across_chunks
        domain: Optional domain for context

    Returns:
        True if successful, False otherwise
    """
    from graph_stores.neo4j_client import execute_query

    try:
        # Batch upsert entities using UNWIND
        entity_params = []
        for (name, entity_type), entity_data in merged_entities.items():
            entity_params.append({
                "name": name,
                "type": entity_type,
                "description": entity_data.get("description", ""),
                "domain": domain or entity_data.get("domain", "unknown"),
                "aliases": entity_data.get("aliases", []),
            })

        if entity_params:
            entity_query = """
            UNWIND $entities AS entity
            MERGE (e:Entity {name: entity.name, type: entity.type})
            ON CREATE SET e.description = entity.description, e.domain = entity.domain, e.aliases = entity.aliases
            ON MATCH SET
                e.description = CASE
                    WHEN length(entity.description) > length(e.description)
                    THEN entity.description
                    ELSE e.description
                END,
                e.aliases = CASE
                    WHEN size(entity.aliases) > 0 THEN COALESCE(e.aliases, []) + entity.aliases
                    ELSE e.aliases
                END
            """
            execute_query(entity_query, {"entities": entity_params})

        # Batch upsert relationships
        rel_params = []
        for (source, target, rel_type), rel_data in merged_relationships.items():
            rel_params.append({
                "source": source,
                "target": target,
                "type": rel_type,
                "description": rel_data.get("description", ""),
            })

        if rel_params:
            rel_query = """
            UNWIND $relationships AS rel
            MATCH (s:Entity {name: rel.source})
            MATCH (t:Entity {name: rel.target})
            MERGE (s)-[r:RELATED_TO {type: rel.type}]->(t)
            ON CREATE SET r.description = rel.description
            ON MATCH SET
                r.description = CASE
                    WHEN length(rel.description) > length(r.description)
                    THEN rel.description
                    ELSE r.description
                END
            """
            execute_query(rel_query, {"relationships": rel_params})

        return True

    except Exception as e:
        from utils.logging_config import get_logger
        logger = get_logger(__name__)
        logger.error("Bulk upsert to Neo4j failed: %s", e)
        return False


def dedup_and_store(
    chunk_results: list[dict[str, Any]],
    domain: str | None = None,
) -> dict[str, Any]:
    """Complete deduplication and storage pipeline.

    Args:
        chunk_results: List of {chunk_id, entities: [...], relationships: [...]}
        domain: Optional domain for alias resolution

    Returns:
        Dict with stats about deduplication
    """
    # Merge entities
    merged_entities = merge_entities_across_chunks(chunk_results, domain)

    # Get valid entity keys for orphan guard
    valid_entities = set(merged_entities.keys())

    # Merge relationships with orphan guard
    merged_relationships = merge_relationships_across_chunks(
        chunk_results, domain, valid_entities
    )

    # Bulk upsert to Neo4j
    success = bulk_upsert_to_neo4j(merged_entities, merged_relationships, domain)

    return {
        "success": success,
        "entity_count": len(merged_entities),
        "relationship_count": len(merged_relationships),
        "domain": domain or "unknown",
    }


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test alias resolution
    print("Testing alias resolution...")
    assert resolve_alias("Napoleon", "french revolution") == "napoleon bonaparte"
    assert resolve_alias("the emperor", "french revolution") == "napoleon bonaparte"
    assert resolve_alias("unknown") == "unknown"
    print("  Alias resolution: OK")

    # Test entity merging
    print("\nTesting entity merging...")
    chunk_results = [
        {
            "chunk_id": "chunk-1",
            "entities": [
                {"name": "Napoleon", "type": "PERSON", "description": "French general"},
                {"name": "French Army", "type": "ORGANIZATION", "description": "Military force"},
            ],
            "relationships": [],
        },
        {
            "chunk_id": "chunk-2",
            "entities": [
                {"name": "Napoleon Bonaparte", "type": "PERSON", "description": "Emperor of the French"},
                {"name": "the emperor", "type": "PERSON", "description": "Ruler of France"},
            ],
            "relationships": [],
        },
    ]

    merged = merge_entities_across_chunks(chunk_results, "french revolution")
    print(f"  Merged {len(merged)} unique entities")
    for key, entity in merged.items():
        print(f"    {key}: {entity['name']} (aliases: {entity['aliases']})")

    # Should have 2 unique entities (Napoleon + French Army)
    assert len(merged) == 2
    assert ("napoleon bonaparte", "PERSON") in merged
    assert merged[("napoleon bonaparte", "PERSON")]["description"] == "Emperor of the French"
    print("  Entity merging: OK")

    print("\nAll tests passed!")
