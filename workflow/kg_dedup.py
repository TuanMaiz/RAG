"""In-memory entity and relationship deduplication before Neo4j storage.

Implements the deduplication strategy from documents/kg_pipeline_plan.md:
- Global and domain-specific alias resolution
- Entity merging by (name, type) key
- Relationship merging by (source, target, type) key
- Orphan edge filtering
- Bulk Neo4j upsert with retry logic
"""

import time
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


def _entity_type_to_label(entity_type: str) -> str:
    """Convert entity type to valid Neo4j label.

    Args:
        entity_type: Entity type string (e.g., "PERSON", "ORG", "LOCATION")

    Returns:
        Valid Neo4j label (e.g., "Person", "Organization", "Location")
    """
    # Mapping of common entity types to labels
    type_mapping = {
        "PERSON": "Person",
        "ORG": "Organization",
        "ORGANIZATION": "Organization",
        "LOC": "Location",
        "LOCATION": "Location",
        "EVENT": "Event",
        "CONCEPT": "Concept",
        "PRODUCT": "Product",
        "DATE": "Date",
        "OTHER": "Other",
    }

    # Normalize input
    normalized = entity_type.upper().strip()

    # Return mapped label or fall back to capitalized version
    return type_mapping.get(normalized, entity_type.capitalize())


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
                    "keywords": set(entity.get("keywords", [])),  # NEW: Track keywords
                    "seen_in_chunks": [chunk_id],
                    "aliases": set(),
                    "domain": domain or "unknown",
                }
            else:
                existing = merged[key]

                # Keep longer description
                if len(entity.get("description", "")) > len(existing.get("description", "")):
                    existing["description"] = entity.get("description", "")

                # Merge keywords
                existing["keywords"].update(entity.get("keywords", []))

                # Track chunks
                existing["seen_in_chunks"].append(chunk_id)

                # Track aliases (if raw name differs from canonical)
                if raw_name != canonical_name:
                    existing["aliases"].add(raw_name)

    # Convert sets to lists for JSON serialization
    for entity in merged.values():
        entity["aliases"] = list(entity["aliases"])
        entity["keywords"] = list(entity["keywords"])  # NEW: Convert keywords
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
            # Check if entity name exists with any type (more lenient than exact type match)
            if valid_entities:
                source_exists = any(name == canonical_source for (name, _) in valid_entities)
                target_exists = any(name == canonical_target for (name, _) in valid_entities)
                if not source_exists or not target_exists:
                    continue

            if key not in merged:
                merged[key] = {
                    "source": canonical_source,
                    "target": canonical_target,
                    "type": rel_type,
                    "semantic_type": rel.get("semantic_type", rel_type),  # NEW: Semantic type
                    "description": rel.get("description", ""),
                    "strength": rel.get("strength", 5),  # NEW: Relationship strength
                    "keywords": set(rel.get("keywords", [])),  # NEW: Track keywords
                    "seen_in_chunks": [chunk_id],
                    "domain": domain or "unknown",
                }
            else:
                existing = merged[key]

                # Keep longer description
                if len(rel.get("description", "")) > len(existing.get("description", "")):
                    existing["description"] = rel.get("description", "")

                # Keep higher strength
                existing["strength"] = max(existing.get("strength", 5), rel.get("strength", 5))

                # Merge keywords
                existing["keywords"].update(rel.get("keywords", []))

                existing["seen_in_chunks"].append(chunk_id)

    # Dedupe chunk lists and convert sets
    for rel in merged.values():
        rel["seen_in_chunks"] = list(set(rel["seen_in_chunks"]))
        rel["keywords"] = list(rel["keywords"])  # NEW: Convert keywords

    return merged


def _infer_entity_type(name: str, entities: list[dict]) -> str:
    """Infer entity type from a list of entities."""
    for e in entities:
        if e.get("name", "").lower().strip() == name.lower().strip():
            return e.get("type", "OTHER")
    return "OTHER"


def _retry_neo4j_operation(
    operation,
    max_retries: int = 3,
    delay: float = 2.0,
    operation_name: str = "Neo4j operation",
) -> Any:
    """Retry a Neo4j operation with exponential backoff.

    Args:
        operation: Callable to execute
        max_retries: Maximum number of retry attempts
        delay: Initial delay in seconds (doubles each retry)
        operation_name: Name for logging

    Returns:
        Result of the operation

    Raises:
        Exception: If all retries fail
    """
    from utils.logging_config import get_logger
    logger = get_logger(__name__)

    last_error = None
    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    "%s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    operation_name, attempt + 1, max_retries, e, wait_time
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    "%s failed after %d attempts: %s",
                    operation_name, max_retries, e
                )

    raise last_error


def _sanitize_relationship_type(rel_type: str) -> str:
    """Sanitize relationship type for Neo4j.

    Neo4j relationship types must:
    - Start with a letter (a-z, A-Z)
    - Contain only letters, numbers, and underscores
    - Not be empty

    Args:
        rel_type: Raw relationship type string

    Returns:
        Sanitized relationship type safe for Neo4j
    """
    import re

    if not rel_type or not isinstance(rel_type, str):
        return "RELATED_TO"

    # Remove any characters that aren't letters, numbers, or underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", rel_type)

    # Ensure it starts with a letter
    if sanitized and sanitized[0].isdigit():
        sanitized = "REL_" + sanitized

    # Handle edge cases
    if not sanitized or sanitized == "_":
        return "RELATED_TO"

    # Limit length (Neo4j has limits on identifier length)
    if len(sanitized) > 100:
        sanitized = sanitized[:100]

    return sanitized.upper()


def bulk_upsert_to_neo4j(
    merged_entities: dict[tuple[str, str], dict[str, Any]],
    merged_relationships: dict[tuple[str, str, str], dict[str, Any]],
    domain: str | None = None,
) -> bool:
    """Bulk upsert merged entities and relationships to Neo4j.

    Creates nodes with type-specific labels only (e.g., :Person, :Organization).

    Args:
        merged_entities: Dict from merge_entities_across_chunks
        merged_relationships: Dict from merge_relationships_across_chunks
        domain: Optional domain for context

    Returns:
        True if successful, False otherwise
    """
    from graph_stores.neo4j_client import execute_query

    try:
        # Group entities by label for batch operations
        entities_by_label: dict[str, list[dict]] = {}
        for (name, entity_type), entity_data in merged_entities.items():
            label = _entity_type_to_label(entity_type)
            if label not in entities_by_label:
                entities_by_label[label] = []
            entities_by_label[label].append({
                "name": name,
                "type": entity_type,  # Keep type property
                "description": entity_data.get("description", ""),
                "keywords": entity_data.get("keywords", []),  # NEW: Entity keywords
                "domain": domain or entity_data.get("domain", "unknown"),
                "aliases": entity_data.get("aliases", []),
                "chunk_ids": entity_data.get("seen_in_chunks", []),  # Qdrant point IDs
            })

        # Batch upsert entities per label (type-specific label + type property)
        for label, entity_list in entities_by_label.items():
            query = f"""
            UNWIND $entities AS entity
            MERGE (e:{label} {{name: entity.name}})
            ON CREATE SET
                e.type = entity.type,
                e.description = entity.description,
                e.keywords = entity.keywords,
                e.domain = entity.domain,
                e.aliases = entity.aliases,
                e.chunk_ids = entity.chunk_ids
            ON MATCH SET
                e.type = entity.type,
                e.description = entity.description,
                e.keywords = CASE
                    WHEN size(entity.keywords) > 0
                    THEN COALESCE(e.keywords, []) + entity.keywords
                    ELSE e.keywords
                END,
                e.aliases = CASE
                    WHEN size(entity.aliases) > 0 THEN COALESCE(e.aliases, []) + entity.aliases
                    ELSE e.aliases
                END,
                e.chunk_ids = CASE
                    WHEN size(entity.chunk_ids) > 0
                    THEN COALESCE(e.chunk_ids, []) + entity.chunk_ids
                    ELSE e.chunk_ids
                END
            """
            execute_query(query, {"entities": entity_list})

        # Batch upsert relationships (label-agnostic)
        rel_params = []
        for (source, target, rel_type), rel_data in merged_relationships.items():
            rel_params.append({
                "source": source,
                "target": target,
                "type": rel_type,
                "semantic_type": rel_data.get("semantic_type", rel_type),  # NEW: Semantic type
                "description": rel_data.get("description", ""),
                "strength": rel_data.get("strength", 5),  # NEW: Relationship strength
                "keywords": rel_data.get("keywords", []),  # NEW: Relationship keywords
            })

        if rel_params:
            # Group relationships by semantic_type for batch operations with dynamic types
            rels_by_type: dict[str, list[dict]] = {}
            for rel in rel_params:
                # Use semantic_type if available, otherwise fallback to generic type
                rel_type_name = rel.get("semantic_type", "RELATED_TO")
                # Sanitize: ensure it starts with letter and only contains valid chars
                rel_type_name = _sanitize_relationship_type(rel_type_name)

                if rel_type_name not in rels_by_type:
                    rels_by_type[rel_type_name] = []
                rels_by_type[rel_type_name].append(rel)

            # Upsert each relationship type separately (Neo4j requires static type in query)
            for rel_type_name, rels_of_type in rels_by_type.items():
                rel_query = f"""
                UNWIND $relationships AS rel
                MATCH (s {{name: rel.source}})
                MATCH (t {{name: rel.target}})
                MERGE (s)-[r:{rel_type_name}]->(t)
                ON CREATE SET
                    r.type = rel.type,
                    r.description = rel.description,
                    r.strength = rel.strength,
                    r.keywords = rel.keywords
                ON MATCH SET
                    r.description = rel.description,
                    r.strength = rel.strength,
                    r.keywords = CASE
                        WHEN size(rel.keywords) > 0
                        THEN COALESCE(r.keywords, []) + rel.keywords
                        ELSE r.keywords
                    END
                """
                execute_query(rel_query, {"relationships": rels_of_type})

        return True

    except Exception as e:
        from utils.logging_config import get_logger
        logger = get_logger(__name__)
        logger.error("Bulk upsert to Neo4j failed: %s", e)
        return False


def dedup_and_store(
    chunk_results: list[dict[str, Any]],
    domain: str | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Complete deduplication and storage pipeline with retry logic.

    Args:
        chunk_results: List of {chunk_id, entities: [...], relationships: [...]}
        domain: Optional domain for alias resolution
        max_retries: Maximum retry attempts for Neo4j operations

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

    # Bulk upsert to Neo4j with retry
    success = _retry_neo4j_operation(
        lambda: bulk_upsert_to_neo4j(merged_entities, merged_relationships, domain),
        max_retries=max_retries,
        delay=2.0,
        operation_name="Bulk upsert to Neo4j"
    )

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
