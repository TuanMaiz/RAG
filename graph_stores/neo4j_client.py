"""Neo4j knowledge graph client - Basic operations.

Simple client for Neo4j CRUD operations.
"""

import os
import re
from typing import Any, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase

from utils.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
ENABLE_KG = os.getenv("ENABLE_KG", "true").lower() == "true"


def _sanitize_rel_type(rel_type: str) -> str:
    """Sanitize relationship type for valid Cypher syntax.

    Neo4j relationship types must be alphanumeric with underscores.
    Converts comma-separated keywords to a valid type and returns
    the original keywords to be stored as a property.

    Args:
        rel_type: Raw relationship type (may contain commas, spaces)

    Returns:
        Tuple of (sanitized_type, original_keywords)
    """
    # Store original keywords
    original = rel_type.strip()

    # Sanitize: keep only alphanumeric and underscore, uppercase
    # Convert spaces and commas to underscores
    sanitized = re.sub(r'[^a-zA-Z0-9]+', '_', rel_type).upper()
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Limit length (Neo4j has limits on identifier length)
    sanitized = sanitized[:50]

    # Fallback if empty
    if not sanitized:
        sanitized = "RELATED_TO"

    return sanitized, original


# Singleton client instance
_driver: Optional[GraphDatabase.driver] = None


def get_driver() -> Optional[GraphDatabase.driver]:
    """Get or create the Neo4j driver instance.

    Returns None if KG is disabled or connection fails.
    """
    global _driver

    if not ENABLE_KG:
        return None

    if _driver is not None:
        return _driver

    try:
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        _driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", NEO4J_URI)
        return _driver
    except Exception as e:
        logger.warning("Failed to connect to Neo4j: %s", e)
        _driver = None
        return None


def close() -> None:
    """Close the Neo4j driver connection."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def execute_query(query: str, params: dict[str, Any] = None) -> list[dict[str, Any]]:
    """Execute a Cypher query and return results.

    Args:
        query: Cypher query string
        params: Query parameters

    Returns:
        List of result dictionaries
    """
    driver = get_driver()
    if driver is None:
        return []

    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query, params or {})
        return [dict(record) for record in result]


def create_node(label: str, properties: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Create a node with the given label and properties.

    Args:
        label: Node label (e.g., "Document", "Entity")
        properties: Node properties

    Returns:
        Created node, or None if KG disabled
    """
    driver = get_driver()
    if driver is None:
        return None

    query = f"CREATE (n:{label} $props) RETURN n"
    results = execute_query(query, {"props": properties})
    return results[0] if results else None


def merge_node(label: str, properties: dict[str, Any], merge_key: str = None) -> Optional[dict[str, Any]]:
    """Merge a node (create if not exists) based on a key.

    Args:
        label: Node label
        properties: Node properties
        merge_key: Property key to match for merge (defaults to "id", or "name" if "id" not present)

    Returns:
        Merged node, or None if KG disabled
    """
    driver = get_driver()
    if driver is None:
        return None

    # Default merge key: prefer "id", then "name", then first key
    if merge_key is None:
        merge_key = "id" if "id" in properties else ("name" if "name" in properties else list(properties.keys())[0])

    if merge_key not in properties:
        raise ValueError(f"Merge key '{merge_key}' not found in properties: {list(properties.keys())}")

    query = f"""
    MERGE (n:{label} {{{merge_key}: $merge_val}})
    ON CREATE SET n = $props
    ON MATCH SET n += $props
    RETURN n
    """
    results = execute_query(query, {"merge_val": properties[merge_key], "props": properties})
    return results[0] if results else None


def create_relationship(
    from_label: str,
    from_props: dict[str, Any],
    to_label: str,
    to_props: dict[str, Any],
    rel_type: str,
    rel_props: dict[str, Any] = None,
) -> bool:
    """Create a relationship between two nodes.

    Args:
        from_label: Source node label
        from_props: Source node properties for matching
        to_label: Target node label
        to_props: Target node properties for matching
        rel_type: Relationship type (e.g., "MENTIONS", "RELATED_TO")
        rel_props: Relationship properties

    Returns:
        True if successful, False otherwise
    """
    driver = get_driver()
    if driver is None:
        return False

    # Sanitize relationship type for Cypher syntax
    sanitized_type, original_keywords = _sanitize_rel_type(rel_type)

    # Store original keywords as property
    if rel_props is None:
        rel_props = {}
    rel_props["keywords"] = original_keywords

    from_key = list(from_props.keys())[0]
    to_key = list(to_props.keys())[0]

    query = f"""
    MATCH (from:{from_label} {{{from_key}: $from_val}})
    MATCH (to:{to_label} {{{to_key}: $to_val}})
    CREATE (from)-[r:{sanitized_type}]->(to)
    SET r += $rel_props
    RETURN r
    """

    with driver.session(database=NEO4J_DATABASE) as session:
        try:
            session.run(
                query,
                from_val=from_props[from_key],
                to_val=to_props[to_key],
                rel_props=rel_props,
            )
            return True
        except Exception as e:
            logger.warning("Failed to create relationship: %s", e)
            return False


def merge_relationship(
    from_label: str,
    from_props: dict[str, Any],
    to_label: str,
    to_props: dict[str, Any],
    rel_type: str,
    rel_props: dict[str, Any] = None,
) -> bool:
    """Merge a relationship (create if not exists).

    Args:
        from_label: Source node label
        from_props: Source node properties for matching
        to_label: Target node label
        to_props: Target node properties for matching
        rel_type: Relationship type
        rel_props: Relationship properties

    Returns:
        True if successful, False otherwise
    """
    driver = get_driver()
    if driver is None:
        return False

    # Sanitize relationship type for Cypher syntax
    sanitized_type, original_keywords = _sanitize_rel_type(rel_type)

    # Store original keywords as property
    if rel_props is None:
        rel_props = {}
    rel_props["keywords"] = original_keywords

    from_key = list(from_props.keys())[0]
    to_key = list(to_props.keys())[0]

    query = f"""
    MATCH (from:{from_label} {{{from_key}: $from_val}})
    MATCH (to:{to_label} {{{to_key}: $to_val}})
    MERGE (from)-[r:{sanitized_type}]->(to)
    ON CREATE SET r += $rel_props
    ON MATCH SET r += $rel_props
    RETURN r
    """

    with driver.session(database=NEO4J_DATABASE) as session:
        try:
            session.run(
                query,
                from_val=from_props[from_key],
                to_val=to_props[to_key],
                rel_props=rel_props,
            )
            return True
        except Exception as e:
            logger.warning("Failed to merge relationship: %s", e)
            return False


# Export
__all__ = [
    "get_driver",
    "close",
    "execute_query",
    "create_node",
    "merge_node",
    "create_relationship",
    "merge_relationship",
    "ENABLE_KG",
    "NEO4J_URI",
]
