#!/usr/bin/env python3
"""
Neo4j Schema Migration Script

Migrates from (name, title) entity keys to (name, type) entity keys.

BACKUP YOUR NEO4J DATABASE BEFORE RUNNING!

Steps:
1. Backup current state to neo4j_backup.json
2. Find entity collisions (same name+type, different titles)
3. Merge colliding entities
4. Update MENTIONS relationships
5. Update RELATED_TO relationships
6. Remove title from entity constraints
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from graph_stores.neo4j_client import execute_query, get_driver
from utils.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)


def backup_neo4j(output_path: str = "neo4j_backup.json") -> dict:
    """Backup all entities and relationships to JSON.

    Args:
        output_path: Path to save the backup

    Returns:
        Dict with backup statistics
    """
    logger.info("Creating Neo4j backup to %s...", output_path)

    # Backup entities
    entity_query = """
    MATCH (e:Entity)
    RETURN e.name as name, e.type as type, e.title as title,
           e.description as description, collect(id(e)) as ids
    """

    # Backup relationships
    rel_query = """
    MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity)
    RETURN e1.name as source, e1.title as source_title,
           e2.name as target, e2.title as target_title,
           r.type as type, r.description as description
    """

    # Backup chunk mentions
    chunk_query = """
    MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity)
    RETURN c.id as chunk_id, c.title as chunk_title,
           e.name as entity_name, e.title as entity_title
    """

    try:
        entities = execute_query(entity_query)
        relationships = execute_query(rel_query)
        chunks = execute_query(chunk_query)

        backup = {
            "timestamp": datetime.now().isoformat(),
            "entities": entities,
            "relationships": relationships,
            "chunk_mentions": chunks,
        }

        with open(output_path, "w") as f:
            json.dump(backup, f, indent=2)

        logger.info("Backup saved: %d entities, %d relationships, %d mentions",
                   len(entities), len(relationships), len(chunks))

        return {
            "entities": len(entities),
            "relationships": len(relationships),
            "mentions": len(chunks),
        }

    except Exception as e:
        logger.error("Backup failed: %s", e)
        return {"error": str(e)}


def find_entity_collisions() -> list[dict]:
    """Find entities with same (name, type) but different titles.

    Returns:
        List of collision info dicts
    """
    logger.info("Finding entity collisions...")

    query = """
    MATCH (e:Entity)
    WITH e.name as name, e.type as type, collect(DISTINCT e.title) as titles
    WHERE size(titles) > 1
    RETURN name, type, titles, count(*) as count
    ORDER BY count DESC
    """

    try:
        results = execute_query(query)
        collisions = [dict(r) for r in results]

        logger.info("Found %d entity collisions", len(collisions))

        for collision in collisions[:5]:  # Show first 5
            logger.info("  - %s (%s): %s", collision["name"],
                       collision["type"], collision["titles"])

        return collisions

    except Exception as e:
        logger.error("Failed to find collisions: %s", e)
        return []


def merge_colliding_entities(dry_run: bool = False) -> int:
    """Merge entities with same (name, type) but different titles.

    Args:
        dry_run: If True, don't actually merge

    Returns:
        Number of entities merged
    """
    logger.info("Merging colliding entities...")

    # Find all entities grouped by (name, type)
    query = """
    MATCH (e:Entity)
    WITH e.name as name, e.type as type, collect(e) as entities
    WHERE size(entities) > 1
    RETURN name, type, entities
    """

    try:
        results = execute_query(query)
        merged_count = 0

        for result in results:
            entities = result["entities"]
            if len(entities) <= 1:
                continue

            # Keep the entity with the most relationships
            entities.sort(key=lambda e: len(e.get("_relationships", 0)), reverse=True)
            primary = entities[0]
            duplicates = entities[1:]

            logger.info("  Merging '%s' (%s): %d duplicates",
                       primary.get("name"), primary.get("type"), len(duplicates))

            if not dry_run:
                # Redirect relationships from duplicates to primary
                for dup in duplicates:
                    # Update RELATED_TO relationships
                    execute_query("""
                        MATCH (dup:Entity)-[r:RELATED_TO]->(other:Entity)
                        WHERE elementId(dup) = $dup_id
                        MATCH (primary:Entity)
                        WHERE elementId(primary) = $primary_id
                        MERGE (primary)-[new:RELATED_TO {type: r.type, description: r.description}]->(other)
                        DELETE r
                    """, {"dup_id": dup.element_id, "primary_id": primary.element_id})

                    execute_query("""
                        MATCH (other:Entity)-[r:RELATED_TO]->(dup:Entity)
                        WHERE elementId(dup) = $dup_id
                        MATCH (primary:Entity)
                        WHERE elementId(primary) = $primary_id
                        MERGE (other)-[new:RELATED_TO {type: r.type, description: r.description}]->(primary)
                        DELETE r
                    """, {"dup_id": dup.element_id, "primary_id": primary.element_id})

                    # Update MENTIONS relationships
                    execute_query("""
                        MATCH (c:Chunk)-[r:MENTIONS]->(dup:Entity)
                        WHERE elementId(dup) = $dup_id
                        MATCH (primary:Entity)
                        WHERE elementId(primary) = $primary_id
                        MERGE (c)-[new:MENTIONS]->(primary)
                        DELETE r
                    """, {"dup_id": dup.element_id, "primary_id": primary.element_id})

                    # Delete duplicate node
                    execute_query("""
                        MATCH (dup:Entity)
                        WHERE elementId(dup) = $dup_id
                        DELETE dup
                    """, {"dup_id": dup.element_id})

                merged_count += 1 + len(duplicates)

        return merged_count

    except Exception as e:
        logger.error("Failed to merge entities: %s", e)
        return 0


def migrate_entity_constraints(dry_run: bool = False) -> bool:
    """Remove title from entity constraints.

    Args:
        dry_run: If True, don't actually modify

    Returns:
        True if successful
    """
    logger.info("Migrating entity constraints...")

    # The main change is in how we query - entities are now (name, type) only
    # No constraint changes needed in Neo4j, just in our queries

    if not dry_run:
        # Set schema version flag
        logger.info("Update KG_SCHEMA_VERSION=2 in .env after migration")
    else:
        logger.info("Dry run: would update schema version")

    return True


def verify_migration() -> dict:
    """Verify the migration was successful.

    Returns:
        Dict with verification stats
    """
    logger.info("Verifying migration...")

    # Count entities
    entity_count = execute_query("MATCH (e:Entity) RETURN count(e) as count")[0]["count"]

    # Count relationships
    rel_count = execute_query("MATCH ()-[r:RELATED_TO]->() RETURN count(r) as count")[0]["count"]

    # Check for any entities with empty names (shouldn't exist)
    empty_names = execute_query("""
        MATCH (e:Entity)
        WHERE e.name IS NULL OR e.name = ''
        RETURN count(e) as count
    """)[0]["count"]

    # Check for orphan relationships
    orphan_rels = execute_query("""
        MATCH ()-[r:RELATED_TO]->(e:Entity)
        WHERE e.name IS NULL OR e.name = ''
        RETURN count(r) as count
    """)[0]["count"]

    stats = {
        "entity_count": entity_count,
        "relationship_count": rel_count,
        "empty_names": empty_names,
        "orphan_relationships": orphan_rels,
        "success": empty_names == 0 and orphan_rels == 0,
    }

    logger.info("Verification: %d entities, %d relationships",
               entity_count, rel_count)

    if not stats["success"]:
        logger.warning("Issues found: %d empty names, %d orphan relationships",
                     empty_names, orphan_rels)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Neo4j Schema Migration")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--backup", default="neo4j_backup.json", help="Backup file path")
    parser.add_argument("--skip-backup", action="store_true", help="Skip backup step")
    args = parser.parse_args()

    print("=" * 60)
    print("Neo4j Schema Migration")
    print("=" * 60)
    print()
    print("This will:")
    print("1. Back up your current Neo4j database")
    print("2. Merge entities with same (name, type) from different titles")
    print("3. Update relationships")
    print("4. Migrate to schema version 2")
    print()
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        print()

    confirm = input("Proceed? (yes/NO): ").strip().lower() if not args.dry_run else "yes"
    if confirm != "yes":
        print("Aborted.")
        return 1

    # Check connection
    driver = get_driver()
    if driver is None:
        print("ERROR: Could not connect to Neo4j")
        return 1

    print()

    # Step 1: Backup
    if not args.skip_backup:
        print("[1/4] Creating backup...")
        backup_stats = backup_neo4j(args.backup)
        if "error" in backup_stats:
            print(f"Backup failed: {backup_stats['error']}")
            return 1
        print(f"  Backup complete: {args.backup}")
    else:
        print("[1/4] Skipping backup (--skip-backup)")

    # Step 2: Find collisions
    print("\n[2/4] Finding entity collisions...")
    collisions = find_entity_collisions()

    if not collisions:
        print("  No collisions found - no merging needed")
    else:
        print(f"  Found {len(collisions)} collisions")

    # Step 3: Merge entities
    print("\n[3/4] Merging colliding entities...")
    merged = merge_colliding_entities(dry_run=args.dry_run)
    if args.dry_run:
        print(f"  Would merge {merged} entities")
    else:
        print(f"  Merged {merged} entities")

    # Step 4: Verify
    print("\n[4/4] Verifying migration...")
    stats = verify_migration()

    if stats["success"]:
        print("\n" + "=" * 60)
        print("Migration complete!")
        print("=" * 60)
        print(f"Entities: {stats['entity_count']}")
        print(f"Relationships: {stats['relationship_count']}")
        print()
        print("Next steps:")
        print("1. Set KG_SCHEMA_VERSION=2 in your .env file")
        print("2. Set KG_ENABLE_IN_MEMORY_DEDUP=true")
        print("3. Restart your application")
        print("4. Keep backup file for at least 1 week")
        return 0
    else:
        print("\nMigration verification failed!")
        print(f"Empty names: {stats['empty_names']}")
        print(f"Orphan relationships: {stats['orphan_relationships']}")
        print()
        print("Please restore from backup and investigate.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
