import argparse

from vector_stores.qdrant import client, COLLECTION_NAME


def drop_qdrant():
    """Delete the Qdrant collection."""
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"[Qdrant] Dropped collection: {COLLECTION_NAME}")
        return True
    else:
        print(f"[Qdrant] Collection '{COLLECTION_NAME}' does not exist")
        return False


def drop_neo4j():
    """Delete all nodes and relationships from Neo4j."""
    try:
        from graph_stores.neo4j_client import execute_query

        # Count before deletion
        result = execute_query("MATCH (n) RETURN count(n) as count")
        count = result[0]["count"] if result else 0

        if count > 0:
            execute_query("MATCH (n) DETACH DELETE n")
            print(f"[Neo4j] Deleted {count} nodes")
            return True
        else:
            print("[Neo4j] Graph is already empty")
            return False
    except Exception as e:
        print(f"[Neo4j] Error: {e}")
        return False


def drop_both():
    """Delete both Qdrant collection and Neo4j graph."""
    print("\n=== Dropping Both ===")
    qdrant_ok = drop_qdrant()
    neo4j_ok = drop_neo4j()
    if qdrant_ok or neo4j_ok:
        print("\n[Done] Dropped selected stores")
    else:
        print("\n[Info] Nothing to drop")


def main():
    parser = argparse.ArgumentParser(description="Drop vector store and/or knowledge graph data")
    parser.add_argument("--qdrant", action="store_true", help="Drop Qdrant collection")
    parser.add_argument("--neo4j", action="store_true", help="Drop Neo4j graph")
    parser.add_argument("--both", action="store_true", help="Drop both Qdrant and Neo4j")

    args = parser.parse_args()

    # If no flags, show interactive menu
    if not any([args.qdrant, args.neo4j, args.both]):
        print("\n=== Drop Data ===")
        print("1. Drop Qdrant (vector store)")
        print("2. Drop Neo4j (knowledge graph)")
        print("3. Drop Both")
        print("0. Cancel")

        choice = input("\nSelect an option: ").strip()

        if choice == "1":
            drop_qdrant()
        elif choice == "2":
            drop_neo4j()
        elif choice == "3":
            drop_both()
        else:
            print("Cancelled")
        return

    # CLI flag mode
    if args.both:
        drop_both()
    elif args.qdrant:
        drop_qdrant()
    elif args.neo4j:
        drop_neo4j()


if __name__ == "__main__":
    main()
