from vector_stores.qdrant import client, COLLECTION_NAME


def drop_collection():
    """Delete the Qdrant collection."""
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"Dropped collection: {COLLECTION_NAME}")
    else:
        print(f"Collection '{COLLECTION_NAME}' does not exist")


if __name__ == "__main__":
    drop_collection()
