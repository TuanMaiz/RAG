import os
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loaders.document_loader import JSONLLoader
from vector_stores.qdrant import (
    COLLECTION_NAME,
    client,
    create_hybrid_collection,
    embeddings,
)
from workflow.hybrid_retrieval import sparse_vector_from_text

# KG indexing flag
ENABLE_KG = os.getenv("ENABLE_KG", "true").lower() == "true"


def extract_domain(source_path: str) -> str:
    """Extract domain name from source file path.

    The domain is determined by the dataset filename.

    Examples:
        dataset/clapnq.jsonl/clapnq.jsonl → clapnq
        dataset/cloud.jsonl/cloud.jsonl → cloud
        dataset/fiqa.jsonl/fiqa.jsonl → fiqa
        dataset/govt.jsonl/govt.jsonl → govt

    Args:
        source_path: Path to the source file

    Returns:
        Domain name (lowercase)
    """
    path = Path(source_path)
    # Get filename without extension, then get parent folder name
    # For path like "dataset/cloud.jsonl/cloud.jsonl", we want "cloud"
    filename = path.stem  # e.g., "cloud.jsonl"
    # Remove .jsonl if present and get base name
    if filename.endswith(".jsonl"):
        filename = filename[:-6]  # Remove ".jsonl" (6 chars)
    return filename.lower()


def store_entities_to_graph(
    doc_id: int,
    text: str,
    domain: str,
) -> bool:
    """Extract entities from text and store in Neo4j.

    Creates Document node, Entity nodes, and MENTIONS relationships.
    Also creates relationships between entities.

    Args:
        doc_id: Qdrant point ID (also Document node ID in Neo4j)
        text: Document text to extract entities from
        domain: Dataset domain (clapnq, cloud, fiqa, govt)

    Returns:
        True if successful or KG disabled, False on error
    """
    if not ENABLE_KG:
        return True  # KG disabled, skip gracefully

    from graph_stores.neo4j_client import get_driver, merge_node, create_relationship
    from workflow.kg_extraction import extract_graph_data

    driver = get_driver()
    if driver is None:
        return False  # KG connection failed

    try:
        # Extract entities and relationships
        graph_data = extract_graph_data(text, domain)

        if not graph_data["entities"]:
            return True  # No entities found, but not an error

        # Create/update Document node
        merge_node("Document", {"id": str(doc_id), "domain": domain})

        # Create/update Entity nodes and MENTIONS relationships
        for entity in graph_data["entities"]:
            merge_node("Entity", {
                "name": entity["name"],
                "type": entity["type"],
                "domain": domain
            })

            # Link Document to Entity
            create_relationship(
                "Document", {"id": str(doc_id)},
                "Entity", {"name": entity["name"], "domain": domain},
                "MENTIONS"
            )

        # Store relationships between entities
        for rel in graph_data.get("relationships", []):
            create_relationship(
                "Entity", {"name": rel["source"], "domain": domain},
                "Entity", {"name": rel["target"], "domain": domain},
                rel["type"]
            )

        return True

    except Exception as e:
        print(f"Warning: Failed to store entities for doc {doc_id}: {e}")
        return False


def load_doc(dataset_dir: str | Path = "dataset") -> list[Document]:
    """
    Index all documents from JSONL files in the dataset directory.

    Args:
        dataset_dir: Base directory containing datasets

    Returns:
        List of split Document objects
    """
    dataset_dir = Path(dataset_dir)

    # Find all .jsonl files recursively (files only, not directories)
    jsonl_files = [p for p in dataset_dir.glob("**/*.jsonl") if p.is_file()]

    if not jsonl_files:
        return []

    all_documents = []
    for jsonl_path in jsonl_files:
        loader = JSONLLoader(jsonl_path)
        docs = loader.load()
        all_documents.extend(docs)

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(all_documents)

    return all_splits


def store_doc(docs: list[Document], batch_size: int = 350, resume: bool = True):
    """
    Store documents with hybrid (dense + sparse) vectors.

    Args:
        docs: List of Document objects to store
        batch_size: Number of documents to store per batch
        resume: If True, continue from existing collection (don't delete)
    """
    import json
    from pathlib import Path
    from qdrant_client.http.models import PointStruct, SparseVector

    total = len(docs)
    progress_file = Path(".indexing_progress.json")
    start_index = 0

    # Resume: check existing collection
    if resume and client.collection_exists(COLLECTION_NAME):
        collection_info = client.get_collection(COLLECTION_NAME)
        existing_count = collection_info.points_count

        # Check if has sparse vectors (hybrid collection)
        has_sparse = collection_info.config.params.sparse_vectors is not None

        if existing_count > 0:
            if has_sparse:
                print(f"Resuming: collection has {existing_count} points with sparse vectors")
                start_index = existing_count
            else:
                print(f"Warning: Collection has {existing_count} points but NO sparse vectors")
                confirm = input("Delete and recreate with hybrid? (y/N): ").strip().lower()
                if confirm == "y":
                    client.delete_collection(COLLECTION_NAME)
                    create_hybrid_collection()
                else:
                    print("Cannot resume - collection missing sparse vectors")
                    return

        if start_index >= total:
            print(f"All {total} documents already indexed!")
            return

    # Create hybrid collection if needed
    if not client.collection_exists(COLLECTION_NAME):
        create_hybrid_collection()

    # Save progress
    def save_progress(current_index):
        progress_file.write_text(json.dumps({
            "total": total,
            "indexed": current_index,
            "timestamp": time.time()
        }))

    save_progress(0)
    start = time.time()

    for i in range(start_index, total, batch_size):
        batch = docs[i : i + batch_size]
        batch_start = time.time()

        # Batch embeddings
        texts = [doc.page_content for doc in batch]
        print(f"  Embedding {len(batch)} documents...", end="", flush=True)
        try:
            dense_vectors = embeddings.embed_documents(texts)
            print(f" done ({time.time() - batch_start:.1f}s)")
        except Exception as e:
            print(f" failed: {e}")
            print("  Waiting 10s and retrying...")
            time.sleep(10)
            dense_vectors = embeddings.embed_documents(texts)
            print(f"  Retry done ({time.time() - batch_start:.1f}s)")

        # Build points
        points = []
        for idx, doc in enumerate(batch):
            doc_id = i + idx

            # Sparse vector (BM25-style)
            sparse_vector = sparse_vector_from_text(doc.page_content)

            # Build payload
            payload = {
                "page_content": doc.page_content,
                **doc.metadata,
            }

            # Store entities to Neo4j knowledge graph
            if ENABLE_KG:
                domain = extract_domain(doc.metadata.get("source", ""))
                store_entities_to_graph(doc_id, doc.page_content, domain)

            point = PointStruct(
                id=doc_id,
                vector={
                    "dense": dense_vectors[idx],
                    "sparse": SparseVector(
                        indices=sparse_vector.indices,
                        values=sparse_vector.values,
                    ),
                },
                payload=payload,
            )
            points.append(point)

        # Batch upsert with retry
        upsert_start = time.time()
        max_retries = 5
        for retry in range(max_retries):
            try:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                )
                break
            except Exception as e:
                if retry == max_retries - 1:
                    # Save progress before raising
                    save_progress(i)
                    raise
                print(f"  Retry {retry + 1}/{max_retries} after error: {str(e)[:50]}")
                time.sleep(5)
        print(f"  Upserted in {time.time() - upsert_start:.1f}s")

        # Save progress every batch
        save_progress(i + len(batch))

        elapsed = time.time() - start
        progress = min(i + len(batch), total) / total * 100
        eta = (elapsed / (i + len(batch) - start_index)) * (total - i - len(batch))
        rate = (i + len(batch) - start_index) / elapsed * 60
        print(f"Stored {min(i + len(batch), total)}/{total} ({progress:.1f}%) - {elapsed:.1f}s - ETA: {eta/60:.1f}min - Rate: {rate:.0f} docs/min")

    # Delete progress file on completion
    if progress_file.exists():
        progress_file.unlink()

    print(f"Finished storing {total} hybrid documents in {time.time() - start:.2f}s")

