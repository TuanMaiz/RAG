import os
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from loaders.document_loader import JSONLLoader
from utils.logging_config import get_logger
from vector_stores.qdrant import (
    COLLECTION_NAME,
    client,
    create_hybrid_collection,
    get_embeddings,
)
from workflow.hybrid_retrieval import sparse_vector_from_text

# KG indexing flag
ENABLE_KG = os.getenv("ENABLE_KG", "true").lower() == "true"

logger = get_logger(__name__)


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
    chunk_id: int,
    text: str,
    title: str,
) -> bool:
    """Extract entities from chunk text and store in Neo4j.

    Creates Chunk node, Entity nodes, and MENTIONS relationships.
    Also creates relationships between entities.

    Args:
        chunk_id: Qdrant point ID (also Chunk node ID in Neo4j)
        text: Chunk text to extract entities from
        title: Document title (groups chunks from same document)

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
        graph_data = extract_graph_data(text)

        if not graph_data["entities"]:
            return True  # No entities found, but not an error

        # Create/update Chunk node
        merge_node("Chunk", {"id": str(chunk_id), "title": title})

        # Create/update Entity nodes and MENTIONS relationships
        for entity in graph_data["entities"]:
            merge_node("Entity", {
                "name": entity["name"],
                "type": entity["type"],
                "title": title
            })

            # Link Chunk to Entity
            create_relationship(
                "Chunk", {"id": str(chunk_id)},
                "Entity", {"name": entity["name"], "title": title},
                "MENTIONS"
            )

        # Store relationships between entities
        for rel in graph_data.get("relationships", []):
            create_relationship(
                "Entity", {"name": rel["source"], "title": title},
                "Entity", {"name": rel["target"], "title": title},
                rel["type"]
            )

        return True

    except Exception as e:
        logger.warning("Failed to store entities for chunk %d: %s", chunk_id, e)
        return False


def store_entities_to_graph_batch(texts_with_ids: list[tuple[int, str, str]]) -> None:
    """Extract and store entities for multiple chunks in batch.

    Args:
        texts_with_ids: List of (chunk_id, text, title) tuples
    """
    # Check if deduplication is enabled
    enable_dedup = os.getenv("KG_ENABLE_IN_MEMORY_DEDUP", "false").lower() == "true"

    if enable_dedup:
        store_entities_to_graph_batch_dedup(texts_with_ids)
    else:
        store_entities_to_graph_batch_legacy(texts_with_ids)


def store_entities_to_graph_batch_legacy(texts_with_ids: list[tuple[int, str, str]]) -> None:
    """Legacy extraction and storage (per-chunk, no deduplication).

    Args:
        texts_with_ids: List of (chunk_id, text, title) tuples
    """
    if not ENABLE_KG or not texts_with_ids:
        return

    from graph_stores.neo4j_client import get_driver, merge_node, create_relationship
    from workflow.kg_extraction import extract_graph_data_batch

    driver = get_driver()
    if driver is None:
        logger.warning("Neo4j not available, skipping KG extraction")
        return

    # Prepare data: map by title
    title_groups: dict[str, list[tuple[int, str]]] = {}
    for chunk_id, text, title in texts_with_ids:
        if title not in title_groups:
            title_groups[title] = []
        title_groups[title].append((chunk_id, text))

    # Process each title group
    for title, texts in title_groups.items():
        try:
            # Batch extraction for this title group
            texts_with_ids = [(chunk_id, text) for chunk_id, text in texts]
            results = extract_graph_data_batch(texts_with_ids, batch_size=10)

            # Store results
            for chunk_id, data in results.items():
                entities = data.get("entities", [])
                relationships = data.get("relationships", [])

                if not entities:
                    continue

                # Create Chunk node
                merge_node("Chunk", {"id": str(chunk_id), "title": title})

                # Create Entity nodes and MENTIONS relationships
                for entity in entities:
                    merge_node("Entity", {
                        "name": entity["name"],
                        "type": entity["type"],
                        "title": title
                    })

                    create_relationship(
                        "Chunk", {"id": str(chunk_id)},
                        "Entity", {"name": entity["name"], "title": title},
                        "MENTIONS"
                    )

                # Store relationships between entities
                for rel in relationships:
                    create_relationship(
                        "Entity", {"name": rel["source"], "title": title},
                        "Entity", {"name": rel["target"], "title": title},
                        rel["type"]
                    )

        except Exception as e:
            logger.warning("Batch KG extraction failed for title '%s': %s", title, e)


def store_entities_to_graph_batch_dedup(texts_with_ids: list[tuple[int, str, str]]) -> None:
    """Extract entities with in-memory deduplication before Neo4j storage.

    This implements the new pipeline:
    1. Extract all entities/relationships from chunks
    2. Deduplicate across chunks (alias resolution, merging)
    3. Bulk upsert to Neo4j

    Args:
        texts_with_ids: List of (chunk_id, text, title) tuples
    """
    if not ENABLE_KG or not texts_with_ids:
        return

    from graph_stores.neo4j_client import get_driver
    from workflow.kg_dedup import dedup_and_store
    from workflow.kg_extraction import extract_graph_data_batch

    driver = get_driver()
    if driver is None:
        logger.warning("Neo4j not available, skipping KG extraction")
        return

    # Group by title/domain for processing
    title_groups: dict[str, list[tuple]] = {}
    for chunk_id, text, title in texts_with_ids:
        if title not in title_groups:
            title_groups[title] = []
        title_groups[title].append((chunk_id, text))

    # Process each title group with deduplication
    for title, texts in title_groups.items():
        try:
            # Extract entities for all chunks in this title group
            texts_with_ids = [(chunk_id, text) for chunk_id, text in texts]
            results = extract_graph_data_batch(texts_with_ids, batch_size=10, domain=title)

            # Convert to format expected by dedup_and_store
            chunk_results = []
            for chunk_id, data in results.items():
                chunk_results.append({
                    "chunk_id": chunk_id,
                    "entities": data.get("entities", []),
                    "relationships": data.get("relationships", []),
                })

            # Deduplicate and bulk upsert
            stats = dedup_and_store(chunk_results, domain=title)
            logger.info(
                "Deduplication for '%s': %d entities, %d relationships",
                title, stats["entity_count"], stats["relationship_count"]
            )

        except Exception as e:
            logger.warning("Deduplication failed for title '%s': %s", title, e)


def load_doc(dataset_dir: str | Path = "dataset", dataset_name: str | None = None) -> list[Document]:
    """
    Index documents from JSONL files in the dataset directory.

    Args:
        dataset_dir: Base directory containing datasets
        dataset_name: Optional specific dataset to load (e.g., "clapnq", "cloud").
                     If None, loads all datasets.

    Returns:
        List of split Document objects
    """
    dataset_dir = Path(dataset_dir)

    # Find all .jsonl files recursively (files only, not directories)
    jsonl_files = [p for p in dataset_dir.glob("**/*.jsonl") if p.is_file()]

    if not jsonl_files:
        return []

    # Filter by dataset name if specified
    if dataset_name:
        jsonl_files = [p for p in jsonl_files if dataset_name.lower() in p.name.lower()]

    if not jsonl_files:
        logger.warning("No files found for dataset: %s", dataset_name)
        return []

    logger.info("Found %d file(s): %s", len(jsonl_files), [f.name for f in jsonl_files])

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
                print(f"Resuming from {existing_count} documents")
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

    # Calculate number of batches
    num_batches = (total - start_index + batch_size - 1) // batch_size

    # Progress bar with tqdm
    with tqdm(total=total - start_index, desc="Indexing", unit="doc",
              initial=start_index, ncols=100) as pbar:
        for i in range(start_index, total, batch_size):
            batch = docs[i : i + batch_size]
            batch_start = time.time()

            # Batch embeddings
            texts = [doc.page_content for doc in batch]
            try:
                dense_vectors = get_embeddings().embed_documents(texts)
            except Exception as e:
                pbar.set_postfix_str("Retrying...")
                time.sleep(10)
                dense_vectors = get_embeddings().embed_documents(texts)

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
                        save_progress(i)
                        raise
                    pbar.set_postfix_str(f"Retry {retry + 1}/{max_retries}")
                    time.sleep(5)

            # Batch KG extraction (if enabled) - process all docs in batch at once
            if ENABLE_KG:
                pbar.set_postfix_str("Extracting entities...")
                # Collect (chunk_id, text, title) tuples for this batch
                kg_texts = [(i + idx, doc.page_content, doc.metadata.get("title", "Unknown"))
                               for idx, doc in enumerate(batch)]

                # Extract and store entities for the batch
                store_entities_to_graph_batch(kg_texts)

            # Save progress every batch
            save_progress(i + len(batch))

            # Update progress bar with stats
            elapsed = time.time() - start
            rate = (i + len(batch) - start_index) / elapsed * 60
            eta = (elapsed / (i + len(batch) - start_index)) * (total - i - len(batch))
            pbar.update(len(batch))
            pbar.set_postfix_str(f"{rate:.0f}/min, ETA {eta/60:.1f}min")

    # Delete progress file on completion
    if progress_file.exists():
        progress_file.unlink()

    elapsed = time.time() - start
    print(f"\nFinished storing {total} documents in {elapsed:.1f}s ({total/elapsed:.0f} docs/min)")

