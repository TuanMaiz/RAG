import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from qdrant_client.http.models import PointStruct, SparseVector

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

# Parallel processing config
DEFAULT_MAX_WORKERS = int(os.getenv("INDEXING_MAX_WORKERS", "4"))

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


def _process_batch(
    docs: list[Document],
    start_idx: int,
    batch_size: int,
) -> dict:
    """Process a single batch with parallel KG extraction.

    This function runs two pipelines concurrently:
    1. Vector pipeline: embeddings → sparse → Qdrant upsert
    2. KG pipeline: entity extraction and storage (if enabled)

    Args:
        docs: List of Documents in this batch
        start_idx: Starting index for document IDs
        batch_size: Batch size for progress tracking

    Returns:
        dict with keys: count, success, error (if any)
    """
    # Prepare data
    texts = [doc.page_content for doc in docs]

    # Generate embeddings (blocking API call)
    try:
        dense_vectors = get_embeddings().embed_documents(texts)
    except Exception as e:
        time.sleep(10)
        dense_vectors = get_embeddings().embed_documents(texts)

    # Build points with sparse vectors
    points = []
    kg_texts = []
    for idx, doc in enumerate(docs):
        doc_id = start_idx + idx

        # Sparse vector (BM25-style, local and fast)
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

        # Collect for KG extraction
        if ENABLE_KG:
            kg_texts.append((
                doc_id,
                doc.page_content,
                doc.metadata.get("title", "Unknown")
            ))

    # Prepare data for KG pipeline
    kg_data = kg_texts if ENABLE_KG else None

    # Define pipelines
    def _vector_pipeline():
        """Execute Qdrant upsert with retry logic."""
        max_retries = 5
        for retry in range(max_retries):
            try:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                )
                return None
            except Exception as e:
                if retry == max_retries - 1:
                    return e
                time.sleep(5)
        return None

    def _kg_pipeline():
        """Execute KG extraction."""
        if kg_data:
            store_entities_to_graph_batch(kg_data)
        return None

    # Run vector pipeline (KG runs in parallel thread)
    kg_thread = None
    if kg_data:
        kg_thread = threading.Thread(target=_kg_pipeline, daemon=False)
        kg_thread.start()

    # Execute vector pipeline (blocking - must complete)
    vector_error = _vector_pipeline()

    # Return both result and KG thread so caller can join it
    if vector_error:
        # Wait for KG thread before raising error
        if kg_thread and kg_thread.is_alive():
            kg_thread.join(timeout=30)
        raise vector_error

    return {
        "count": len(docs),
        "success": True,
        "kg_thread": kg_thread,  # Return thread for later joining
    }


def store_doc(docs: list[Document], batch_size: int = 350, resume: bool = True, max_workers: int | None = None):
    """
    Store documents with hybrid (dense + sparse) vectors using parallel batch processing.

    Multiple batches are processed concurrently using ThreadPoolExecutor. Within each batch,
    vector storage (embeddings + Qdrant) and KG extraction run in parallel since KG only
    needs the original text.

    Args:
        docs: List of Document objects to store
        batch_size: Number of documents to store per batch (default: 350)
        resume: If True, continue from existing collection (don't delete)
        max_workers: Number of parallel workers (default: INDEXING_MAX_WORKERS env var, or 4)
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

    # Configure parallel workers
    if max_workers is None:
        max_workers = DEFAULT_MAX_WORKERS

    print(f"Parallel indexing with {max_workers} workers (batch_size={batch_size})")

    # Thread-safe progress tracking
    progress_lock = threading.Lock()
    processed_count = [start_index]  # List for mutability in closure

    def update_progress(delta: int):
        """Thread-safe progress update."""
        with progress_lock:
            processed_count[0] += delta
            pbar.update(delta)
            # Save progress periodically (every batch_size * workers)
            if processed_count[0] % (batch_size * max_workers) == 0 or processed_count[0] == total:
                save_progress(processed_count[0])

    # Prepare batches as (batch_docs, start_idx) tuples
    batches = [
        (docs[i:i + batch_size], i)
        for i in range(start_index, total, batch_size)
    ]

    # Track KG threads for cleanup
    kg_threads: list[threading.Thread] = []
    kg_lock = threading.Lock()

    # Progress bar with tqdm
    with tqdm(total=total - start_index, desc="Indexing", unit="doc",
              initial=start_index, ncols=120) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(_process_batch, batch, start_idx, batch_size): (start_idx, len(batch))
                for batch, start_idx in batches
            }

            # Process completed batches
            for future in as_completed(future_to_batch):
                batch_start_idx, batch_len = future_to_batch[future]
                try:
                    result = future.result()
                    if result["success"]:
                        update_progress(result["count"])
                        # Collect KG thread for later joining
                        if result.get("kg_thread"):
                            with kg_lock:
                                kg_threads.append(result["kg_thread"])
                    else:
                        # Handle failure - save progress and raise
                        save_progress(processed_count[0])
                        raise result["error"]
                except Exception as e:
                    save_progress(processed_count[0])
                    raise

    # Wait for all KG threads to complete before returning
    if kg_threads:
        alive_count = sum(1 for t in kg_threads if t.is_alive())
        print(f"\nWaiting for {alive_count}/{len(kg_threads)} KG extraction threads to complete...")
        # Join ALL threads (not just alive ones, to handle race conditions)
        for i, thread in enumerate(kg_threads, 1):
            if thread.is_alive():
                thread.join()  # No timeout - wait until truly complete
                print(f"  Thread {i}/{len(kg_threads)} done", end="\r")
        print(f"\nAll {len(kg_threads)} KG threads complete!")

    # Delete progress file on completion
    if progress_file.exists():
        progress_file.unlink()

    elapsed = time.time() - start
    print(f"\nFinished storing {total} documents in {elapsed:.1f}s ({total/elapsed:.0f} docs/min)")

