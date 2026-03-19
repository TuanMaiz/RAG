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

