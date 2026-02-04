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


def store_doc(docs: list[Document], batch_size: int = 300):
    """
    Store documents with hybrid (dense + sparse) vectors.

    Args:
        docs: List of Document objects to store
        batch_size: Number of documents to store per batch
    """
    from qdrant_client.http.models import PointStruct, SparseVector

    # Ensure hybrid collection exists
    if client.collection_exists(COLLECTION_NAME):
        # Recreate with hybrid schema
        client.delete_collection(COLLECTION_NAME)
    create_hybrid_collection()

    total = len(docs)
    start = time.time()

    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        batch_start = time.time()

        # Batch embeddings - much faster than one-by-one
        texts = [doc.page_content for doc in batch]
        print(f"  Embedding {len(batch)} documents...", end="", flush=True)
        dense_vectors = embeddings.embed_documents(texts)
        print(f" done ({time.time() - batch_start:.1f}s)")

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
                    raise
                print(f"  Retry {retry + 1}/{max_retries} after error: {e}")
                time.sleep(5)
        print(f"  Upserted in {time.time() - upsert_start:.1f}s")

        elapsed = time.time() - start
        progress = min(i + batch_size, total) / total * 100
        eta = (elapsed / (i + len(batch))) * (total - i - len(batch))
        print(f"Stored {min(i + batch_size, total)}/{total} ({progress:.1f}%) - {elapsed:.1f}s - ETA: {eta/60:.1f}min")

    print(f"Finished storing {total} hybrid documents in {time.time() - start:.2f}s")

