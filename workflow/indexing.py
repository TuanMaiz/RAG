from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loaders.document_loader import JSONLLoader


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

def store_doc(docs: list[Document], batch_size: int = 1000):
    """
    Store documents in Qdrant vector store.

    Args:
        doc: List of Document objects to store
        batch_size: Number of documents to store per batch
    """
    import time
    from vector_stores.qdrant import get_vector_store

    vector_store = get_vector_store()

    # Store in batches
    total = len(docs)
    start = time.time()

    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        vector_store.add_documents(batch)
        elapsed = time.time() - start
        progress = min(i + batch_size, total) / total * 100
        print(f"Stored {min(i + batch_size, total)}/{total} ({progress:.1f}%) - {elapsed:.1f}s")

    print(f"Finished storing {total} documents in {time.time() - start:.2f}s")

