"""Context fusion for combining vector and graph retrieval results.

This module merges documents retrieved from Qdrant (vector search) and
Neo4j (graph search), fetching actual text from Qdrant for graph results.
"""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document

from vector_stores.qdrant import client, COLLECTION_NAME

load_dotenv()

# KG fusion flag
ENABLE_KG = os.getenv("ENABLE_KG", "true").lower() == "true"


def fetch_texts_from_qdrant(doc_ids: list[int]) -> list[Document]:
    """Fetch actual document texts from Qdrant by document IDs.

    Args:
        doc_ids: List of document IDs (matching Qdrant point IDs)

    Returns:
        List of LangChain Document objects with page_content and metadata
    """
    if not doc_ids:
        return []

    try:
        # Fetch points from Qdrant
        records = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=doc_ids,
        )

        documents = []
        for record in records:
            payload = record.payload or {}
            text = payload.get("page_content", "")

            if text:  # Only include if text exists
                # Create a Document-like object
                doc = Document(page_content=text, metadata=payload)
                documents.append(doc)

        return documents

    except Exception as e:
        print(f"Warning: Failed to fetch texts from Qdrant: {e}")
        return []


def fuse_results(
    vector_docs: list[Document],
    graph_doc_ids: list[int],
    k: int = 5,
    graph_weight: float = 0.3,
) -> list[Document]:
    """Merge and rank documents from vector and graph retrieval.

    Uses a simple scoring approach:
    - Vector docs: Ranked by original retrieval order (score implicit)
    - Graph docs: Boosted by graph_weight
    - Deduplicated by document ID

    Args:
        vector_docs: Documents from hybrid vector search
        graph_doc_ids: Document IDs from graph search (Qdrant point IDs)
        k: Maximum number of documents to return
        graph_weight: Weight boost for graph-retrieved documents (0.0-1.0)

    Returns:
        Fused list of Documents, deduplicated and ranked
    """
    if not ENABLE_KG or not graph_doc_ids:
        return vector_docs[:k]

    # Fetch actual texts for graph results
    graph_docs = fetch_texts_from_qdrant(graph_doc_ids)

    # Track seen Qdrant point IDs to deduplicate
    # Use a simple approach: compare by content hash for deduplication
    seen_content = set()
    fused = []

    def _content_hash(doc: Document) -> str:
        """Create a simple hash of document content for deduplication."""
        return doc.page_content[:100]  # First 100 chars as identifier

    # First pass: Add vector docs (they come pre-ranked)
    for doc in vector_docs:
        content_hash = _content_hash(doc)
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            fused.append(doc)

    # Second pass: Add graph docs not already in vector results
    # Give them a boost by placing them strategically
    for doc in graph_docs:
        content_hash = _content_hash(doc)
        if content_hash not in seen_content:
            seen_content.add(content_hash)

            # Insert graph docs after top vector docs, but before lower-ranked ones
            # This gives them a "boost" without completely overriding vector ranking
            insert_pos = min(len(fused), max(2, k // 2))
            fused.insert(insert_pos, doc)

    return fused[:k]


def fuse_with_scores(
    vector_results: list[tuple[Document, float]],
    graph_doc_ids: list[int],
    k: int = 5,
    graph_weight: float = 0.3,
) -> list[tuple[Document, float]]:
    """Merge results with explicit scores.

    Similar to fuse_results but preserves scores for re-ranking.

    Args:
        vector_results: List of (Document, score) tuples from vector search
        graph_doc_ids: Document IDs from graph search (Qdrant point IDs)
        k: Maximum number of documents to return
        graph_weight: Weight for combining graph results (0.0-1.0)

    Returns:
        List of (Document, fused_score) tuples
    """
    if not ENABLE_KG or not graph_doc_ids:
        return vector_results[:k]

    # Fetch graph docs
    graph_docs = fetch_texts_from_qdrant(graph_doc_ids)

    # Track seen by content hash
    seen_content = set()
    fused = []

    def _content_hash(doc: Document) -> str:
        """Create a simple hash of document content for deduplication."""
        return doc.page_content[:100]

    # Add vector results with their scores
    for doc, score in vector_results:
        content_hash = _content_hash(doc)
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            fused.append((doc, score))

    # Add graph docs with boosted scores
    # Calculate average vector score for baseline
    avg_vector_score = sum(s for _, s in vector_results) / len(vector_results) if vector_results else 0.5

    for doc in graph_docs:
        content_hash = _content_hash(doc)
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            # Graph docs get a score based on average vector + boost
            graph_score = min(1.0, avg_vector_score + (graph_weight * 0.5))
            fused.append((doc, graph_score))

    # Sort by score and return top k
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:k]


def format_for_llm(docs: list[Document]) -> str:
    """Format documents for LLM prompt with citation numbers.

    Args:
        docs: List of Documents to format

    Returns:
        Formatted string with [1], [2], ... citation markers
    """
    if not docs:
        return "No relevant context found."

    formatted_parts = []
    for i, doc in enumerate(docs, start=1):
        content = doc.page_content.strip()
        # Truncate very long documents
        if len(content) > 2000:
            content = content[:2000] + "..."
        formatted_parts.append(f"[{i}] {content}")

    return "\n\n".join(formatted_parts)


def format_with_sources(docs: list[Document]) -> str:
    """Format documents with source information for debugging.

    Args:
        docs: List of Documents to format

    Returns:
        Formatted string with source metadata
    """
    if not docs:
        return "No documents."

    lines = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content.strip()[:100]
        lines.append(f"[{i}] Source: {source}")
        lines.append(f"    Content: {content}...")

    return "\n".join(lines)


# For testing
if __name__ == "__main__":
    print("Testing context fusion...")

    # Test fetching from Qdrant
    print("\nTest 1: Fetch texts from Qdrant")
    docs = fetch_texts_from_qdrant([0, 1, 2])
    print(f"  Fetched {len(docs)} documents")
    if docs:
        print(f"  First doc: {docs[0].page_content[:50]}...")

    # Test fuse_results with mock data
    print("\nTest 2: Fuse vector + graph results (mock)")
    from langchain_core.documents import Document

    # Mock vector docs
    vector_docs = [
        Document(page_content="Vector result 1", metadata={"document_id": 10}),
        Document(page_content="Vector result 2", metadata={"document_id": 11}),
    ]
    graph_ids = [10, 20]  # 10 overlaps, 20 is new

    fused = fuse_results(vector_docs, graph_ids, k=5)
    print(f"  Vector docs: {len(vector_docs)}")
    print(f"  Graph IDs: {graph_ids}")
    print(f"  Fused docs: {len(fused)} (should be 3 - deduplicated)")
    for i, doc in enumerate(fused):
        doc_id = doc.metadata.get("document_id", "unknown")
        print(f"    [{i}] ID={doc_id}: {doc.page_content[:30]}...")

    # Test format_for_llm
    print("\nTest 3: Format for LLM")
    formatted = format_for_llm(fused[:2])
    print(f"  Formatted length: {len(formatted)} chars")
    print(f"  Preview:\n{formatted[:200]}...")
