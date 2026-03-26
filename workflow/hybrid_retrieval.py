"""Hybrid retrieval combining dense (embeddings) and sparse (BM25) search."""

from typing import Any

import numpy as np
from qdrant_client import models
from qdrant_client.http.models import (
    NamedSparseVector,
    NamedVector,
    PointStruct,
    SparseVector,
)

from vector_stores.qdrant import (
    COLLECTION_NAME,
    DENSE_WEIGHT,
    SPARSE_WEIGHT,
    client,
    get_embeddings,
)


def tokenize_bm25(text: str) -> dict[str, float]:
    """
    BM25-style tokenization for sparse vectors.

    Returns a dict of {token: weight} for non-zero entries.
    Qdrant will compute proper IDF.
    """
    # Common stop words to filter out
    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "you", "your", "what", "when",
        "where", "who", "which", "why", "how", "this", "these", "those",
    }

    # Simple word-level tokenization (lowercase, alphanumeric)
    tokens = text.lower().split()
    # Remove non-alphanumeric chars and filter empty
    tokens = ["".join(c for c in t if c.isalnum()) for t in tokens]
    # Filter short tokens and stop words
    tokens = [t for t in tokens if len(t) > 2 and t not in STOP_WORDS]

    # Term frequency (TF) - count occurrences
    tf: dict[str, float] = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1

    # TF normalization
    max_tf = max(tf.values()) if tf else 1
    return {token: count / max_tf for token, count in tf.items()}


def sparse_vector_from_text(text: str) -> SparseVector:
    """
    Convert text to Qdrant sparse vector format.

    Qdrant uses integer hashes for keys to save space.
    """
    tokens_weights = tokenize_bm25(text)

    # Hash tokens to integers
    indices = [hash(token) & 0xFFFFFFFF for token in tokens_weights.keys()]
    values = list(tokens_weights.values())

    return SparseVector(indices=indices, values=values)


def hybrid_search(
    query: str,
    k: int = 5,
    dense_weight: float = DENSE_WEIGHT,
    sparse_weight: float = SPARSE_WEIGHT,
) -> list[tuple[Any, float]]:
    """
    Perform hybrid search combining dense and sparse retrieval.

    Uses Qdrant's prefetch + fusion for efficient hybrid search.

    Args:
        query: Search query text
        k: Number of results to return
        dense_weight: Weight for dense (embedding) search
        sparse_weight: Weight for sparse (BM25) search

    Returns:
        List of (document, score) tuples sorted by hybrid score
    """
    # Get dense embedding
    dense_vector = get_embeddings().embed_query(query)

    # Get sparse vector
    sparse_vector = sparse_vector_from_text(query)

    # Prefetch: get top candidates from both dense and sparse
    prefetch = [
        models.Prefetch(
            query=dense_vector,
            using="dense",
            limit=50,  # Get more candidates, then fuse
        ),
        models.Prefetch(
            query=sparse_vector,
            using="sparse",
            limit=50,
        ),
    ]

    # Query with fusion (Reciprocal Rank Fusion)
    # dense_weight controls semantic search influence, sparse_weight controls keyword search
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
        limit=k,
        with_payload=True,
        with_vectors=False,
    )

    # Convert to (doc, score) format
    # Extract text from payload
    output = []
    for result in results.points:
        payload = result.payload or {}
        text = payload.get("page_content", "")
        # Create a simple document-like object
        doc = type("Document", (), {"page_content": text, "metadata": payload})()
        output.append((doc, result.score))

    return output


def similarity_search_with_score(
    query: str,
    k: int = 5,
    dense_weight: float = DENSE_WEIGHT,
    sparse_weight: float = SPARSE_WEIGHT,
) -> list[tuple[Any, float]]:
    """
    Alias for hybrid_search to maintain compatibility with existing code.

    This allows swapping in hybrid retrieval without changing call sites.
    """
    return hybrid_search(query, k=k, dense_weight=dense_weight, sparse_weight=sparse_weight)


# Store documents with hybrid vectors
def store_hybrid_documents(docs: list[Any], batch_size: int = 1000) -> None:
    """
    Store documents with both dense and sparse vectors.

    Args:
        docs: List of LangChain Document objects
        batch_size: Batch size for upsert
    """
    from qdrant_client.http.models import PointStruct

    total = len(docs)

    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]

        points = []
        for idx, doc in enumerate(batch):
            doc_id = i + idx

            # Dense embedding
            dense_vector = get_embeddings().embed_query(doc.page_content)

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
                    "dense": dense_vector,
                },
                sparse_vector={
                    "sparse": sparse_vector,
                },
                payload=payload,
            )
            points.append(point)

        # Batch upsert
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        print(f"Stored {i + len(points)}/{total} documents with hybrid vectors")


def retrieve_with_scores(
    query: str,
    history: list[dict[str, str]] | None = None,
    k: int = 5,
) -> tuple[list[Any], float]:
    """
    Retrieve documents with hybrid search, returning docs and max score.

    Compatible with the retrieve_with_scores signature in generation.py.

    Args:
        query: Search query
        history: Conversation history (currently unused, for future context)
        k: Number of documents to retrieve

    Returns:
        Tuple of (documents, max_score)
    """
    results = hybrid_search(query, k=k)

    docs = [doc for doc, score in results]
    scores = [score for doc, score in results]
    max_score = max(scores) if scores else 0.0

    print(f"[Hybrid Retrieved {len(docs)} docs, max_score: {max_score:.3f}]")

    return docs, max_score
