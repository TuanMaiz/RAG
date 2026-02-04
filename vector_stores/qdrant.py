import os
from typing import Literal

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams

load_dotenv()

# Embeddings - use model and base_url from env
embeddings = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# Qdrant client configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "rag_documents")

# Hybrid search weights
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.7"))
SPARSE_WEIGHT = 1.0 - DENSE_WEIGHT

# Create client
if QDRANT_API_KEY:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    client = QdrantClient(url=QDRANT_URL)


def create_hybrid_collection() -> None:
    """
    Create a Qdrant collection with both dense and sparse vectors for hybrid search.

    This replaces the default collection. Use this for new hybrid retrieval.
    """
    # Get vector size from embeddings
    vector_size = len(embeddings.embed_query("sample text"))

    # Delete existing collection if present
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")

    # Create collection with named vectors
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=vector_size, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(modifier="idf"),
        },
    )
    print(f"Created hybrid collection: {COLLECTION_NAME}")
    print(f"  - Dense vectors: {vector_size} dims, COSINE distance")
    print(f"  - Sparse vectors: BM25 with IDF modifier")
    print(f"  - Weights: dense={DENSE_WEIGHT}, sparse={SPARSE_WEIGHT}")


def get_vector_store(embedding_model: Literal["openai"] = "openai") -> QdrantVectorStore:
    """
    Get or create Qdrant vector store (legacy - dense only).

    For hybrid retrieval, use the client directly with named vectors.

    Args:
        embedding_model: Embedding model to use

    Returns:
        QdrantVectorStore instance
    """
    # Get vector size
    vector_size = len(embeddings.embed_query("sample text"))

    # Create collection if it doesn't exist
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Created collection: {COLLECTION_NAME}")

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


# Export for convenience
__all__ = [
    "client",
    "embeddings",
    "get_vector_store",
    "create_hybrid_collection",
    "COLLECTION_NAME",
    "DENSE_WEIGHT",
    "SPARSE_WEIGHT",
]