import os
from typing import Literal

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Qdrant client configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "rag_documents")

# Create client
if QDRANT_API_KEY:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    client = QdrantClient(url=QDRANT_URL)


def get_vector_store(embedding_model: Literal["openai"] = "openai") -> QdrantVectorStore:
    """
    Get or create Qdrant vector store.

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
__all__ = ["client", "embeddings", "get_vector_store", "COLLECTION_NAME"]