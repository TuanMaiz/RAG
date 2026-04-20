"""Cross-Encoder Reranker for high-precision document ranking."""

import os
from typing import Any

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from utils.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)

# Configuration
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"

# Lazy initialization of the model
_reranker_model = None


def get_reranker():
    """Get or create CrossEncoder instance (lazy initialization)."""
    global _reranker_model
    if _reranker_model is None and ENABLE_RERANK:
        try:
            logger.info("Loading Reranker model: %s...", RERANK_MODEL_NAME)
            _reranker_model = CrossEncoder(RERANK_MODEL_NAME)
            logger.info("Reranker model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load Reranker model: %s", e)
            return None
    return _reranker_model


def rerank_documents(query: str, documents: list[Any], top_k: int = 5) -> tuple[list[Any], float]:
    """
    Rerank a list of documents relative to a query using Cross-Encoder.

    Args:
        query: The user query or rewritten query
        documents: List of Document objects to rerank
        top_k: Number of top documents to return after reranking

    Returns:
        Tuple of (Sorted list of top_k Document objects, max_score)
    """
    if not documents:
        return [], 0.0

    if not ENABLE_RERANK:
        # If disabled, return original order and a dummy high score 
        # (or 0.0 if you want IDK to trigger more often when rerank is off)
        return documents[:top_k], 0.0

    model = get_reranker()
    if model is None:
        return documents[:top_k], 0.0

    try:
        # Prepare pairs for Cross-Encoder [query, document_text]
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Predict scores
        logger.debug("Reranking %d documents...", len(documents))
        scores = model.predict(pairs)
        
        # Attach scores and sort
        doc_score_pairs = list(zip(documents, scores))
        # Sort by score descending
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        max_score = float(doc_score_pairs[0][1])
        
        # Log top score for debugging
        logger.info("Rerank complete. Max score: %.4f", max_score)
            
        # Return top K docs and the max score
        return [doc for doc, score in doc_score_pairs[:top_k]], max_score

    except Exception as e:
        logger.error("Error during reranking: %s", e)
        return documents[:top_k], 0.0
