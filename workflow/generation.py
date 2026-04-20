"""RAG generation with query rewriting for multi-turn conversations."""

import os
import re
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from utils.logging_config import get_logger
from workflow.context_fusion import fuse_results, format_for_llm
from workflow.graph_retrieval import graph_search
from workflow.hybrid_retrieval import retrieve_with_scores as hybrid_retrieve
from workflow.memory import ConversationMemory
from workflow.prompts import (
    IDK_MESSAGE,
    QUERY_REWRITE_PROMPT,
    REWRITE_JUDGE_PROMPT,
    RAG_SYSTEM_PROMPT,
)
from workflow.reranker import rerank_documents, ENABLE_RERANK

load_dotenv()

# KG integration flag
ENABLE_KG = os.getenv("ENABLE_KG", "true").lower() == "true"

# Lazy LLM for rewriting and judging
_rewrite_llm: ChatOpenAI | None = None

logger = get_logger(__name__)


def _get_rewrite_llm() -> ChatOpenAI:
    """Get or create rewrite LLM instance (lazy initialization)."""
    global _rewrite_llm
    if _rewrite_llm is None:
        _rewrite_llm = ChatOpenAI(
            model=os.getenv("OPENAI_QUERY_MODEL", os.getenv("OPENAI_LLM_MODEL", "openai/gpt-4o-mini")),
            temperature=0,
            openai_api_base=os.getenv("OPENAI_BASE_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _rewrite_llm

# IDK Detection settings
# Vector score threshold (usually 0.4-0.6)
IDK_SCORE_THRESHOLD = float(os.getenv("IDK_SCORE_THRESHOLD", "0.5"))
# Rerank score threshold (BGE v2 m3 scores are typically > 0 for relevant docs)
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.0"))


def should_rewrite(query: str, history: list[dict[str, str]]) -> bool:
    """
    Judge if query needs rewriting based on conversation history.

    Returns True if the query appears to depend on previous context
    (pronouns, references like "that", "the former", etc.).
    """
    if not history:
        return False

    history_text = "\n".join(
        f"Q: {h['query']}\nA: {h['response']}" for h in history[-5:]
    )

    prompt = REWRITE_JUDGE_PROMPT.format(
        history_text=history_text,
        query=query
    )

    response = _get_rewrite_llm().invoke(prompt).content.strip().lower()
    return response.startswith("y")


def rewrite_query(query: str, history: list[dict[str, str]]) -> str:
    """
    Rewrite query to be standalone using conversation history.

    The rewritten query should contain all necessary context
    (replace pronouns, add subject, etc.).
    """
    if not history:
        return query

    history_text = "\n".join(
        f"Q: {h['query']}\nA: {h['response']}" for h in history[-5:]
    )

    prompt = QUERY_REWRITE_PROMPT.format(
        history_text=history_text,
        query=query
    )

    response = _get_rewrite_llm().invoke(prompt).content.strip()
    return response


def duplicate_query(query: str) -> str:
    """
    Duplicate query for better retrieval (embedding-weighted).

    Paper finding: Duplicating the query gives it more weight
    in similarity scoring, improving retrieval performance.
    """
    return f"{query} {query}"


def retrieve_with_scores(query: str, history: list[dict[str, str]], k: int = 5):
    """
    Retrieve documents with hybrid (dense + sparse) search for IDK detection.

    Returns:
        tuple: (docs, max_score) where docs is list of Document objects
               and max_score is the highest similarity score (vector or rerank)
    """
    # Determine the query to use for retrieval
    retrieval_query = query
    if history and should_rewrite(query, history):
        retrieval_query = rewrite_query(query, history)
        logger.debug("├─ Query rewritten: '%s' → '%s'", query, retrieval_query)
    else:
        logger.debug("├─ Query: '%s' (standalone)", query)

    # Duplicate for better retrieval
    duplicated = duplicate_query(retrieval_query)

    # Use hybrid retrieval (fetch more docs for reranking)
    fetch_k = k * 2 if ENABLE_RERANK else k
    retrieved_docs, vector_max_score = hybrid_retrieve(duplicated, history=None, k=fetch_k)
    
    # Apply reranking if enabled
    reranked_docs, rerank_max_score = rerank_documents(retrieval_query, retrieved_docs, top_k=k)
    
    # Logic: if rerank is enabled, use rerank score for IDK, else use vector score
    final_score = rerank_max_score if ENABLE_RERANK else vector_max_score
    
    return reranked_docs, final_score


def retrieve_with_kg(query: str, history: list[dict[str, str]], k: int = 5):
    """
    Retrieve documents using both vector and knowledge graph search.

    Combines hybrid retrieval (Qdrant) with graph-based retrieval (Neo4j)
    and fuses the results for enhanced context.

    Args:
        query: User query
        history: Conversation history
        k: Maximum number of documents to return

    Returns:
        tuple: (docs, max_score) where docs is list of Document objects
               and max_score is the highest similarity score (vector or rerank)
    """
    # Determine the query to use for retrieval
    retrieval_query = query
    if history and should_rewrite(query, history):
        retrieval_query = rewrite_query(query, history)
        logger.debug("├─ Query rewritten: '%s' → '%s'", query, retrieval_query)
    else:
        logger.debug("├─ Query: '%s' (standalone)", query)

    # Duplicate for better retrieval
    duplicated = duplicate_query(retrieval_query)

    # Vector search (hybrid dense + sparse)
    fetch_k = k * 2 if ENABLE_RERANK else k
    vector_docs, vector_max_score = hybrid_retrieve(duplicated, history=None, k=fetch_k)

    # Graph search (if enabled)
    graph_doc_ids = []
    if ENABLE_KG:
        try:
            graph_doc_ids = graph_search(
                query=retrieval_query,
                title=None,
                k=k,
                expand=True,
                max_hops=1
            )
            if graph_doc_ids:
                logger.debug("├─ Graph: found %d docs", len(graph_doc_ids))
        except Exception as e:
            logger.warning("├─ Graph search failed: %s", e)

    # Fuse results
    fused_docs = fuse_results(vector_docs, graph_doc_ids, k=fetch_k)
    
    # Apply reranking
    reranked_docs, rerank_max_score = rerank_documents(retrieval_query, fused_docs, top_k=k)

    # Use rerank score if enabled, else vector score
    final_score = rerank_max_score if ENABLE_RERANK else vector_max_score
    
    return reranked_docs, final_score


def query(query: str, model, memory: ConversationMemory) -> str:
    """
    Query the RAG system with conversation memory and IDK detection.

    Uses knowledge graph-enhanced retrieval when ENABLE_KG=true.

    Args:
        query: User query
        model: LLM model
        memory: ConversationMemory instance

    Returns:
        Response text (or IDK message if no relevant docs found)
    """
    history = memory.get_history()

    # ─── Retrieval ────────────────────────────────────────────────────────────
    k = 10
    if ENABLE_KG:
        retrieved_docs, max_score = retrieve_with_kg(query, history, k=k)
        logger.info("└─ Mode: Vector + Graph search")
    else:
        retrieved_docs, max_score = retrieve_with_scores(query, history, k=k)
        logger.info("└─ Mode: Hybrid search (dense + sparse)")

    # IDK Detection based on the selected scoring method
    threshold = RERANK_SCORE_THRESHOLD if ENABLE_RERANK else IDK_SCORE_THRESHOLD
    
    if max_score < threshold:
        logger.warning("└─ IDK triggered: score %.3f < threshold %.3f", max_score, threshold)
        memory.add_turn(query, IDK_MESSAGE)
        return IDK_MESSAGE

    logger.info("└─ Retrieved: %d docs | Confidence score: %.3f", len(retrieved_docs), max_score)

    # ─── Context (debug level to avoid clutter) ───────────────────────────────
    docs_content = "\n\n".join(
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(retrieved_docs)
    )

    logger.debug("┌─ Context provided to LLM (%d docs)", len(retrieved_docs))
    for i, doc in enumerate(retrieved_docs):
        preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        logger.debug("│ [%d] %s", i + 1, preview)
    logger.debug("└─ End context")

    # Build the prompt directly
    system_prompt = RAG_SYSTEM_PROMPT.format(
        docs_content=docs_content,
        question=query
    )

    # Invoke model directly with system prompt
    # Note: question is already in system_prompt, so HumanMessage is just a trigger
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please provide your answer based on the context above."),
    ]

    response = model.invoke(messages)
    response_text = response.content

    # ─── Append References ─────────────────────────────────────────────────────
    # Extract unique citation numbers used in response
    citations = sorted(set(int(m) for m in re.findall(r'\[(\d+)\]', response_text)))

    if citations:
        references = "\n\n---\n**References:**\n"
        for i, doc in enumerate(retrieved_docs):
            if i + 1 in citations:
                # Get source info if available
                source = doc.metadata.get("source", "Unknown source")
                # Show full text
                references += f"\n[{i+1}] {source}\n    {doc.page_content}\n"
        response_text += references

    # Save turn to memory
    memory.add_turn(query, response_text)

    return response_text
