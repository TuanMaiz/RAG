"""RAG generation with query rewriting for multi-turn conversations."""

import os
from dotenv import load_dotenv

from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

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

load_dotenv()

# KG integration flag
ENABLE_KG = os.getenv("ENABLE_KG", "true").lower() == "true"

# LLM for rewriting and judging (can use same as generation or cheaper model)
_rewrite_llm = ChatOpenAI(
    model=os.getenv("OPENAI_LLM_MODEL", "openai/gpt-4o-mini"),
    temperature=0,
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# IDK Detection settings
IDK_SCORE_THRESHOLD = 0.5


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

    response = _rewrite_llm.invoke(prompt).content.strip().lower()
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

    response = _rewrite_llm.invoke(prompt).content.strip()
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
               and max_score is the highest similarity score
    """
    # Determine the query to use for retrieval
    retrieval_query = query
    if history and should_rewrite(query, history):
        retrieval_query = rewrite_query(query, history)
        print(f"[Query Rewritten] '{query}' → '{retrieval_query}'")
    else:
        print(f"[Query Standalone] '{query}'")

    # Duplicate for better retrieval (still helps with hybrid)
    duplicated = duplicate_query(retrieval_query)

    # Use hybrid retrieval
    return hybrid_retrieve(duplicated, history=None, k=k)


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
               and max_score is the highest similarity score
    """
    # Determine the query to use for retrieval
    retrieval_query = query
    if history and should_rewrite(query, history):
        retrieval_query = rewrite_query(query, history)
        print(f"[Query Rewritten] '{query}' → '{retrieval_query}'")
    else:
        print(f"[Query Standalone] '{query}'")

    # Duplicate for better retrieval
    duplicated = duplicate_query(retrieval_query)

    # Vector search (hybrid dense + sparse)
    vector_docs, max_score = hybrid_retrieve(duplicated, history=None, k=k)

    # Graph search (if enabled)
    graph_doc_ids = []
    if ENABLE_KG:
        try:
            graph_doc_ids = graph_search(
                query=retrieval_query,
                domain=None,  # Auto-infer from entities
                k=k,
                expand=True,
                max_hops=1
            )
            if graph_doc_ids:
                print(f"[Graph Search] Found {len(graph_doc_ids)} document IDs: {graph_doc_ids}")
        except Exception as e:
            print(f"[Graph Search] Failed: {e}")

    # Fuse results
    fused_docs = fuse_results(vector_docs, graph_doc_ids, k=k)

    # Track max score from vector search (graph results don't have scores)
    # This is used for IDK detection
    return fused_docs, max_score


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """
    Inject context into state messages.

    This middleware retrieves docs that were already fetched
    (stored in state by query() function) and injects them.
    """
    # Get pre-retrieved docs from state (set by query() function)
    retrieved_docs = request.state.get("retrieved_docs", [])

    docs_content = "\n\n".join(
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(retrieved_docs)
    )

    # Get the user question from state
    messages = request.state.get("messages", [])
    question = messages[-1].get("content", "") if messages else ""

    system_message = RAG_SYSTEM_PROMPT.format(
        docs_content=docs_content,
        question=question
    )

    return system_message


def create_rag_agent(model):
    """Create a RAG agent with context injection."""
    from langchain.agents import create_agent

    return create_agent(model, tools=[], middleware=[prompt_with_context])


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

    # Retrieve with KG enhancement if enabled
    if ENABLE_KG:
        print("[KG Retrieval] Using vector + graph search")
        retrieved_docs, max_score = retrieve_with_kg(query, history, k=5)
    else:
        print("[Vector Retrieval] Using hybrid search only")
        retrieved_docs, max_score = retrieve_with_scores(query, history, k=5)

    print(f"[Retrieved] {len(retrieved_docs)} documents, max_score={max_score:.3f}")

    # IDK Detection: if best match is below threshold, return IDK message
    if max_score < IDK_SCORE_THRESHOLD:
        print(f"[IDK: max_score {max_score:.3f} < threshold {IDK_SCORE_THRESHOLD}]")
        # Still save to memory so conversation continues
        memory.add_turn(query, IDK_MESSAGE)
        return IDK_MESSAGE

    # Generate response using retrieved docs
    agent = create_rag_agent(model)

    # Pass retrieved docs to the agent via state
    config = RunnableConfig()
    state = {
        "messages": [{"role": "user", "content": query}],
        "conversation_history": history,
        "retrieved_docs": retrieved_docs,
    }

    response = agent.invoke(state, config)

    # Save turn to memory
    response_text = response["messages"][-1].content
    memory.add_turn(query, response_text)

    return response_text
