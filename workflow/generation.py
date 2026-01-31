"""RAG generation with query rewriting for multi-turn conversations."""

from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from vector_stores.qdrant import get_vector_store
from workflow.memory import ConversationMemory


# LLM for rewriting and judging (can use same as generation or cheaper model)
_rewrite_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# IDK Detection settings
IDK_SCORE_THRESHOLD = 0.5
IDK_MESSAGE = "The available documents don't contain information to answer this question. Please try a different question about the topics covered in the documentation."


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

    prompt = f"""You are a judge. Determine if the current query needs context from previous turns to be understood.

Chat history:
{history_text}

Current query: {query}

Answer ONLY "yes" or "no". Does this query need context from previous turns?"""

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

    prompt = f"""You are a query rewriter for a retrieval system. Rewrite the current query to be standalone and clear.

Chat history:
{history_text}

Current query: {query}

Rewrite the query to:
- Replace pronouns (he, she, it, they, this, that) with the actual entities
- Make it fully understandable without the chat history
- Keep it concise and natural

Return ONLY the rewritten query, nothing else."""

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
    Retrieve documents with scores for IDK detection.

    Returns:
        tuple: (docs, max_score) where docs is list of (doc, score) tuples
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

    # Retrieve with scores
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(duplicated, k=k)

    docs = [doc for doc, score in results]
    scores = [score for doc, score in results]
    max_score = max(scores) if scores else 0.0

    print(f"[Retrieved {len(docs)} docs, max_score: {max_score:.3f}]")

    return docs, max_score


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

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message


def create_rag_agent(model):
    """Create a RAG agent with context injection."""
    from langchain.agents import create_agent

    return create_agent(model, tools=[], middleware=[prompt_with_context])


def query(query: str, model, memory: ConversationMemory) -> str:
    """
    Query the RAG system with conversation memory and IDK detection.

    Args:
        query: User query
        model: LLM model
        memory: ConversationMemory instance

    Returns:
        Response text (or IDK message if no relevant docs found)
    """
    history = memory.get_history()

    # Retrieve with scores (includes rewrite logic)
    retrieved_docs, max_score = retrieve_with_scores(query, history, k=5)

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
