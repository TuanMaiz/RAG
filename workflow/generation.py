from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from vector_stores.qdrant import get_vector_store


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    # Get the last user message
    last_query = request.state["messages"][-1].content

    # Retrieve relevant documents
    vector_store = get_vector_store()
    retrieved_docs = vector_store.similarity_search(last_query, k=4)

    docs_content = "\n\n".join(f"[{i+1}] {doc.page_content}" for i, doc in enumerate(retrieved_docs))

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )
    
    print(f"promp_with_context", system_message)

    return system_message


def create_rag_agent(model):
    """Create a RAG agent with context injection."""
    from langchain.agents import create_agent

    return create_agent(model, tools=[], middleware=[prompt_with_context])


def query(query: str, model) -> str:
    """
    Query the RAG system.

    Args:
        query: User query
        model: LLM model

    Returns:
        Response text
    """
    agent = create_rag_agent(model)

    config = RunnableConfig()
    response = agent.invoke({"messages": [{"role": "user", "content": query}]}, config)

    return response["messages"][-1].content
