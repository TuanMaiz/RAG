#!/usr/bin/env python3
"""mtRAG Task C (RAG) evaluation script.

Reads mtRAG RAG.jsonl format, runs our RAG system, outputs predictions.
"""

import json
import sys
from pathlib import Path
from langchain.chat_models import init_chat_model
from tqdm import tqdm

from workflow.generation import (
    retrieve_with_scores,
    should_rewrite,
    rewrite_query,
    duplicate_query,
    IDK_SCORE_THRESHOLD,
    IDK_MESSAGE,
    create_rag_agent,
)
from workflow.memory import ConversationMemory
from vector_stores.qdrant import get_vector_store
from langchain_core.runnables import RunnableConfig


def filter_ibmcloud_questions(input_file: str) -> list[dict]:
    """Filter RAG.jsonl for IBM Cloud questions only."""
    ibmcloud_tasks = []

    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Check if any context has IBM Cloud document ID
            has_ibmcld = any(
                c['document_id'].startswith('ibmcld_')
                for c in data.get('contexts', [])
            )
            if has_ibmcld:
                ibmcloud_tasks.append(data)

    return ibmcloud_tasks


def generate_prediction(
    task_input: dict,
    model,
    memory: ConversationMemory,
) -> dict:
    """Generate prediction for a single task."""
    question = task_input["input"][0]["text"]
    history = memory.get_history()

    # Determine retrieval query (with rewrite if needed)
    retrieval_query = question
    if history and should_rewrite(question, history):
        retrieval_query = rewrite_query(question, history)

    # Duplicate for better retrieval
    duplicated = duplicate_query(retrieval_query)

    # Retrieve with scores
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(duplicated, k=5)

    # Build contexts with document_id and score
    contexts = []
    docs = []
    for doc, score in results:
        doc_id = doc.metadata.get("document_id", doc.metadata.get("_id", "unknown"))
        contexts.append({
            "document_id": doc_id,
            "text": doc.page_content,
            "score": float(score),
        })
        docs.append(doc)

    max_score = max([s for _, s in results]) if results else 0.0

    # Generate prediction or IDK
    if max_score < IDK_SCORE_THRESHOLD:
        prediction_text = IDK_MESSAGE
    else:
        # Generate using the agent
        agent = create_rag_agent(model)
        state = {
            "messages": [{"role": "user", "content": question}],
            "conversation_history": history,
            "retrieved_docs": docs,
        }

        response = agent.invoke(state, RunnableConfig())
        prediction_text = response["messages"][-1].content

    # Save to memory
    memory.add_turn(question, prediction_text)

    # Return prediction format
    result = task_input.copy()
    result["contexts"] = contexts
    result["predictions"] = [{"text": prediction_text}]
    return result


def run_evaluation(
    input_file: str,
    output_file: str,
    model_name: str = "gpt-4o-mini",
):
    """Run evaluation on IBM Cloud questions."""
    print(f"Reading input from: {input_file}")

    # Filter for IBM Cloud questions
    tasks = filter_ibmcloud_questions(input_file)
    print(f"Found {len(tasks)} IBM Cloud tasks")

    # Initialize model
    model = init_chat_model(model_name)

    # Track conversations by conversation_id
    conversations = {}

    # Generate predictions
    predictions = []

    for task in tqdm(tasks, desc="Generating predictions"):
        conv_id = task["conversation_id"]

        # Get or create memory for this conversation
        if conv_id not in conversations:
            conversations[conv_id] = ConversationMemory(window_size=5)
        memory = conversations[conv_id]

        # Create prediction input format
        task_input = {
            "conversation_id": task["conversation_id"],
            "task_id": task["task_id"],
            "Collection": "mt-rag-ibmcloud-elser-512-100-20240502",
            "input": task["input"],
        }

        # Generate prediction
        try:
            prediction = generate_prediction(task_input, model, memory)
            predictions.append(prediction)
        except Exception as e:
            print(f"\nError processing {task['task_id']}: {e}")
            # Still add a partial prediction
            predictions.append({
                **task_input,
                "contexts": [],
                "predictions": [{"text": IDK_MESSAGE}],
            })

    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

    print(f"\nWrote {len(predictions)} predictions to: {output_file}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluation.py <input.jsonl> <output.jsonl>")
        print(f"\nExample: python evaluation.py evaluation_dataset/human/RAG.jsonl predictions/output.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    run_evaluation(input_file, output_file)


if __name__ == "__main__":
    main()
