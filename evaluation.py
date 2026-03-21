#!/usr/bin/env python3
"""mtRAG Task C (RAG) evaluation script.

Reads mtRAG RAG.jsonl format, runs our RAG system, outputs predictions.
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm

from model import get_llm

from workflow.generation import (
    retrieve_with_kg,
    should_rewrite,
    rewrite_query,
    duplicate_query,
    IDK_SCORE_THRESHOLD,
    IDK_MESSAGE,
    create_rag_agent,
    ENABLE_KG,
)
from workflow.hybrid_retrieval import hybrid_search
from workflow.memory import ConversationMemory
from vector_stores.qdrant import COLLECTION_NAME, client
from langchain_core.runnables import RunnableConfig


# Dataset prefixes for filtering
DATASETS = {
    "1": ("clapnq", "clapnq_"),
    "2": ("cloud", "ibmcld_"),
    "3": ("fiqa", "fiqa_"),
    "4": ("govt", "govt_"),
    "5": ("all", None),  # All datasets
}


def filter_tasks_by_prefix(input_file: str, prefix: str | None) -> list[dict]:
    """Filter tasks by document_id prefix."""
    tasks = []

    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)

            # If no prefix (all datasets), include everything
            if prefix is None:
                tasks.append(data)
                continue

            # Check if any context has the matching prefix
            has_matching_prefix = any(
                c['document_id'].startswith(prefix)
                for c in data.get('contexts', [])
            )
            if has_matching_prefix:
                tasks.append(data)

    return tasks


def generate_prediction(
    task_input: dict,
    model,
    memory: ConversationMemory,
) -> dict:
    """Generate prediction for a single task."""

    question = task_input["input"][0]["text"]
    history = memory.get_history()

    # Use KG-enhanced retrieval if enabled, otherwise use standard retrieval
    if ENABLE_KG:
        # retrieve_with_kg handles rewrite, duplicate, and fusion internally
        docs, max_score = retrieve_with_kg(question, history, k=5)
    else:
        # Manual retrieval pipeline for non-KG mode
        retrieval_query = question
        if history and should_rewrite(question, history):
            retrieval_query = rewrite_query(question, history)

        duplicated = duplicate_query(retrieval_query)
        hybrid_results = hybrid_search(duplicated, k=5)

        docs = [doc for doc, _ in hybrid_results]
        max_score = max([score for _, score in hybrid_results]) if hybrid_results else 0.0

    # Build contexts with document_id and score
    contexts = []
    for doc in docs:
        doc_id = doc.metadata.get("document_id", doc.metadata.get("_id", "unknown"))
        text = doc.page_content
        contexts.append({
            "document_id": doc_id,
            "text": text,
            "score": float(max_score),  # Use shared max_score for all
        })

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
    dataset_choice: str = "5",
):
    """Run evaluation on selected dataset.

    Args:
        input_file: Path to RAG.jsonl
        output_file: Path for predictions output
        model_name: LLM model to use
        dataset_choice: 1-5 (1=clapnq, 2=cloud, 3=fiqa, 4=govt, 5=all)
    """
    print(f"Reading input from: {input_file}")

    # Get prefix based on choice
    _, prefix = DATASETS.get(dataset_choice, (None, None))

    # Filter tasks by prefix
    tasks = filter_tasks_by_prefix(input_file, prefix)
    dataset_name = DATASETS[dataset_choice][0].upper()
    print(f"Found {len(tasks)} {dataset_name} tasks")

    # Initialize model
    model = get_llm(model_name)

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
            "Collection": task.get("Collection", ""),
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
    # Default paths
    input_file = "evaluation_dataset/human/RAG.jsonl"
    output_file = "predictions/predictions.jsonl"

    # Override with args if provided
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]

    # Dataset menu
    print("\n" + "=" * 50)
    print("mtRAG Evaluation")
    print("=" * 50)
    print("Select dataset to evaluate:")
    print("  1. Clapnq (French Revolution)")
    print("  2. Cloud (IBM Cloud)")
    print("  3. Fiqa (Financial QA)")
    print("  4. Govt (Government)")
    print("  5. ALL datasets")
    print("=" * 50)

    choice = input("\nSelect option [1-5]: ").strip()
    if choice not in DATASETS:
        print("Invalid option. Defaulting to ALL datasets.")
        choice = "5"

    dataset_name = DATASETS[choice][0].upper()
    print(f"\nRunning evaluation on: {dataset_name}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print()

    run_evaluation(input_file, output_file, dataset_choice=choice)


if __name__ == "__main__":
    main()
