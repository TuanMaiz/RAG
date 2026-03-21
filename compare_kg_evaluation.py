#!/usr/bin/env python3
"""Knowledge Graph comparison evaluation.

Compares RAG system performance with and without Knowledge Graph enhancement.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Import ENABLE_KG flag to toggle
from dotenv import load_dotenv

from model import get_llm

load_dotenv()

# Dataset prefixes for filtering
DATASETS = {
    "1": ("clapnq", "clapnq_"),
    "2": ("cloud", "ibmcld_"),
    "3": ("fiqa", "fiqa_"),
    "4": ("govt", "govt_"),
    "5": ("all", None),
}


def filter_tasks_by_prefix(input_file: str, prefix: str | None) -> list[dict]:
    """Filter tasks by document_id prefix."""
    tasks = []

    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)

            if prefix is None:
                tasks.append(data)
                continue

            has_matching_prefix = any(
                c['document_id'].startswith(prefix)
                for c in data.get('contexts', [])
            )
            if has_matching_prefix:
                tasks.append(data)

    return tasks


def evaluate_with_config(
    input_file: str,
    output_file: str,
    model_name: str = "gpt-4o-mini",
    dataset_choice: str = "5",
    enable_kg: bool = True,
) -> dict[str, Any]:
    """Run evaluation with specific KG configuration.

    Returns:
        dict with metrics: tasks_processed, time_taken, idk_count
    """
    # Set ENABLE_KG environment variable
    os.environ["ENABLE_KG"] = "true" if enable_kg else "false"

    # Force reload of modules to pick up new ENABLE_KG value
    import importlib
    import workflow.generation as gen_mod
    importlib.reload(gen_mod)

    # Import common functions
    from workflow.generation import (
        should_rewrite,
        rewrite_query,
        duplicate_query,
        IDK_SCORE_THRESHOLD,
        IDK_MESSAGE,
        create_rag_agent,
    )

    # Choose retrieval function based on KG flag
    if enable_kg:
        from workflow.generation import retrieve_with_kg
    else:
        # Create fallback for non-KG mode
        from workflow.hybrid_retrieval import hybrid_search
        def retrieve_with_kg(query, history, k=5):
            """Fallback to hybrid search when KG is disabled."""
            retrieval_query = query
            if history and should_rewrite(query, history):
                retrieval_query = rewrite_query(query, history)
            duplicated = duplicate_query(retrieval_query)
            results = hybrid_search(duplicated, k=k)
            docs = [doc for doc, _ in results]
            max_score = max([score for _, score in results]) if results else 0.0
            return docs, max_score

    from workflow.hybrid_retrieval import hybrid_search
    from workflow.memory import ConversationMemory
    from langchain_core.runnables import RunnableConfig

    print(f"\n{'=' * 60}")
    print(f"Evaluation: KG={'ENABLED' if enable_kg else 'DISABLED'}")
    print(f"{'=' * 60}")

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
    idk_count = 0
    start_time = time.time()

    for task in tqdm(tasks, desc=f"Generating (KG={enable_kg})"):
        conv_id = task["conversation_id"]

        # Get or create memory for this conversation
        if conv_id not in conversations:
            conversations[conv_id] = ConversationMemory(window_size=5)
        memory = conversations[conv_id]

        question = task["input"][0]["text"]
        history = memory.get_history()

        # Determine retrieval query
        retrieval_query = question
        if history and should_rewrite(question, history):
            retrieval_query = rewrite_query(question, history)

        # Duplicate for better retrieval
        duplicated = duplicate_query(retrieval_query)

        # Retrieval based on config
        if enable_kg:
            # Use KG-enhanced retrieval
            from workflow.graph_retrieval import graph_search
            from workflow.context_fusion import fetch_texts_from_qdrant

            # Vector search
            hybrid_results = hybrid_search(duplicated, k=5)
            vector_docs = [doc for doc, _ in hybrid_results]
            max_score = max([score for _, score in hybrid_results]) if hybrid_results else 0.0

            # Graph search
            graph_doc_ids = []
            try:
                graph_doc_ids = graph_search(
                    query=retrieval_query,
                    domain=None,
                    k=5,
                    expand=True,
                )
            except Exception:
                pass

            # Fuse results
            from workflow.context_fusion import fuse_results
            docs = fuse_results(vector_docs, graph_doc_ids, k=5)
        else:
            # Vector-only search
            hybrid_results = hybrid_search(duplicated, k=5)
            docs = [doc for doc, _ in hybrid_results]
            max_score = max([score for _, score in hybrid_results]) if hybrid_results else 0.0

        # Build contexts
        contexts = []
        for doc in docs:
            doc_id = doc.metadata.get("document_id", doc.metadata.get("_id", "unknown"))
            text = doc.page_content
            contexts.append({
                "document_id": doc_id,
                "text": text,
                "score": float(max_score),
            })

        # Generate prediction or IDK
        if max_score < IDK_SCORE_THRESHOLD:
            prediction_text = IDK_MESSAGE
            idk_count += 1
        else:
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

        # Build result
        predictions.append({
            "task_id": task["task_id"],
            "conversation_id": task["conversation_id"],
            "Collection": task.get("Collection", ""),
            "input": task["input"],
            "contexts": contexts,
            "predictions": [{"text": prediction_text}],
        })

    elapsed = time.time() - start_time

    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

    return {
        "tasks_processed": len(predictions),
        "time_taken": elapsed,
        "idk_count": idk_count,
        "output_file": str(output_path),
    }


def compare_predictions(
    kg_file: str,
    no_kg_file: str,
) -> dict[str, Any]:
    """Compare two prediction files.

    Returns:
        dict with comparison metrics
    """
    kg_preds = {}
    no_kg_preds = {}

    # Read KG predictions
    with open(kg_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            task_id = data["task_id"]
            kg_preds[task_id] = data

    # Read no-KG predictions
    with open(no_kg_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            task_id = data["task_id"]
            no_kg_preds[task_id] = data

    # Compare
    different_responses = 0
    same_responses = 0
    kg_only_answered = 0
    no_kg_only_answered = 0

    for task_id in kg_preds:
        kg_pred = kg_preds[task_id]
        kg_text = kg_pred["predictions"][0]["text"]
        kg_idk = "couldn't find relevant information" in kg_text.lower()

        if task_id in no_kg_preds:
            no_kg_pred = no_kg_preds[task_id]
            no_kg_text = no_kg_pred["predictions"][0]["text"]
            no_kg_idk = "couldn't find relevant information" in no_kg_text.lower()

            if kg_text == no_kg_text:
                same_responses += 1
            else:
                different_responses += 1

            # Check who answered
            if not kg_idk and no_kg_idk:
                kg_only_answered += 1
            elif kg_idk and not no_kg_idk:
                no_kg_only_answered += 1

    # Context comparison
    avg_contexts_kg = sum(len(p["contexts"]) for p in kg_preds.values()) / len(kg_preds)
    avg_contexts_no_kg = sum(len(p["contexts"]) for p in no_kg_preds.values()) / len(no_kg_preds)

    return {
        "total_tasks": len(kg_preds),
        "same_responses": same_responses,
        "different_responses": different_responses,
        "kg_only_answered": kg_only_answered,
        "no_kg_only_answered": no_kg_only_answered,
        "avg_contexts_kg": avg_contexts_kg,
        "avg_contexts_no_kg": avg_contexts_no_kg,
    }


def main():
    input_file = "evaluation_dataset/human/RAG.jsonl"

    # Dataset menu
    print("\n" + "=" * 50)
    print("Knowledge Graph Comparison Evaluation")
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
    print(f"\nRunning comparison on: {dataset_name}")

    # Output files
    timestamp = int(time.time())
    kg_output = f"predictions/kg_{choice}_{timestamp}.jsonl"
    no_kg_output = f"predictions/no_kg_{choice}_{timestamp}.jsonl"

    # Run evaluation WITHOUT KG
    print("\n" + "-" * 50)
    print("Step 1: Running WITHOUT Knowledge Graph...")
    print("-" * 50)
    no_kg_metrics = evaluate_with_config(
        input_file=input_file,
        output_file=no_kg_output,
        dataset_choice=choice,
        enable_kg=False,
    )

    # Run evaluation WITH KG
    print("\n" + "-" * 50)
    print("Step 2: Running WITH Knowledge Graph...")
    print("-" * 50)
    kg_metrics = evaluate_with_config(
        input_file=input_file,
        output_file=kg_output,
        dataset_choice=choice,
        enable_kg=True,
    )

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    comparison = compare_predictions(kg_output, no_kg_output)

    print(f"\nDataset: {dataset_name}")
    print(f"Total Tasks: {comparison['total_tasks']}")
    print(f"\n--- Response Differences ---")
    print(f"Same responses: {comparison['same_responses']} ({100*comparison['same_responses']/comparison['total_tasks']:.1f}%)")
    print(f"Different responses: {comparison['different_responses']} ({100*comparison['different_responses']/comparison['total_tasks']:.1f}%)")
    print(f"\n--- Answerability ---")
    print(f"KG answered, No-KG didn't: {comparison['kg_only_answered']}")
    print(f"No-KG answered, KG didn't: {comparison['no_kg_only_answered']}")
    print(f"\n--- Context Stats ---")
    print(f"Avg contexts (KG): {comparison['avg_contexts_kg']:.2f}")
    print(f"Avg contexts (No-KG): {comparison['avg_contexts_no_kg']:.2f}")
    print(f"\n--- Performance ---")
    print(f"Time (KG): {kg_metrics['time_taken']:.1f}s ({kg_metrics['time_taken']/60:.1f}min)")
    print(f"Time (No-KG): {no_kg_metrics['time_taken']:.1f}s ({no_kg_metrics['time_taken']/60:.1f}min)")
    print(f"IDK count (KG): {kg_metrics['idk_count']}")
    print(f"IDK count (No-KG): {no_kg_metrics['idk_count']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if comparison['kg_only_answered'] > comparison['no_kg_only_answered']:
        improvement = comparison['kg_only_answered'] - comparison['no_kg_only_answered']
        print(f"✓ KG helped answer {improvement} more questions!")
    elif comparison['no_kg_only_answered'] > comparison['kg_only_answered']:
        regression = comparison['no_kg_only_answered'] - comparison['kg_only_answered']
        print(f"✗ KG caused {regression} more IDK responses.")

    print(f"\nOutput files:")
    print(f"  KG:      {kg_output}")
    print(f"  No-KG:   {no_kg_output}")

    # Run format checker
    print("\n" + "=" * 60)
    print("Running format checker...")
    print("=" * 60)
    os.system(f"python format_checker.py --input_file {input_file} --prediction_file {kg_output} --mode rag_taskc")


if __name__ == "__main__":
    main()
