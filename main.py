import argparse
from pathlib import Path


# Deferred imports for faster startup
def _get_llm():
    from model import get_llm
    return get_llm


def _get_query():
    from workflow.generation import query
    return query


def _get_load_doc():
    from workflow.indexing import load_doc
    return load_doc


def _get_store_doc():
    from workflow.indexing import store_doc
    return store_doc


def _get_streaming_indexer():
    from workflow.threading_indexing import StreamingIndexer
    return StreamingIndexer


def _get_conversation_memory():
    from workflow.memory import ConversationMemory
    return ConversationMemory


def _get_qdrant():
    from vector_stores.qdrant import client, COLLECTION_NAME
    return client, COLLECTION_NAME


# ANSI colors
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RED = "\033[91m"
    RESET = "\033[0m"


# Dataset configurations
DATASETS = {
    "1": ("clapnq", "dataset/clapnq.jsonl"),
    "2": ("cloud", "dataset/cloud.jsonl"),
    "3": ("fiqa", "dataset/fiqa.jsonl"),
    "4": ("govt", "dataset/govt.jsonl"),
    "5": ("all", "dataset"),
}

DEFAULT_DATASET_DIR = "dataset"


def choose_indexer_mode():
    """Prompt user to choose indexer mode.

    Returns:
        1 for current indexer, 2 for streaming indexer
    """
    print(f"\n{Colors.CYAN}=== Indexing Mode Selection ==={Colors.RESET}")
    print(f"  {Colors.GREEN}1.{Colors.RESET} Current (threaded batches) - Proven, stable")
    print(f"  {Colors.GREEN}2.{Colors.RESET} Streaming (experimental) - Faster, optimized batch size")

    while True:
        choice = input(f"\n{Colors.YELLOW}Choose mode [1-2]:{Colors.RESET} ").strip()
        if choice in ("1", "2"):
            return int(choice)
        print(f"{Colors.RED}Invalid choice, please enter 1 or 2{Colors.RESET}")


def scan_datasets(dataset_dir: str = DEFAULT_DATASET_DIR) -> list[tuple[str, str]]:
    """Scan dataset directory for .jsonl files.

    Returns:
        List of (name, path) tuples for found datasets
    """
    base_path = Path(dataset_dir)
    if not base_path.exists():
        return []

    datasets = []
    for jsonl_file in base_path.glob("**/*.jsonl"):
        if jsonl_file.is_file():
            # Extract dataset name from filename (e.g., "clapnq" from "clapnq.jsonl")
            name = jsonl_file.stem.lower()
            if name.endswith(".jsonl"):
                name = name[:-6]  # Remove .jsonl extension if present in stem
            datasets.append((name, str(jsonl_file)))

    # Sort by name and remove duplicates
    datasets = sorted(set(datasets))
    return datasets


def chat():
    """Chat with the RAG model."""
    import os

    get_llm = _get_llm()
    ConversationMemory = _get_conversation_memory()
    query = _get_query()

    # Use generation-specific model, or fallback to default
    generation_model_name = os.getenv("OPENAI_GENERATION_MODEL") or os.getenv("OPENAI_LLM_MODEL")
    model = get_llm(generation_model_name)
    memory = ConversationMemory(window_size=5)

    # Import rich UI components
    from utils.rich_ui import (
        print_welcome_message,
        get_user_input,
        print_user_message,
        print_assistant_message,
        print_goodbye_message,
    )

    print_welcome_message()
    while True:
        user_input = get_user_input()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        if not user_input:
            continue

        print_user_message(user_input)
        response = query(user_input, model, memory)
        print_assistant_message(response)

    print_goodbye_message()


def process_documents(dataset_dir: str = DEFAULT_DATASET_DIR, dataset_name: str | None = None):
    """Process and store documents.

    Args:
        dataset_dir: Directory containing datasets
        dataset_name: Specific dataset to index, or None for all
    """
    client, COLLECTION_NAME = _get_qdrant()
    load_doc = _get_load_doc()

    # Check if collection exists and has documents
    if client.collection_exists(COLLECTION_NAME):
        collection_info = client.get_collection(COLLECTION_NAME)
        doc_count = collection_info.points_count

        if doc_count and doc_count > 0:
            print(f"{Colors.YELLOW}Vector store has {doc_count} documents.{Colors.RESET}")
            confirm = input("Re-index? (y/N): ").strip().lower()
            if confirm != "y":
                print("Skipping indexing.")
                return

    # Index and store documents
    print(f"\n{Colors.BLUE}Loading documents from:{Colors.RESET} {dataset_dir}")
    if dataset_name:
        print(f"{Colors.BLUE}Dataset:{Colors.RESET} {dataset_name}")

    docs = load_doc(dataset_dir, dataset_name)
    if not docs:
        print(f"{Colors.RED}No documents found to index.{Colors.RESET}")
        return

    print(f"{Colors.GREEN}Total documents to index: {len(docs)}{Colors.RESET}")

    # Choose indexer mode
    indexer_mode = choose_indexer_mode()

    if indexer_mode == 2:
        # Use streaming indexer
        StreamingIndexer = _get_streaming_indexer()

        print(f"\n{Colors.BLUE}Using Streaming Indexer (optimized batch size)...{Colors.RESET}")
        indexer = StreamingIndexer()
        results = indexer.index(docs, show_progress=True)

        print(f"\n{Colors.GREEN}Indexing complete!{Colors.RESET}")
        print(f"  {Colors.CYAN}Extracted:{Colors.RESET} {results['extracted_docs']} documents")
        print(f"  {Colors.CYAN}Upserted:{Colors.RESET} {results['upserted_docs']} documents")
        print(f"  {Colors.CYAN}Time:{Colors.RESET} {results['elapsed_seconds']:.1f} seconds")
        print(f"  {Colors.CYAN}Speed:{Colors.RESET} {results['docs_per_minute']:.1f} docs/min")

        if results['exceptions']:
            print(f"  {Colors.YELLOW}Warnings:{Colors.RESET} {len(results['exceptions'])} exceptions occurred")
    else:
        # Use existing store_doc
        store_doc = _get_store_doc()
        print(f"\n{Colors.BLUE}Storing documents in vector store...{Colors.RESET}")
        store_doc(docs)
        print(f"{Colors.GREEN}Indexing complete!{Colors.RESET}")


def select_dataset_interactive(dataset_dir: str = DEFAULT_DATASET_DIR) -> str | None:
    """Interactive dataset selection menu.

    Args:
        dataset_dir: Directory to scan for datasets

    Returns:
        Selected dataset name, or None for "all"
    """
    datasets = scan_datasets(dataset_dir)

    if not datasets:
        print(f"{Colors.RED}No .jsonl files found in {dataset_dir}{Colors.RESET}")
        return None

    print(f"\n{Colors.BLUE}Available datasets in {dataset_dir}:{Colors.RESET}")
    for i, (name, _) in enumerate(datasets, 1):
        print(f"  {i}. {name}")
    print(f"  0. All datasets")

    choice = input(f"\n{Colors.YELLOW}Select dataset to index [0-{len(datasets)}]:{Colors.RESET} ").strip()

    if choice == "0":
        return None
    elif choice.isdigit() and 1 <= int(choice) <= len(datasets):
        return datasets[int(choice) - 1][0]
    else:
        print(f"{Colors.RED}Invalid choice.{Colors.RESET}")
        return None


def main():
    parser = argparse.ArgumentParser(description="RAG System - Multi-Turn Conversational Retrieval")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR,
                       help=f"Directory containing datasets (default: {DEFAULT_DATASET_DIR})")
    parser.add_argument("--dataset", default=None,
                       help="Specific dataset to index (e.g., clapnq, cloud). If not specified, prompts interactively.")
    parser.add_argument("--chat", action="store_true",
                       help="Skip menu and go directly to chat mode")
    parser.add_argument("--index", action="store_true",
                       help="Skip menu and go directly to indexing mode")

    args = parser.parse_args()

    # If --index specified but no --dataset, show interactive selection
    if args.index and args.dataset is None:
        args.dataset = select_dataset_interactive(args.dataset_dir)
        if args.dataset is None and not scan_datasets(args.dataset_dir):
            return  # No datasets found

    # Direct mode (CLI flags)
    if args.chat or args.index:
        if args.chat:
            chat()
        elif args.index:
            process_documents(args.dataset_dir, args.dataset)
        return

    # Interactive menu mode
    while True:
        print(f"\n{Colors.BLUE}=== RAG System ==={Colors.RESET}")
        print("1. Chat with model")
        print("2. Index documents")
        print("3. Exit")

        choice = input(f"\n{Colors.YELLOW}Select an option:{Colors.RESET} ").strip()

        if choice == "1":
            chat()
        elif choice == "2":
            # Prompt for directory first
            dir_input = input(f"{Colors.YELLOW}Enter dataset directory, default: [{DEFAULT_DATASET_DIR}]:{Colors.RESET} ").strip()
            dataset_dir = dir_input if dir_input else DEFAULT_DATASET_DIR

            # Then select dataset from that directory
            dataset_name = select_dataset_interactive(dataset_dir)
            process_documents(dataset_dir, dataset_name)
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print(f"{Colors.RED}Invalid option. Try again.{Colors.RESET}")


if __name__ == "__main__":
    main()
