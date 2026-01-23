from langchain.chat_models import init_chat_model

from vector_stores.qdrant import client, COLLECTION_NAME
from workflow.generation import query
from workflow.indexing import load_doc, store_doc


# ANSI colors
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


def chat():
    """Chat with the RAG model."""
    model = init_chat_model("gpt-4o-mini")

    print(f"\n{Colors.YELLOW}--- Chat mode (type 'quit' to exit) ---{Colors.RESET}")
    while True:
        user_input = input(f"\n{Colors.CYAN}You:{Colors.RESET} ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        if not user_input:
            continue

        print(f"{Colors.GREEN}Assistant:{Colors.RESET} ", end="", flush=True)
        response = query(user_input, model)
        print(response)


def process_documents():
    """Process and store documents."""
    # Check if collection exists and has documents
    if client.collection_exists(COLLECTION_NAME):
        collection_info = client.get_collection(COLLECTION_NAME)
        doc_count = collection_info.points_count

        if doc_count and doc_count > 0:
            confirm = input(f"Vector store has {doc_count} documents. Re-index? (y/N): ").strip().lower()
            if confirm != "y":
                print("Skipping indexing.")
                return

    # Index and store documents
    print("Indexing documents...")
    docs = load_doc()
    print(f"Total documents indexed: {len(docs)}")

    # Store in vector store
    print("\nStoring documents in vector store...")
    store_doc(docs)


def main():
    while True:
        print(f"\n{Colors.BLUE}=== RAG System ==={Colors.RESET}")
        print("1. Chat with model")
        print("2. Process documents")
        print("3. Exit")

        choice = input(f"\n{Colors.YELLOW}Select an option:{Colors.RESET} ").strip()

        if choice == "1":
            chat()
        elif choice == "2":
            process_documents()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()
