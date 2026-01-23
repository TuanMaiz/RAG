from pathlib import Path
from typing import Iterator

from langchain_core.documents import Document


class JSONLLoader:
    """Load documents from JSONL files."""

    def __init__(self, file_path: str | Path, text_key: str = "text", metadata_keys: list[str] | None = None):
        """
        Initialize the JSONL loader.

        Args:
            file_path: Path to the JSONL file
            text_key: Key to use as the document text content
            metadata_keys: Keys to include in metadata (default: all except text_key)
        """
        self.file_path = Path(file_path)
        self.text_key = text_key
        self.metadata_keys = metadata_keys

    def load(self) -> list[Document]:
        """Load all documents from the JSONL file."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load documents from the JSONL file."""
        import json

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get(self.text_key, "")
                    if not text:
                        continue

                    # Build metadata
                    if self.metadata_keys:
                        metadata = {k: data.get(k) for k in self.metadata_keys if k in data}
                    else:
                        metadata = {k: v for k, v in data.items() if k != self.text_key}

                    # Add source info
                    metadata["source"] = str(self.file_path)
                    metadata["line_number"] = line_num

                    yield Document(page_content=text, metadata=metadata)
                except json.JSONDecodeError:
                    continue


def load_dataset(dataset_name: str, dataset_dir: str | Path = "dataset") -> list[Document]:
    """
    Load a dataset by name from the dataset directory.

    Args:
        dataset_name: Name of the dataset folder (e.g., "clapnq.jsonl", "cloud.jsonl")
        dataset_dir: Base directory containing datasets

    Returns:
        List of Document objects
    """
    dataset_path = Path(dataset_dir) / dataset_name / f"{dataset_name}"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    loader = JSONLLoader(dataset_path)
    return loader.load()