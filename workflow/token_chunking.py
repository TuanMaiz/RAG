"""Token-aware chunking with domain grouping.

Implements the chunking strategy from documents/kg_pipeline_plan.md:
- Classify documents by token count (short < 150, normal 150-512, long > 512)
- Group by domain (title or id prefix)
- Buffer short documents until MIN_TOKENS reached
- Split long documents with overlap
"""

import os
import uuid
from typing import Literal

# Token counting configuration
MIN_TOKENS = int(os.getenv("KG_MIN_CHUNK_TOKENS", "150"))
MAX_TOKENS = int(os.getenv("KG_MAX_CHUNK_TOKENS", "512"))
OVERLAP_TOKENS = int(os.getenv("KG_OVERLAP_TOKENS", "50"))

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    _tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        """Count tokens using tiktoken (GPT-4 tokenizer)."""
        return len(_tokenizer.encode(text))
except ImportError:
    # Fallback: approximate by word count * 1.3
    def count_tokens(text: str) -> int:
        """Approximate token count (word count * 1.3)."""
        return int(len(text.split()) * 1.3)


def classify_document(line: dict) -> Literal["short", "normal", "long"]:
    """Classify a document by token count.

    Args:
        line: Document dict with at least "text" key

    Returns:
        "short" if < MIN_TOKENS, "long" if > MAX_TOKENS, "normal" otherwise
    """
    tokens = count_tokens(line.get("text", ""))
    if tokens < MIN_TOKENS:
        return "short"
    elif tokens > MAX_TOKENS:
        return "long"
    return "normal"


def get_domain(line: dict) -> str:
    """Extract domain from a document line.

    Args:
        line: Document dict with "title" or "id" key

    Returns:
        Domain string (title if present, otherwise id prefix)
    """
    if "title" in line and line["title"]:
        return line["title"]
    # Fallback to id prefix
    doc_id = line.get("id", "unknown")
    return doc_id.split("-")[0] if "-" in doc_id else doc_id


def group_by_domain(lines: list[dict]) -> dict[str, list[dict]]:
    """Group documents by domain.

    Args:
        lines: List of document dicts

    Returns:
        Dict mapping domain to list of documents
    """
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for line in lines:
        domain = get_domain(line)
        groups[domain].append(line)
    return dict(groups)


def make_chunk(lines: list[dict], domain: str) -> dict:
    """Create a chunk from a list of short documents.

    Args:
        lines: List of document dicts to merge
        domain: Domain name for this chunk

    Returns:
        Chunk dict with chunk_id, text, domain, title, source_ids
    """
    parts = []
    source_ids = []

    for line in lines:
        if "title" in line and line["title"]:
            parts.append(f"[{line['title']}] {line['text']}")
        else:
            parts.append(line["text"])
        source_ids.append(line.get("id", ""))

    chunk_id = f"chunk-{uuid.uuid4().hex[:8]}"

    return {
        "chunk_id": chunk_id,
        "text": "\n".join(parts),
        "domain": domain,
        "title": lines[0].get("title", domain) if lines else domain,
        "source_ids": source_ids,
    }


def split_long_line(line: dict, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[dict]:
    """Split a long document into chunks with overlap.

    Args:
        line: Document dict to split
        max_tokens: Maximum tokens per chunk
        overlap: Token overlap between chunks

    Returns:
        List of chunk dicts
    """
    text = line["text"]
    words = text.split()
    chunks = []

    # Approximate words per token (cl100k_base: ~0.75 words/token)
    words_per_chunk = int(max_tokens / 0.75)
    overlap_words = int(overlap / 0.75)

    start = 0
    chunk_num = 0

    while start < len(words):
        end = start + words_per_chunk
        chunk_text = " ".join(words[start:end])

        # Prepend title if available
        if "title" in line and line["title"]:
            chunk_text = f"[{line['title']}] {chunk_text}"

        chunks.append({
            "chunk_id": f"chunk-{uuid.uuid4().hex[:8]}",
            "text": chunk_text,
            "domain": get_domain(line),
            "title": line.get("title", ""),
            "source_ids": [line.get("id", "")],
        })

        start = end - overlap_words
        chunk_num += 1

    return chunks


def chunk_domain_group(lines: list[dict], domain: str) -> list[dict]:
    """Chunk a domain group using buffer-aware strategy.

    Args:
        lines: List of documents in the domain (in order)
        domain: Domain name

    Returns:
        List of chunk dicts
    """
    chunks = []
    buffer = []
    buffer_tokens = 0

    for line in lines:
        kind = classify_document(line)
        line_tokens = count_tokens(line.get("text", ""))

        if kind == "short":
            # Add to buffer
            buffer.append(line)
            buffer_tokens += line_tokens

            # Flush if buffer reached minimum
            if buffer_tokens >= MIN_TOKENS:
                chunks.append(make_chunk(buffer, domain))
                buffer, buffer_tokens = [], 0

        elif kind == "normal":
            # Flush buffer first, then create chunk for this document
            if buffer:
                chunks.append(make_chunk(buffer, domain))
                buffer, buffer_tokens = [], 0
            chunks.append(make_chunk([line], domain))

        elif kind == "long":
            # Flush buffer first, then split the long document
            if buffer:
                chunks.append(make_chunk(buffer, domain))
                buffer, buffer_tokens = [], 0
            chunks.extend(split_long_line(line))

    # Flush remaining buffer
    if buffer:
        chunks.append(make_chunk(buffer, domain))

    return chunks


def preprocess_jsonl_to_chunks(docs: list[dict]) -> list[dict]:
    """Convert loaded JSONL documents to chunks using token-aware strategy.

    Args:
        docs: List of document dicts from JSONL loader.
              Each should have at least: {"text": "...", "id": "..."}
              Optional: "title" field

    Returns:
        List of chunk dicts with: chunk_id, text, domain, title, source_ids
    """
    # Group by domain
    domain_groups = group_by_domain(docs)

    all_chunks = []
    for domain, lines in domain_groups.items():
        chunks = chunk_domain_group(lines, domain)
        all_chunks.extend(chunks)

    return all_chunks


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test with sample data
    sample_docs = [
        {"id": "doc-001", "title": "French Revolution", "text": "A short text about Napoleon."},
        {"id": "doc-002", "title": "French Revolution", "text": "Another short text."},
        {
            "id": "doc-003",
            "title": "French Revolution",
            "text": " ".join(["word"] * 200)  # Long text
        },
    ]

    print("Testing token-aware chunking...")
    print(f"MIN_TOKENS={MIN_TOKENS}, MAX_TOKENS={MAX_TOKENS}, OVERLAP={OVERLAP_TOKENS}")

    # Test classification
    for doc in sample_docs:
        kind = classify_document(doc)
        tokens = count_tokens(doc["text"])
        print(f"  {doc['id']}: {tokens} tokens → {kind}")

    # Test chunking
    chunks = preprocess_jsonl_to_chunks(sample_docs)
    print(f"\nGenerated {len(chunks)} chunks:")
    for chunk in chunks:
        chunk_tokens = count_tokens(chunk["text"])
        print(f"  {chunk['chunk_id']}: {chunk_tokens} tokens, domain={chunk['domain']}")
        print(f"    source_ids: {chunk['source_ids']}")
