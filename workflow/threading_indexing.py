"""Streaming multi-threaded indexing for RAG documents.

Implements a producer-consumer architecture with:
- Extract worker pool for parallel KG extraction and embedding
- Memory queue with spill-to-disk backpressure (90% spill, 60% restore)
- Single upsert worker with two-tier deduplication
- Checkpoint-based resume capability

Performance target: 600k docs in ~2 hours (vs ~500 hours sequential).
"""

import os
import threading
from typing import Any


# ============================================================================
# Thread-Safe Counter
# ============================================================================

class ThreadSafeCounter:
    """Thread-safe counter for progress tracking.

    Supports multiple threads incrementing and updating total
    without race conditions.
    """

    def __init__(self, total: int = 0):
        """Initialize counter with optional total.

        Args:
            total: Initial total count
        """
        self._value = 0
        self._total = total
        self._lock = threading.Lock()

    def increment(self, n: int = 1) -> int:
        """Increment counter by n and return new value.

        Args:
            n: Amount to increment by

        Returns:
            New value after increment
        """
        with self._lock:
            self._value += n
            return self._value

    def set_total(self, total: int) -> None:
        """Set the total count.

        Args:
            total: New total count
        """
        with self._lock:
            self._total = total

    def get(self) -> tuple[int, int]:
        """Get current (value, total).

        Returns:
            Tuple of (current_value, total)
        """
        with self._lock:
            return self._value, self._total


# ============================================================================
# Batch Size Calculator
# ============================================================================

# Constants
SYSTEM_PROMPT_TOKENS = 2000
CHARS_PER_TOKEN = 4  # Rough estimate for English text
TOKENS_PER_DOC_MARKER = 50  # Overhead for document markers
MAX_BATCH_SIZE_CAP = 1000


def calculate_optimal_batch_size(
    avg_doc_chars: int | None = None,
    context_window: int | None = None,
    safety_margin: float | None = None,
) -> int:
    """Calculate optimal LLM batch size based on model context window.

    Optimizes for maximum throughput while staying within model's
    context limit. Uses conservative safety margin to account for
    token estimation errors.

    Args:
        avg_doc_chars: Average document character count (from env: AVG_DOC_CHARS)
        context_window: Model context window in tokens (from env: MODEL_TOKEN_LIMIT)
        safety_margin: Percentage of context to use (from env: SAFETY_MARGIN)

    Returns:
        Optimal batch size (capped at MAX_BATCH_SIZE_CAP)
    """
    if avg_doc_chars is None:
        avg_doc_chars = int(os.getenv("AVG_DOC_CHARS", "500"))
    if context_window is None:
        context_window = int(os.getenv("MODEL_TOKEN_LIMIT", "128000"))
    if safety_margin is None:
        safety_margin = float(os.getenv("SAFETY_MARGIN", "0.8"))

    # Estimate tokens per document
    tokens_per_doc = avg_doc_chars / CHARS_PER_TOKEN + TOKENS_PER_DOC_MARKER

    # Calculate available tokens
    available_tokens = (context_window - SYSTEM_PROMPT_TOKENS) * safety_margin

    # Calculate batch size
    batch_size = int(available_tokens / tokens_per_doc)

    # Cap at maximum and ensure at least 1
    return max(1, min(batch_size, MAX_BATCH_SIZE_CAP))
