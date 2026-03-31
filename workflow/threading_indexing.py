"""Streaming multi-threaded indexing for RAG documents.

Implements a producer-consumer architecture with:
- Extract worker pool for parallel KG extraction and embedding
- Memory queue with spill-to-disk backpressure (90% spill, 60% restore)
- Single upsert worker with two-tier deduplication
- Checkpoint-based resume capability

Performance target: 600k docs in ~2 hours (vs ~500 hours sequential).
"""

import os
import shelve
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from utils.logging_config import get_logger

logger = get_logger(__name__)


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


# ============================================================================
# Memory Queue with Spill/Restore
# ============================================================================

# Constants
DEFAULT_MAX_MEMORY_GB = 4
DEFAULT_SPILL_THRESHOLD_PCT = 90
DEFAULT_RESTORE_THRESHOLD_PCT = 60
DEFAULT_SPILL_PATH = "./indexing_spill"

# Memory estimation constants (conservative)
BYTES_PER_POINT = 4 * 1536 + 200  # Dense vector + overhead
BYTES_PER_ENTITY = 500
BYTES_PER_RELATIONSHIP = 200
ENTITIES_PER_DOC = 50
RELS_PER_DOC = 10
BYTES_PER_DOC_OVERHEAD = 1000


def estimate_batch_memory(num_docs: int) -> int:
    """Estimate memory in bytes for a batch result.

    Args:
        num_docs: Number of documents in batch

    Returns:
        Estimated bytes
    """
    # Vectors
    vector_memory = num_docs * BYTES_PER_POINT

    # KG data
    kg_memory = num_docs * (ENTITIES_PER_DOC * BYTES_PER_ENTITY + RELS_PER_DOC * BYTES_PER_RELATIONSHIP)

    # Overhead
    overhead = num_docs * BYTES_PER_DOC_OVERHEAD

    return vector_memory + kg_memory + overhead


class MemoryQueue:
    """Thread-safe queue with automatic spill-to-disk backpressure.

    Monitors memory usage and automatically spills newest items to disk
    when threshold is reached. Restores items from disk when space
    becomes available.
    """

    def __init__(
        self,
        max_memory_gb: float | None = None,
        spill_threshold: float | None = None,
        restore_threshold: float | None = None,
        spill_path: str | None = None,
    ):
        """Initialize memory queue with spill/restore.

        Args:
            max_memory_gb: Maximum memory in GB before triggering spill
            spill_threshold: Percentage (0-1) to trigger spill (default: 0.9)
            restore_threshold: Percentage (0-1) to trigger restore (default: 0.6)
            spill_path: Directory for shelve files
        """
        self.max_memory_gb = max_memory_gb or float(os.getenv("INDEXING_MAX_MEMORY_GB", DEFAULT_MAX_MEMORY_GB))
        self.spill_threshold = spill_threshold or float(os.getenv("SPILL_THRESHOLD_PCT", DEFAULT_SPILL_THRESHOLD_PCT)) / 100
        self.restore_threshold = restore_threshold or float(os.getenv("RESTORE_THRESHOLD_PCT", DEFAULT_RESTORE_THRESHOLD_PCT)) / 100
        self.spill_path = Path(spill_path or os.getenv("SPILL_PATH", DEFAULT_SPILL_PATH))
        self.spill_path.mkdir(parents=True, exist_ok=True)

        # Internal deque for FIFO ordering with newest access
        self._deque = deque()
        self._lock = threading.Lock()

        # Shelve for spill storage
        self._shelve_path = self.spill_path / "queue_shelf"

    def put(self, item: dict[str, Any]) -> None:
        """Put item in queue.

        Args:
            item: Batch result dict
        """
        with self._lock:
            self._deque.append(item)

    def get(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Get oldest item from queue.

        Args:
            timeout: Max seconds to wait (None = wait forever)

        Returns:
            Batch result dict, or None if timeout
        """
        start_time = time.time()

        while True:
            with self._lock:
                if self._deque:
                    return self._deque.popleft()

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return None

            # Wait a bit before checking again
            time.sleep(0.01)

    def pop_newest(self) -> dict[str, Any] | None:
        """Get newest item from queue (for spilling).

        Returns:
            Batch result dict, or None if empty
        """
        with self._lock:
            if self._deque:
                return self._deque.pop()
        return None

    def size(self) -> int:
        """Get number of items in queue (in memory only).

        Returns:
            Number of items
        """
        with self._lock:
            return len(self._deque)

    def estimate_size_bytes(self) -> int:
        """Estimate current memory usage in bytes.

        Returns:
            Estimated bytes
        """
        with self._lock:
            total = 0
            for item in self._deque:
                num_docs = len(item.get("points", []))
                total += estimate_batch_memory(num_docs)
            return total

    def estimate_size_gb(self) -> float:
        """Estimate current memory usage in GB.

        Returns:
            Estimated GB
        """
        return self.estimate_size_bytes() / (1024**3)

    def _spill_to_disk(self) -> int:
        """Spill newest items to shelve until below threshold.

        Returns:
            Number of items spilled
        """
        num_spilled = 0

        with shelve.open(self._shelve_path) as shelf:
            while self.estimate_size_gb() >= self.max_memory_gb * self.spill_threshold:
                item = self.pop_newest()
                if item is None:
                    break
                shelf[str(item["batch_id"])] = item
                num_spilled += 1
                logger.debug("Spilled batch %s to disk", item.get("batch_id"))

        if num_spilled > 0:
            logger.info("Spilled %d batches to disk (queue at %.2f GB/%.2f GB)",
                       num_spilled, self.estimate_size_gb(), self.max_memory_gb)

        return num_spilled

    def _restore_from_disk(self) -> int:
        """Restore oldest items from shelve until above threshold.

        Returns:
            Number of items restored
        """
        num_restored = 0

        with shelve.open(self._shelve_path, writeback=True) as shelf:
            # Get sorted keys (oldest first based on batch_id)
            keys = sorted(shelf.keys(), key=lambda k: int(k) if k.lstrip('-').isdigit() else 0)

            while (self.estimate_size_gb() <= self.max_memory_gb * self.restore_threshold
                   and keys):
                key = keys.pop(0)  # Get oldest key
                item = shelf[key]
                self.put(item)
                del shelf[key]
                num_restored += 1
                logger.debug("Restored batch %s from disk", item.get("batch_id"))

        if num_restored > 0:
            logger.info("Restored %d batches from disk (queue at %.2f GB/%.2f GB)",
                       num_restored, self.estimate_size_gb(), self.max_memory_gb)

        return num_restored


# ============================================================================
# Retry Logic
# ============================================================================

MAX_RETRY_ATTEMPTS = 3


def retry_with_backoff(
    operation,
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    delay: float = 1.0,
    operation_name: str = "operation",
) -> Any:
    """Execute operation with exponential backoff retry.

    Args:
        operation: Callable to execute
        max_attempts: Maximum retry attempts
        delay: Initial delay in seconds (doubles each retry)
        operation_name: Name for logging

    Returns:
        Operation result, or None if all attempts fail
    """
    last_error = None

    for attempt in range(max_attempts):
        try:
            return operation()
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                wait_time = delay * (2 ** attempt)
                logger.warning(
                    "%s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    operation_name, attempt + 1, max_attempts, e, wait_time
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    "%s failed after %d attempts: %s",
                    operation_name, max_attempts, e
                )

    return None
