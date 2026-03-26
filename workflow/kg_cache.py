"""LLM response caching for knowledge graph extraction.

Provides disk-backed caching to avoid re-calling the LLM for identical chunks.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class ExtractionCache:
    """Cache for LLM extraction results.

    Stores responses on disk to avoid re-calling the LLM for identical chunks.
    The cache key is a hash of the input text and prompt template.
    """

    def __init__(self, cache_path: str = ".kg_extraction_cache.json"):
        """Initialize the extraction cache.

        Args:
            cache_path: Path to the cache file on disk
        """
        self.cache_path = Path(cache_path)
        self._cache: dict[str, dict[str, Any]] = {}
        self._dirty = False

        # Load existing cache if it exists
        self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r") as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                # Cache file corrupted, start fresh
                self._cache = {}

    def _save(self) -> None:
        """Save cache to disk."""
        if self._dirty:
            try:
                with open(self.cache_path, "w") as f:
                    json.dump(self._cache, f, indent=2)
                self._dirty = False
            except IOError:
                # Fail silently - caching is optional
                pass

    def make_key(self, text: str, prompt_template: str) -> str:
        """Create a cache key from text and prompt template.

        Args:
            text: The input text
            prompt_template: The prompt template used

        Returns:
            A hash key for caching
        """
        # Hash both text and template to ensure consistency
        combined = f"{prompt_template}|||{text[:500]}"  # First 500 chars sufficient
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def get(self, cache_key: str) -> dict[str, Any] | None:
        """Get a cached result.

        Args:
            cache_key: The cache key to look up

        Returns:
            The cached result dict, or None if not found
        """
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            # Return a copy to avoid mutation
            return entry.get("result")
        return None

    def set(self, cache_key: str, result: dict[str, Any], timestamp: float | None = None) -> None:
        """Store a result in the cache.

        Args:
            cache_key: The cache key to store under
            result: The result dict to cache (entities, relationships)
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()

        self._cache[cache_key] = {
            "result": result,
            "timestamp": timestamp,
        }
        self._dirty = True

        # Auto-save every 100 writes
        if len(self._cache) % 100 == 0:
            self._save()

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._dirty = True

    def save(self) -> None:
        """Force save cache to disk."""
        self._save()

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    def __contains__(self, cache_key: str) -> bool:
        """Check if a key is in the cache."""
        return cache_key in self._cache

    def __del__(self):
        """Save cache on destruction."""
        self._save()
