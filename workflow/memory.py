"""Conversation memory for multi-turn RAG."""


class ConversationMemory:
    """Stores conversation history for query rewriting."""

    def __init__(self, window_size: int = 5):
        """
        Initialize conversation memory.

        Args:
            window_size: Number of recent turns to use for query rewriting.
        """
        self.history: list[dict[str, str]] = []
        self.window_size = window_size

    def add_turn(self, query: str, response: str) -> None:
        """Add a conversation turn."""
        self.history.append({"query": query, "response": response})

    def get_context(self) -> str:
        """
        Get formatted conversation context for rewrite prompt.

        Returns the last N turns (window_size) formatted as:
        Q: ...
        A: ...
        """
        recent = self.history[-self.window_size:] if self.history else []
        if not recent:
            return ""

        lines = []
        for turn in recent:
            lines.append(f"Q: {turn['query']}")
            lines.append(f"A: {turn['response']}")
        return "\n".join(lines)

    def get_history(self) -> list[dict[str, str]]:
        """Get full history (for reference, not typically used for rewriting)."""
        return self.history.copy()

    def clear(self) -> None:
        """Clear conversation history."""
        self.history.clear()
