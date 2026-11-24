"""
memory/conversation_memory.py — Sliding-Window Conversation Memory

Stores conversation history per session and provides the last N turns
as a list of OpenAI-style messages for the agent's context window.
"""
from __future__ import annotations
from collections import defaultdict, deque
from config import MAX_HISTORY


class ConversationMemory:
    """
    In-memory conversation store with per-session sliding windows.

    Each session keeps up to MAX_HISTORY recent exchanges.
    An "exchange" = 1 user message + 1 assistant message = 2 messages.
    """

    def __init__(self, max_turns: int = MAX_HISTORY):
        # Each session: deque of (user_msg, assistant_msg) tuples
        self._sessions: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_turns)
        )
        self.max_turns = max_turns

    def add_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Add a completed exchange to session history."""
        self._sessions[session_id].append((user_message, assistant_message))

    def get_history(self, session_id: str) -> list[dict]:
        """
        Return conversation history as OpenAI-style messages list.

        Returns:
            [
                {"role": "user",      "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        """
        messages = []
        for user_msg, asst_msg in self._sessions[session_id]:
            messages.append({"role": "user",      "content": user_msg})
            messages.append({"role": "assistant",  "content": asst_msg})
        return messages

    def clear_session(self, session_id: str) -> None:
        """Clear all history for a session."""
        if session_id in self._sessions:
            self._sessions[session_id].clear()

    def get_session_count(self, session_id: str) -> int:
        """Return number of exchanges stored for a session."""
        return len(self._sessions.get(session_id, []))

    def list_sessions(self) -> list[str]:
        """Return all known session IDs."""
        return list(self._sessions.keys())
