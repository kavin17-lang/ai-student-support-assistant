"""
memory_store.py
----------------
Implements the "Memory" capability of the AI Student Support Assistant.

Two kinds of memory are provided:
1. Short-term conversational memory: the last N turns of the chat, used to
   give the agent context about what was just discussed (so it can handle
   follow-up questions like "what about the fee for that?").
2. Long-term session memory: simple key-value facts remembered about the
   student for the duration of the session (e.g. their name, department,
   or the last topic they asked about), stored to a local JSON file so it
   survives across app restarts for a given user id.
"""

import json
import os
from collections import deque
from datetime import datetime


class ConversationMemory:
    """Keeps the last `max_turns` (user, assistant) exchanges in memory."""

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.turns = deque(maxlen=max_turns)

    def add_turn(self, user_msg: str, assistant_msg: str):
        self.turns.append(
            {
                "user": user_msg,
                "assistant": assistant_msg,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def get_recent_context(self) -> str:
        """Return the recent conversation formatted as plain text context."""
        if not self.turns:
            return ""
        lines = []
        for t in self.turns:
            lines.append(f"Student: {t['user']}")
            lines.append(f"Assistant: {t['assistant']}")
        return "\n".join(lines)

    def last_user_message(self) -> str:
        if not self.turns:
            return ""
        return self.turns[-1]["user"]

    def clear(self):
        self.turns.clear()


class LongTermMemory:
    """Persists simple facts about a student across sessions using a JSON file."""

    def __init__(self, storage_path: str = "data/student_memory.json"):
        self.storage_path = storage_path
        self._data = {}
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {}
        else:
            self._data = {}

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def remember(self, student_id: str, key: str, value):
        self._data.setdefault(student_id, {})[key] = value
        self._save()

    def recall(self, student_id: str, key: str, default=None):
        return self._data.get(student_id, {}).get(key, default)

    def profile(self, student_id: str) -> dict:
        return self._data.get(student_id, {})
