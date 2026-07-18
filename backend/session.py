"""
Session memory — in-memory store for conversation history.
Session-scoped, last N turns, passed directly into the LLM context window.
No vector memory store, no summarization layer.
"""

from .models import Session, Turn
from .config import SESSION_MAX_TURNS
import logging

logger = logging.getLogger(__name__)


class SessionStore:
    """Thread-safe in-memory session store."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
            logger.info(f"[session] created new session: {session_id}")
        return self._sessions[session_id]

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Add a turn and enforce the max-turn window."""
        session = self.get_or_create(session_id)
        session.turns.append(Turn(role=role, content=content))
        # Keep only the last SESSION_MAX_TURNS turns
        if len(session.turns) > SESSION_MAX_TURNS:
            session.turns = session.turns[-SESSION_MAX_TURNS:]

    def get_history_for_prompt(self, session_id: str) -> list[dict]:
        """Return turns formatted for LLM messages."""
        session = self.get_or_create(session_id)
        return [{"role": t.role, "content": t.content} for t in session.turns]

    def pin_document(self, session_id: str, content: str) -> None:
        """Pin an uploaded document's content to the session."""
        session = self.get_or_create(session_id)
        session.pinned_document = content
        logger.info(f"[session] pinned document ({len(content)} chars) to session {session_id}")

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        logger.info(f"[session] deleted session: {session_id}")

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())


# Singleton instance — imported by other modules
session_store = SessionStore()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Session Store:")
    store = SessionStore()
    store.add_turn("session-1", "user", "Hello")
    store.add_turn("session-1", "assistant", "Hi")
    history = store.get_history_for_prompt("session-1")
    print(f"History for session-1: {history}")
    print(f"Active sessions: {store.list_sessions()}")
