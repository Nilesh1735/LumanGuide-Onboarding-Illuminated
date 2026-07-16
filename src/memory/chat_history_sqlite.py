"""
Persistent chat history backed by a local SQLite database.

This module provides a drop-in replacement for the MongoDB-backed history
in ``src.memory.chat_history_mongo``. Messages are persisted to a single
local file (``lumanguide.db`` by default) using the Python standard
library ``sqlite3`` module, so no external database server is required
and data survives process restarts.

The public surface mirrors the MongoDB implementation:
  - ``SQLiteChatMessageHistory(session_id)`` implements LangChain's
    ``BaseChatMessageHistory`` with async ``add_message``, ``get_messages``
    and ``clear`` methods, matching the calling convention used in
    ``src.api.routes`` (e.g. ``asyncio.create_task(history.add_message(...))``).
  - ``ChatHistory.get_session_history(session_id)`` returns a history
    instance for a session, so swapping the import is sufficient.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict

logger = logging.getLogger(__name__)

# Resolve the database path relative to the repository root so the file is
# stable regardless of the current working directory at process launch.
_DEFAULT_DB_NAME = "lumanguide.db"
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = str(_REPO_ROOT / _DEFAULT_DB_NAME)
DB_PATH = os.getenv("LUMANGUIDE_SQLITE_PATH", DEFAULT_DB_PATH)

# SQLite objects must not be shared across threads. Each call opens a short
#-lived connection scoped to the operation, guarded by a module-level lock to
# serialise writes and avoid "database is locked" contention under load.
_WRITE_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id            TEXT    PRIMARY KEY,
    session_id    TEXT    NOT NULL,
    message_type  TEXT    NOT NULL,
    content       TEXT    NOT NULL,
    additional_kwargs TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON chat_messages (session_id, created_at);
"""


def _row_to_message_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a persisted row into LangChain's ``messages_from_dict`` shape.

    Args:
        row: A ``sqlite3.Row`` with columns id, session_id, message_type,
            content, additional_kwargs and created_at.

    Returns:
        A dictionary in the ``{"type": ..., "data": {...}}`` form expected
        by ``langchain_core.messages.messages_from_dict``.
    """
    try:
        additional_kwargs = json.loads(row["additional_kwargs"] or "{}")
    except (ValueError, TypeError):
        additional_kwargs = {}
    return {
        "type": row["message_type"],
        "data": {
            "content": row["content"],
            "additional_kwargs": additional_kwargs,
        },
    }


class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """Async, SQLite-backed implementation of ``BaseChatMessageHistory``.

    The async interface is preserved to remain a drop-in replacement for
    the MongoDB implementation, which is awaited from FastAPI routes via
    ``asyncio.create_task``. All blocking SQLite work is dispatched to a
    background thread through ``asyncio.to_thread`` so the event loop is
    never blocked.
    """

    def __init__(
        self,
        session_id: str,
        db_path: str = DB_PATH,
    ) -> None:
        """Initialise the history for a session.

        Args:
            session_id: Unique identifier for the conversation session.
            db_path: Filesystem path to the SQLite database file. Defaults
                to the module-level ``DB_PATH`` (``lumanguide.db`` in the
                repository root, overridable via
                ``LUMANGUIDE_SQLITE_PATH``).
        """
        self.session_id = session_id
        self.db_path = db_path
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Internal helpers (synchronous, run in worker threads)
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a short-lived SQLite connection.

        Returns:
            A ``sqlite3.Connection`` configured for row access by name and
            with foreign-key pragmas enabled.
        """
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None,  # autocommit; we manage txns explicitly
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute("PRAGMA foreign_keys=ON;")
        return connection

    def _ensure_schema(self) -> None:
        """Create the schema and supporting index if they do not yet exist.

        This is safe to call repeatedly; ``CREATE ... IF NOT EXISTS`` is a
        no-op once the objects exist.
        """
        try:
            with _WRITE_LOCK:
                connection = self._connect()
                try:
                    connection.executescript(_SCHEMA)
                finally:
                    connection.close()
        except sqlite3.Error as exc:
            logger.exception("Failed to initialise SQLite schema at %s: %s",
                             self.db_path, exc)
            raise

    # ------------------------------------------------------------------
    # BaseChatMessageHistory implementation (async)
    # ------------------------------------------------------------------

    async def add_message(self, message: BaseMessage) -> None:
        """Persist a single message for this session.

        Args:
            message: A LangChain ``BaseMessage`` (e.g. ``HumanMessage``,
                ``AIMessage``) to store.
        """
        try:
            additional_kwargs = json.dumps(
                message.additional_kwargs or {}, default=str
            )
        except (TypeError, ValueError):
            additional_kwargs = "{}"

        await asyncio.to_thread(self._add_message_sync, message, additional_kwargs)

    def _add_message_sync(
        self,
        message: BaseMessage,
        additional_kwargs_json: str,
    ) -> None:
        """Synchronous insertion worker used by ``add_message``.

        Args:
            message: The message being persisted.
            additional_kwargs_json: Pre-serialised ``additional_kwargs``.
        """
        message_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        sql = (
            "INSERT INTO chat_messages "
            "(id, session_id, message_type, content, additional_kwargs, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?);"
        )
        try:
            with _WRITE_LOCK:
                connection = self._connect()
                try:
                    connection.execute(
                        sql,
                        (
                            message_id,
                            self.session_id,
                            message.type,
                            _coerce_text(message.content),
                            additional_kwargs_json,
                            created_at,
                        ),
                    )
                finally:
                    connection.close()
        except sqlite3.Error as exc:
            logger.exception(
                "Failed to persist message for session=%s: %s", self.session_id, exc
            )

    async def get_messages(self) -> List[BaseMessage]:
        """Load all messages for this session in chronological order.

        Returns:
            A list of ``BaseMessage`` instances ordered by insertion time.
            Returns an empty list if the session has no history or if a
            storage error occurs.
        """
        rows = await asyncio.to_thread(self._get_messages_sync)
        if not rows:
            return []
        return messages_from_dict([_row_to_message_dict(row) for row in rows])

    def _get_messages_sync(self) -> List[sqlite3.Row]:
        """Synchronous read worker used by ``get_messages``.

        Returns:
            A list of ``sqlite3.Row`` objects ordered by ``created_at``.
        """
        sql = (
            "SELECT id, session_id, message_type, content, additional_kwargs, "
            "created_at FROM chat_messages WHERE session_id = ? "
            "ORDER BY created_at ASC;"
        )
        try:
            # WAL mode permits concurrent readers, so reads do not need the
            # write lock. We open a fresh connection per read for safety.
            connection = self._connect()
            try:
                cursor = connection.execute(sql, (self.session_id,))
                return cursor.fetchall()
            finally:
                connection.close()
        except sqlite3.Error as exc:
            logger.exception(
                "Failed to load messages for session=%s: %s", self.session_id, exc
            )
            return []

    async def clear(self) -> None:
        """Delete every message recorded for this session."""
        await asyncio.to_thread(self._clear_sync)

    def _clear_sync(self) -> None:
        """Synchronous delete worker used by ``clear``."""
        sql = "DELETE FROM chat_messages WHERE session_id = ?;"
        try:
            with _WRITE_LOCK:
                connection = self._connect()
                try:
                    connection.execute(sql, (self.session_id,))
                finally:
                    connection.close()
        except sqlite3.Error as exc:
            logger.exception(
                "Failed to clear messages for session=%s: %s", self.session_id, exc
            )


def _coerce_text(content: Any) -> str:
    """Normalise message content to a plain string for storage.

    LangChain message ``content`` may be a string or a list of content
    blocks (e.g. ``[{"type": "text", "text": "..."}]``). This helper
    flattens either form to a single string so it can be stored in a TEXT
    column and faithfully reconstructed on read.

    Args:
        content: The raw ``content`` attribute of a message.

    Returns:
        A string representation of the content.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                parts.append(json.dumps(block, default=str))
        return "".join(parts)
    if content is None:
        return ""
    return json.dumps(content, default=str)


class ChatHistory:
    """Factory that mirrors ``src.memory.chat_history_mongo.ChatHistory``.

    Exposing the same ``get_session_history`` classmethod means a caller
    can switch backends by changing only the import statement.
    """

    @classmethod
    def get_session_history(
        cls,
        session_id: str,
        config: Optional[dict] = None,
    ) -> SQLiteChatMessageHistory:
        """Return the persistent history instance for a session.

        Args:
            session_id: Unique identifier for the conversation session.
            config: Optional configuration dictionary. Recognised keys:
                ``db_path`` (override the SQLite file path). Unused keys
                are ignored for forward compatibility.

        Returns:
            A ``SQLiteChatMessageHistory`` bound to the session.
        """
        db_path = DB_PATH
        if config and isinstance(config.get("db_path"), str):
            db_path = config["db_path"]
        return SQLiteChatMessageHistory(session_id=session_id, db_path=db_path)


__all__ = [
    "SQLiteChatMessageHistory",
    "ChatHistory",
    "DB_PATH",
    "DEFAULT_DB_PATH",
]
