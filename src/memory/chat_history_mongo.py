"""
Chat history storage using MongoDB backend.
"""

from datetime import datetime
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

from src.db.mongo_client import db

# In-process fallback store used when MongoDB is unavailable.
_IN_MEMORY_STORE: dict[str, list[dict]] = {}

collection = db["chat_history"]


class MongoDBChatMessageHistory(BaseChatMessageHistory):
    """Chat history backed by MongoDB, with an in-memory fallback if Mongo is
    unreachable. This keeps the rest of the app functional during development
    when a local Mongo instance isn't running.
    """

    def __init__(self, session_id: str):
        """
        Initialize chat history for a session.

        Args:
            session_id: Unique session identifier.
        """
        self.session_id = session_id

    async def add_message(self, message: BaseMessage) -> None:
        """
        Save a message to MongoDB. If the DB operation fails (e.g. no Mongo
        server), fall back to storing the message in an in-memory dict.

        Args:
            message: The message to save.
        """
        doc = {
            "session_id": self.session_id,
            "type": message.type,
            "content": message.content,
            "additional_kwargs": message.additional_kwargs,
            "timestamp": datetime.utcnow(),
        }
        try:
            await collection.insert_one(doc)
        except Exception:
            # Fallback to in-memory store (best-effort, not persistent)
            _IN_MEMORY_STORE.setdefault(self.session_id, []).append(doc)

    async def get_messages(self) -> List[BaseMessage]:
        """
        Load all messages for a session from MongoDB, falling back to the in-
        memory store if necessary.

        Returns:
            List of messages in chronological order.
        """
        from langchain_core.messages import messages_from_dict

        try:
            cursor = collection.find({"session_id": self.session_id}).sort("timestamp", 1)
            docs = await cursor.to_list(length=1000)
        except Exception:
            docs = _IN_MEMORY_STORE.get(self.session_id, [])

        # Convert to BaseMessage objects
        return messages_from_dict([
            {
                "type": d["type"],
                "data": {
                    "content": d["content"],
                    "additional_kwargs": d.get("additional_kwargs", {}),
                }
            }
            for d in docs
        ])

    async def clear(self) -> None:
        """Delete all messages for a session, from DB or in-memory fallback."""
        try:
            await collection.delete_many({"session_id": self.session_id})
        except Exception:
            _IN_MEMORY_STORE.pop(self.session_id, None)


class ChatHistory:
    """Factory for MongoDB-backed chat history."""

    @classmethod
    def get_session_history(
        cls,
        session_id: str,
        config: dict = None
    ) -> MongoDBChatMessageHistory:
        """
        Get or create chat history for a session.

        Args:
            session_id: Unique session identifier.
            config: Optional configuration dictionary.

        Returns:
            MongoDBChatMessageHistory instance for the session.
        """
        return MongoDBChatMessageHistory(session_id)
