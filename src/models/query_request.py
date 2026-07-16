"""
Query request model.
"""

from typing import Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    query: str
    session_id: str
    # If true, force answering from the most recently uploaded persisted document
    use_latest: Optional[bool] = False
    # Optional index (0-based) of the persisted document to use for answering.
    # If provided, takes precedence over free-text triggers but is ignored when use_latest is True.
    persisted_doc_index: Optional[int] = None