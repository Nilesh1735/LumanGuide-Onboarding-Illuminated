from typing import TypedDict, Annotated, Optional, List
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

class WorkflowExecutionState(TypedDict):
    """State schema for the LangGraph orchestrator."""
    messages: Annotated[list[AnyMessage], add_messages]
    binary_score: Optional[str]
    route: Optional[str]
    latest_query: Optional[str]
    tenant_id: Optional[str]
    consecutive_errors: int
    doc_type_filter: Optional[str]
    user_openai_key: Optional[str]  # ADDED FOR BYOK INTEGRATION

# Alias to satisfy the import in graph_tools.py
State = WorkflowExecutionState