"""
ReAct agent setup using LangGraph's prebuilt agent compiler.
"""

import os
from langgraph.prebuilt import create_react_agent
from langchain_core.tools.retriever import create_retriever_tool
from src.config.settings import Config
from src.llms.openai import llm  
from src.rag.retriever_setup import get_retriever

config = Config()

# 1. Fetch the raw retriever
raw_retriever = get_retriever()

# 2. Convert it into an authorized tool with schema definitions for the agent
retriever_tool = create_retriever_tool(
    retriever=raw_retriever,
    name="dense_index_search",
    description="Search and retrieve relevant context from the baseline vector database."
)

# 3. Pass the tool list to the agent compiler
tools = [retriever_tool]

try:
    agent_executor = create_react_agent(
        llm,
        tools,
        prompt=config.prompt("system_prompt")
    )
except Exception as exc:
    # Graceful fallback when the configured model/agent setup is incompatible
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("create_react_agent failed; using fallback agent executor: %s", exc)

    class _FallbackAgent:
        def invoke(self, input: dict):
            # Return minimal structure expected by retriever_node
            return {"output": "Fallback agent: tool execution disabled.", "intermediate_steps": []}

    agent_executor = _FallbackAgent()
