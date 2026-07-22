"""
State-Driven Adaptive RAG State Machine.
Integrates OpenAI intent classification, vector lookup, and self-healing error loops.
Features enterprise-grade resilience (3-Tier LLM Fallback) and structured logging.
"""

import os
import re
import logging
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import START, END
from langgraph.graph.state import StateGraph
from langchain_openai import ChatOpenAI  # ADDED FOR BYOK

from src.rag.retriever_setup import get_retriever, get_all_documents
from src.rag.guardrail_node import (
    guardrail_node,
    guardrail_router,
    _BLOCKED_FLAG,
)
from src.config.settings import Config
from src.llms.openai import llm
from src.models.grade import Grade
from src.models.route_identifier import RouteIdentifier
from src.models.state import WorkflowExecutionState
from src.tools.graph_tools import routing_tool, doc_tool, rewrite_router

logger = logging.getLogger(__name__)
config = Config()

# ADDED: BYOK Helper Function
def get_active_llm(state: WorkflowExecutionState):
    """Returns the user's OpenAI LLM if a key is provided in the state, otherwise falls back to the global LLM."""
    user_key = state.get("user_openai_key")
    if user_key:
        return ChatOpenAI(model="gpt-4o-mini", openai_api_key=user_key, temperature=0.7, max_retries=0)
    return llm

def evaluate_query_intent(state: WorkflowExecutionState):
    question = state["messages"][-1].content
    tenant_id = state.get("tenant_id", "default_tenant")

    logger.info(f"--- EVALUATING INTENT FOR TENANT: {tenant_id} via LLM ---")

    _team_intent_pattern = re.compile(
        r"\b(who (knows|is|owns|works|should|can)|who'?s the (sme|expert|owner)|"
        r"subject matter expert|smes?|codeowners?|team (member|navigator)|"
        r"who to ask|who do i ask|who should i (talk|ask|contact)|"
        r"point of contact|poc for|expert (in|on|for))\b",
        re.IGNORECASE,
    )
    if _team_intent_pattern.search(question):
        logger.info("Router dispatched query to: 'team_nav' (deterministic pre-check match)")
        return {
            "messages": state["messages"],
            "route": "team_nav",
            "latest_query": question,
            "consecutive_errors": state.get("consecutive_errors", 0),
        }

    try:
        retriever = get_retriever()
        docs = retriever.invoke(question) if retriever else []
        context = "\n\n".join([f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs])
    except Exception as e:
        logger.error(f"Retriever failed during intent evaluation: {e}")
        context = "No context available."

    classify_prompt = PromptTemplate(
        template=config.prompt("classify_prompt"),
        input_variables=["question", "context"]
    )
    
    try:
        active_llm = get_active_llm(state) # BYOK INTEGRATION
        # FIX: Use standard invoke to avoid DeepSeek 400 error with strict response_format
        chain = classify_prompt | active_llm
        raw_result = chain.invoke({"question": question, "context": context})
        
        # Parse the text output to find the route
        decision_text = raw_result.content.strip().lower()
        if "team" in decision_text or "nav" in decision_text:
            route = "team_nav"
        elif "index" in decision_text:
            route = "index"
        elif "search" in decision_text:
            route = "search"
        else:
            route = "general"

        logger.info(f"Router dispatched query to: '{route}'")
        return {
            "messages": state["messages"],
            "route": route,
            "latest_query": question,
            "consecutive_errors": state.get("consecutive_errors", 0)
        }
    except Exception as e:
        logger.error(f"LLM classification failed: {e}. Defaulting to 'general'.")
        return {
            "messages": state["messages"],
            "route": "general",
            "latest_query": question,
            "consecutive_errors": state.get("consecutive_errors", 0)
        }

def general_llm(state: WorkflowExecutionState):
    logger.info("--- COMPUTING DIRECT GENERATION NODE ---")
    try:
        active_llm = get_active_llm(state) # BYOK INTEGRATION
        result = active_llm.invoke(state["messages"])
        return {"messages": result}
    except Exception as e:
        logger.error(f"General LLM invocation failed: {e}")
        return {"messages": [AIMessage(content="I'm having trouble connecting to the AI engine right now. Please try again.")]}

def retriever_node(state: WorkflowExecutionState):
    query = state["latest_query"]
    logger.info(f"--- QUERYING DENSE INDEX ---")

    try:
        retriever = get_retriever()
        if not retriever:
            return {"messages": [AIMessage(content="No documents found in index.")]}

        docs = retriever.invoke(query)
        context_str = "\n\n".join([f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs])
        
        new_message = AIMessage(content=context_str if context_str else "No relevant documents found.")
        return {"messages": [new_message]}
        
    except Exception as e:
        logger.error(f"Retrieval Node execution failed: {e}")
        consecutive = state.get("consecutive_errors", 0) + 1
        return {
            "messages": [AIMessage(content=f"An execution issue occurred: {str(e)}")],
            "consecutive_errors": consecutive
        }

def grade(state: WorkflowExecutionState):
    logger.info("--- EVALUATING CONTENT RELEVANCE METRICS ---")
    grading_prompt = PromptTemplate(
        template=config.prompt("grading_prompt"),
        input_variables=["question", "context"]
    )
    context = state["messages"][-1].content
    question = state["latest_query"]

    try:
        active_llm = get_active_llm(state) # BYOK INTEGRATION
        # FIX: Use standard invoke to avoid DeepSeek 400 error with strict response_format
        chain_graded = grading_prompt | active_llm
        raw_result = chain_graded.invoke({"question": question, "context": context})
        
        # Parse the text output to find yes/no
        result_text = raw_result.content.strip().lower()
        if "yes" in result_text:
            score = "yes"
        else:
            score = "no"
    except Exception as e:
        logger.warning(f"Grading LLM failed: {e}. Defaulting to 'yes'.")
        score = "yes"

    logger.info(f"Content relevance status: '{score.upper()}'")
    return {"messages": state["messages"], "binary_score": score}

def rewrite_query(state: WorkflowExecutionState):
    logger.info("--- REWRITING QUERY TO RESOLVE SEMANTIC GAP ---")
    query = state["latest_query"]
    rewrite_prompt = PromptTemplate(
        template=config.prompt("rewrite_prompt"),
        input_variables=["query"]
    )
    
    try:
        active_llm = get_active_llm(state) # BYOK INTEGRATION
        chain = rewrite_prompt | active_llm
        result = chain.invoke({"query": query})
        return {"latest_query": result.content}
    except Exception as e:
        logger.error(f"Rewrite LLM failed: {e}. Using original query.")
        return {"latest_query": query}

def team_navigator(state: WorkflowExecutionState):
    """Fixes the bug where it always returns the same person by strictly using retriever metadata."""
    logger.info("--- EXECUTING TEAM NAVIGATOR NODE ---")
    query = state["latest_query"]
    
    try:
        retriever = get_retriever()
        if not retriever:
            return {"messages": [AIMessage(content="Team Navigator data not loaded.")]}
        
        docs = retriever.invoke(query)
        team_docs = [d for d in docs if d.metadata.get("doc_type") == "team_profile"]
        
        if not team_docs:
            team_docs = docs

        context_str = "\n\n".join([f"[Source: {d.metadata.get('source', 'team_config.yaml')}]\n{d.page_content}" for d in team_docs])
        
        if not context_str:
            return {"messages": [AIMessage(content="No team members found matching this query.")]}
        
        return {"messages": [AIMessage(content=context_str)]}
        
    except Exception as e:
        logger.error(f"Team Navigator failed: {e}")
        return {"messages": [AIMessage(content=f"Team Navigator error: {str(e)}")]}

def generate(state: WorkflowExecutionState):
    logger.info("--- SYNTHESIZING FINAL RESPONSE ---")
    context = state["messages"][-1].content
    question = state["latest_query"]

    route = state.get("route", "")
    if route == "team_nav":
        prompt_text = """
        You are the Contextual Team Navigator. Your job is to answer questions about team members based STRICTLY on the provided context.
        DO NOT use any prior knowledge. DO NOT default to the same person for every answer.
        Read the context carefully and identify the exact person who matches the user's query.
        
        Context:
        {context}
        
        User Question: {question}
        
        Based strictly on the context above, answer the User Question. 
        You must append the source of the information at the end of your response.
        Format: [Source: filename]
        """
        logger.info("--- USING HARDCODED NAVIGATOR PROMPT ---")
    else:
        prompt_text = """
        You are an expert AI assistant. Answer the user's question based strictly on the provided context.
        
        Context:
        {context}
        
        User Question: {question}
        
        Based on the context above, answer the User Question.
        You must append the source of the information at the end of your response.
        Format: [Source: filename]
        """

    generate_prompt = PromptTemplate(
        template=prompt_text,
        input_variables=["context", "question"]
    )

    try:
        active_llm = get_active_llm(state) # BYOK INTEGRATION
        generate_chain = generate_prompt | active_llm
        result = generate_chain.invoke({"context": context, "question": question})
        return {"messages": [{"role": "assistant", "content": result.content}]}
    except Exception as e:
        logger.error(f"Final generation LLM failed: {e}")
        return {"messages": [AIMessage(content="I found the relevant context but am currently unable to synthesize a response due to an AI engine issue.")]}

def web_search(state: WorkflowExecutionState):
    logger.info("--- EXECUTING EXTERNAL WEB SEARCH FOR RECENT EVENTS ---")
    
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.error("Web search failed: TAVILY_API_KEY is not set in the environment.")
        return {"messages": [AIMessage(content="Web search failed: TAVILY_API_KEY is not set in the environment.")]}
        
    try:
        search_tool = TavilySearchResults(max_results=3, api_key=tavily_api_key)
        result = search_tool.invoke(state["latest_query"])

        contents = [f"[Source: {item.get('url', 'web')}]\n{item['content']}" for item in result if "content" in item]
        return {
            "messages": [{"role": "assistant", "content": "\n\n".join(contents)}]
        }
    except Exception as e:
        logger.error(f"Tavily web search failed: {e}")
        return {"messages": [AIMessage(content=f"Web search encountered an error: {str(e)}")]}

def verify_retry_limit(state: WorkflowExecutionState) -> str:
    if state.get("consecutive_errors", 0) >= 3:
        logger.warning("Infinite Loop Safeguard Activated. Halting execution pipeline immediately.")
        return "halt_execution"
    return "proceed"

# Construct state machine nodes
graph = StateGraph(WorkflowExecutionState)

graph.add_node("guardrail", guardrail_node)
graph.add_node("evaluate_query_intent", evaluate_query_intent)
graph.add_node("retriever", retriever_node)
graph.add_node("grade", grade)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite_query)
graph.add_node("web_search", web_search)
graph.add_node("general_llm", general_llm)
graph.add_node("team_navigator", team_navigator)

graph.add_edge(START, "guardrail")
graph.add_conditional_edges(
    "guardrail",
    guardrail_router,
    {"block": END, "proceed": "evaluate_query_intent"},
)
graph.add_edge("web_search", "generate")
graph.add_edge("retriever", "grade")
graph.add_edge("generate", END)
graph.add_edge("general_llm", END)
graph.add_edge("team_navigator", "grade")

graph.add_conditional_edges(
    "rewrite",
    rewrite_router,
    {"retriever": "retriever", "team_navigator": "team_navigator"}
)

graph.add_conditional_edges(
    "retriever",
    verify_retry_limit,
    {"halt_execution": END, "proceed": "grade"}
)
graph.add_conditional_edges("evaluate_query_intent", routing_tool)
graph.add_conditional_edges("grade", doc_tool)

builder = graph.compile()