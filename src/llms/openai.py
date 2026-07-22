import os
import logging
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

def get_master_llm():
    """
    Initializes the DeepSeek LLM for LangGraph generation.
    """
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        raise ValueError("DEEPSEEK_API_KEY is not configured in the environment.")
    
    logger.info("Initializing DeepSeek LLM.")
    return ChatOpenAI(
        model="deepseek-chat", 
        api_key=deepseek_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.7, 
        max_retries=3, 
        request_timeout=30
    )

# Export the master LLM instance to be used by LangGraph
llm = get_master_llm()