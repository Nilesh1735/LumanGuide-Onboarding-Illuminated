import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

def get_master_llm():
    """
    Initializes the Google Gemini LLM.
    """
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        raise ValueError("GOOGLE_API_KEY is not configured in the environment.")
    
    logger.info("Initializing Google Gemini LLM (gemini-1.5-flash-002).")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-002",  # <--- CHANGED TO PINNED STABLE VERSION
        google_api_key=gemini_key, 
        temperature=0.7,
        max_retries=0
    )

# Export the master LLM instance to be used by LangGraph
llm = get_master_llm()