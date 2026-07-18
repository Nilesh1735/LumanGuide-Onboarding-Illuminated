import os
import logging
from langchain_mistralai import ChatMistralAI

logger = logging.getLogger(__name__)

def get_master_llm():
    """
    Initializes the Mistral LLM.
    """
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        raise ValueError("MISTRAL_API_KEY is not configured in the environment.")
    
    logger.info("Initializing Mistral AI LLM.")
    return ChatMistralAI(
        model="mistral-large-latest", 
        api_key=mistral_key, 
        temperature=0.7,
        max_retries=0
    )

# Export the master LLM instance to be used by LangGraph
llm = get_master_llm()