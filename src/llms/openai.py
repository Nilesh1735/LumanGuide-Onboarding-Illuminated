import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI

logger = logging.getLogger(__name__)

def get_master_llm():
    """
    Initializes the best available Cloud LLM with 3-Tier Fallback logic.
    1. Mistral AI (Primary - with retries to handle LangGraph rate limits)
    2. OpenAI (Secondary - Fast, structured output)
    3. Google Gemini (Tertiary - Hardcoded model to save quota)
    """
    # 1. Try Mistral AI First
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if mistral_key:
        try:
            logger.info("Initializing Mistral AI LLM (Tier 1).")
            return ChatMistralAI(
                model="mistral-large-latest", 
                api_key=mistral_key, 
                temperature=0.7,
                max_retries=3  # <--- THIS FIXES THE 429 RATE LIMIT CRASH
            )
        except Exception as e:
            logger.error(f"Mistral LLM initialization failed: {e}")

    # 2. Try OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key != "sk-your_openai_key_here":
        logger.info("Initializing OpenAI LLM (Tier 2).")
        return ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.7, 
            openai_api_key=openai_key, 
            max_retries=0, 
            request_timeout=30
        )
    
    # 3. Fallback to Google Gemini
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if gemini_key and gemini_key != "AIzaSy_your_gemini_key_here":
        logger.info("Initializing Google Gemini LLM (gemini-1.5-pro-latest).")
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest", 
            google_api_key=gemini_key, 
            temperature=0.7,
            max_retries=2
        )

    raise ValueError("No Cloud LLM API keys (Mistral, OpenAI, Gemini) configured properly. Please check your .env file.")

# Export the master LLM instance to be used by LangGraph
llm = get_master_llm()