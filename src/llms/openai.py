import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

def get_gemini_llm(api_key: str):
    """
    Dynamically queries the Google API to find an available FLASH model.
    Uses the raw Google SDK for the smoke test to bypass LangChain's internal retry loop.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        available_models = genai.list_models()
        
        flash_models = []
        for m in available_models:
            if 'generateContent' in m.supported_generation_methods:
                model_name = m.name.split('/')[-1]
                if 'pro' in model_name.lower() or 'vision' in model_name.lower():
                    continue
                if 'flash' in model_name.lower():
                    flash_models.append(model_name)
        
        for model_name in flash_models:
            try:
                logger.info(f"Testing Gemini model (raw SDK): {model_name}")
                test_model = genai.GenerativeModel(model_name)
                test_model.generate_content("Reply with OK")
                
                logger.info(f"Dynamically selected and verified Gemini model: {model_name}")
                return ChatGoogleGenerativeAI(
                    model=model_name, 
                    google_api_key=api_key, 
                    max_retries=2,  # Allow retries for rate limits
                    transport="rest"
                )
            except Exception as test_e:
                logger.warning(f"Model {model_name} failed smoke test: {str(test_e)[:100]}")
                continue
                
        logger.warning("Gemini API key valid, but no FLASH models passed the smoke test.")
        return None
            
    except Exception as e:
        logger.error(f"Failed to discover Gemini models dynamically: {e}")
        return None

def get_master_llm():
    """
    Initializes Google Gemini dynamically.
    """
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if gemini_key and gemini_key != "AIzaSy_your_gemini_key_here":
        logger.info("Attempting to initialize Google Gemini...")
        gemini_llm = get_gemini_llm(gemini_key)
        if gemini_llm:
            return gemini_llm

    raise ValueError("GOOGLE_API_KEY is not configured properly. Please check your .env file.")

# Export the master LLM instance to be used by LangGraph
llm = get_master_llm()