import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI

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
                    max_retries=0,
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
    Initializes the best available Cloud LLM with 3-Tier Fallback logic.
    1. Mistral AI (Primary - Generous free tier, excellent reasoning)
    2. OpenAI (Secondary - Fast, structured output)
    3. Google Gemini (Tertiary - Dynamic model selection)
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
                max_retries=0
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
    
    # 3. Fallback to Google Gemini (Dynamic Discovery)
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if gemini_key and gemini_key != "AIzaSy_your_gemini_key_here":
        logger.info("Attempting to initialize Google Gemini (Tier 3)...")
        gemini_llm = get_gemini_llm(gemini_key)
        if gemini_llm:
            return gemini_llm

    raise ValueError("No Cloud LLM API keys (Mistral, OpenAI, Gemini) configured properly. Please check your .env file.")

# Export the master LLM instance to be used by LangGraph
llm = get_master_llm()