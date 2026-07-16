import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Ensure environment variables are loaded immediately
load_dotenv(override=True)

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = "data/faiss_index"

def _initialize_embeddings():
    """
    3-Tier Embedding Fallback:
    1. Try Mistral AI (Fast, Cloud, Generous Free Tier)
    2. Try OpenAI (Fast, Cloud, Industry Standard)
    3. Fallback to HuggingFace (Fast, Local CPU - 100% reliable)
    """
    # 1. Try Mistral AI
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if mistral_key:
        try:
            logger.info("Initializing Mistral AI Embeddings (Tier 1)...")
            return MistralAIEmbeddings(model="mistral-embed", api_key=mistral_key)
        except Exception as e:
            logger.warning(f"Mistral Embeddings failed to initialize: {e}.")
    
    # 2. Try OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key != "sk-your_openai_key_here":
        try:
            logger.info("Initializing OpenAI Embeddings (Tier 2)...")
            return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_key)
        except Exception as e:
            logger.warning(f"OpenAI Embeddings failed to initialize: {e}.")
            
    # 3. Fallback to HuggingFace (Runs locally on CPU)
    logger.info("Falling back to HuggingFace Embeddings (Tier 3 - Local CPU)...")
    try:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        logger.error(f"Fatal: HuggingFace Embeddings failed to initialize: {e}")
        raise RuntimeError("Could not initialize any embedding model. Check your API keys and internet connection.")

# Initialize the global embeddings instance
embeddings = _initialize_embeddings()

def _load_vectorstore():
    """Load the FAISS vectorstore object if it exists."""
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return None
    return None

def get_all_documents():
    """Extract all Document objects from the FAISS docstore (Fixes iteration error)."""
    vectorstore = _load_vectorstore()
    if vectorstore:
        try:
            return list(vectorstore.docstore._dict.values())
        except Exception as e:
            logger.error(f"Failed to extract documents from FAISS docstore: {e}")
            return []
    return []

def add_documents_to_retriever(documents, tenant_id="default_tenant"):
    """Add documents to the FAISS index and save."""
    if not documents:
        return
    try:
        vectorstore = _load_vectorstore()
        if vectorstore:
            vectorstore.add_documents(documents)
        else:
            vectorstore = FAISS.from_documents(documents, embeddings)
        
        vectorstore.save_local(FAISS_INDEX_PATH)
        logger.info(f"Successfully added {len(documents)} documents to FAISS index.")
    except Exception as e:
        logger.error(f"Error adding documents to FAISS: {e}")
        raise e

def get_retriever():
    """Return the retriever interface for the FAISS index."""
    vectorstore = _load_vectorstore()
    if vectorstore:
        return vectorstore.as_retriever()
    return None