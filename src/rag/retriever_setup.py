import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Ensure environment variables are loaded immediately
load_dotenv(override=True)

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = "data/faiss_index"

def _initialize_embeddings():
    """
    Initialize embeddings using Google Gemini.
    We use Gemini as the primary cloud provider to save RAM on Render's free tier.
    """
    gemini_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not gemini_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. The FAISS vector store requires Google "
            "Gemini embeddings. Please add it to your environment variables."
        )
    
    logger.info("Initializing Google Gemini Embeddings...")
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=gemini_key,
    )

# Initialize the global embeddings instance
embeddings = _initialize_embeddings()

def _load_vectorstore():
    """Load the FAISS vectorstore object if it exists."""
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            return FAISS.load_local(
                FAISS_INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return None
    return None

def get_all_documents():
    """Extract all Document objects from the FAISS docstore."""
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