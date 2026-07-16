import logging
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

def team_navigator(state):
    """
    Team Navigator node. Uses the FAISS retriever to find specific team members.
    """
    print("--- TEAM NAVIGATOR NODE ---")
    question = state["latest_query"]
    
    try:
        # Import the correct function from the updated retriever_setup
        from src.rag.retriever_setup import get_retriever
        
        retriever = get_retriever()
        if not retriever:
            return {"messages": [AIMessage(content="Team Navigator data not loaded.")]}
        
        # Fetch documents specifically related to team profiles
        docs = retriever.invoke(question)
        
        # Format context with source metadata to force LLM to read it
        context_str = "\n\n".join([
            f"[Source: {d.metadata.get('source', 'team_config.yaml')}]\n{d.page_content}" 
            for d in docs if d.metadata.get("doc_type") == "team_profile"
        ])
        
        if not context_str:
            # Fallback to all docs if specific metadata filter misses
            context_str = "\n\n".join([
                f"[Source: {d.metadata.get('source', 'team_config.yaml')}]\n{d.page_content}" 
                for d in docs
            ])
            
        if not context_str:
            return {"messages": [AIMessage(content="No team members found matching this query.")]}
        
        # Pass strict context to generate node
        return {"messages": [AIMessage(content=context_str)]}
        
    except Exception as e:
        logger.error(f"Team Navigator node failed: {e}")
        return {"messages": [AIMessage(content=f"Team Navigator error: {str(e)}")]}