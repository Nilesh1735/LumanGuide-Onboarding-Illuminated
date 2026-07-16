import streamlit as st
import os
from utils.theme import get_custom_css

st.set_page_config(page_title="Admin Telemetry | LumanGuide", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-header">
        <h1>Admin <span>Telemetry</span></h1>
        <p class="app-subtitle">LLMOps Observability Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
### LangSmith Integration
This application integrates with **LangSmith** to provide full observability into the LangGraph execution pipeline. 

Every LLM call, tool execution, and retrieval step is traced and logged in real-time to monitor:
- Token usage and API costs
- Latency and execution time per node
- Agent thought processes and tool inputs
- Retrieval relevance grading
""")

# Check if LangSmith is configured
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
project_name = os.getenv("LANGCHAIN_PROJECT", "lumanguide")

if not langsmith_api_key:
    st.warning("⚠️ LangSmith is not configured. Please add `LANGCHAIN_API_KEY` and `LANGCHAIN_TRACING_V2=true` to your `.env` file to enable tracing.")
else:
    st.success(f"✅ LangSmith is active! Tracing project: `{project_name}`")
    
    # Construct the LangSmith URL
    langsmith_url = f"https://smith.langchain.com/projects/p/{project_name}"
    
    st.markdown(f"#### Live Trace Dashboard")
    st.markdown(f"View live traces of your RAG pipeline [here]({langsmith_url}).")
    
    # Embed the LangSmith dashboard
    st.components.v1.iframe(langsmith_url, height=800, scrolling=True)