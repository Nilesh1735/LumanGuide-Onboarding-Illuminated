import importlib
import streamlit as st
import requests
import utils.api_client as api_client
from utils.theme import get_custom_css

api_client = importlib.reload(api_client)

st.set_page_config(page_title="LumanGuide | Welcome", layout="wide")

# Apply the unified reference palette
st.markdown(get_custom_css(page="home"), unsafe_allow_html=True)

def is_success(response):
    if isinstance(response, dict):
        return response.get("ok", False)
    return response is True

def error_message(response, fallback):
    if isinstance(response, dict):
        return response.get("detail") or response.get("error", fallback)
    if response is False:
        return "The API client returned False without details. Restart Streamlit so it reloads utils/api_client.py."
    return fallback

# Wrap the API call in a try-except block to prevent crashes when backend is down
try:
    if "api_token" not in st.session_state:
        st.session_state["api_token"] = api_client.get_api_token()
    st.session_state["api_initialized"] = True
except requests.exceptions.ConnectionError:
    st.session_state["api_initialized"] = False

if st.session_state.get("api_initialized"):
    if "jwt_token" not in st.session_state:
        
        # --- SPLIT SCREEN LAYOUT ---
        st.markdown('<div class="landing-grid">', unsafe_allow_html=True)
        
        # LEFT PANEL: Pitch & Telemetry
        st.markdown("""
        <div class="left-panel">
            <h1 class="brand-title">LumanGuide</h1>
            <p class="brand-subtitle">Onboarding, Illuminated.</p>
            
            <div class="capabilities">
                <div class="cap-item"><span class="cap-dot"></span> State-Driven Adaptive RAG</div>
                <div class="cap-item"><span class="cap-dot"></span> AppSec & JWT Secured Infrastructure</div>
                <div class="cap-item"><span class="cap-dot"></span> Multi-Cloud LLM Fallback Architecture</div>
                <div class="cap-item"><span class="cap-dot"></span> Built on LangGraph & FastAPI</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Live Telemetry Terminal
        try:
            status = api_client.get_team_status()
            nav_status = "Online" if status.get("navigator_loaded") else "Offline"
            sme_count = status.get("member_count", 0)
            
            st.markdown(f"""
            <div class="terminal-box">
                <div class="terminal-header">SYSTEM TELEMETRY</div>
                <div class="terminal-line">> System Status: <span class="ok-text">Nominal</span></div>
                <div class="terminal-line">> Vector DB: FAISS (<span class="ok-text">Online</span>)</div>
                <div class="terminal-line">> LLM Router: <span class="ok-text">Active</span></div>
                <div class="terminal-line">> Team Navigator: {sme_count} SMEs loaded</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.markdown("""
            <div class="terminal-box">
                <div class="terminal-header">SYSTEM TELEMETRY</div>
                <div class="terminal-line">> System Status: <span class="err-text">Offline</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True) # Close left-panel
        
        # RIGHT PANEL: Auth Box
        st.markdown('<div class="right-panel">', unsafe_allow_html=True)
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Log in", "Sign up"])

        with tab1:
            st.subheader("Log in to your workspace")
            login_username = st.text_input("Username or Email", key="login_user", placeholder="Enter your username or email")
            login_password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")

            st.write("") # Margin spacing
            if st.button("Log in", use_container_width=True, type="primary"):
                auth_response = api_client.login_user(login_username, login_password, st.session_state["api_token"])

                if is_success(auth_response) and isinstance(auth_response, dict) and auth_response.get("token"):
                    st.session_state["jwt_token"] = auth_response.get("token")
                    st.session_state["session_id"] = login_username
                    st.rerun()
                else:
                    st.error(error_message(auth_response, "Invalid credentials."))
                    if isinstance(auth_response, dict) and auth_response.get("attempts"):
                        with st.expander("Diagnostic details", expanded=True):
                            for line in auth_response.get("attempts", []):
                                st.write(line)

        with tab2:
            st.subheader("Create a new account")
            signup_username = st.text_input("Choose a username", key="signup_user", placeholder="Your username")
            signup_email = st.text_input("Email", key="signup_email", placeholder="your@email.com")
            signup_password = st.text_input("Choose a password", type="password", key="signup_pass", placeholder="Minimum 6 characters")

            st.write("") # Margin spacing
            if st.button("Sign up", use_container_width=True, type="primary"):
                if signup_username and signup_email and signup_password:
                    try:
                        result = api_client.create_user(signup_username, signup_email, signup_password, st.session_state["api_token"])

                        if is_success(result):
                            st.success("Account created successfully. Please switch to the Log in tab.")
                        else:
                            st.error(f"Account creation failed. {error_message(result, 'Unknown server error')}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during signup: {str(e)}")
                else:
                    st.warning("Please fill out all fields.")
                    
        st.markdown('</div>', unsafe_allow_html=True) # Close auth-container
        
        # Tech Stack Pills
        st.markdown("""
        <div class="tech-pills">
            <span class="tech-pill">Secured by JWT</span>
            <span class="tech-pill">LangGraph</span>
            <span class="tech-pill">FastAPI</span>
            <span class="tech-pill">FAISS</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True) # Close right-panel
        st.markdown('</div>', unsafe_allow_html=True) # Close landing-grid

    else:
        # --- LOGGED IN VIEW ---
        st.markdown("## LumanGuide Dashboard")
        with st.container(border=True):
            st.success(f"Successfully authenticated as **{st.session_state.get('session_id', 'User')}**")
            st.write("") # Margin spacing
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Go to Chat Workspace", use_container_width=True, type="primary"):
                    st.switch_page("pages/Chat.py")
            with col2:
                if st.button("Manage Documents", use_container_width=True):
                    st.switch_page("pages/Chat.py")
            with col3:
                if st.button("Log out session", use_container_width=True):
                    del st.session_state["jwt_token"]
                    st.session_state.pop("session_id", None)
                    st.success("Logged out successfully.")
                    st.rerun()
else:
    with st.container(border=True):
        st.warning("Application features are currently offline due to a backend connection timeout.")
        if st.button("Retry backend handshake"):
            st.rerun()