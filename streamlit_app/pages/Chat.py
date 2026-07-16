import importlib
import streamlit as st
import utils.api_client as api_client
from utils.theme import get_custom_css
from streamlit_feedback import streamlit_feedback
import re

api_client = importlib.reload(api_client)

st.set_page_config(page_title="LumanGuide Workspace", layout="wide", initial_sidebar_state="expanded")
st.markdown(get_custom_css(), unsafe_allow_html=True)

if "show_logout_confirm" not in st.session_state:
    st.session_state.show_logout_confirm = False

# --- Top Header Area ---
col_title, col_logout = st.columns([9, 1])
with col_title:
    st.markdown("## LumanGuide Workspace")
with col_logout:
    if st.button("Log out", use_container_width=True):
        st.session_state.show_logout_confirm = True

if st.session_state.show_logout_confirm:
    with st.container(border=True):
        st.warning("Are you sure you want to log out?")
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("Yes, log out", type="primary", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.switch_page("Home.py")
        with col_cancel:
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_logout_confirm = False
                st.rerun()

if "session_id" not in st.session_state or "jwt_token" not in st.session_state:
    st.warning("Please log in first.")
    st.stop()

# --- RBAC LOGIC: Check if the logged-in user is the Admin ---
# You can change "admin" to whatever username you want to use.
is_admin = st.session_state.get("session_id") == "admin"

# --- Sidebar Area ---
with st.sidebar:
    st.markdown("## Navigation")
    
    # Inner Container 1: Data Ingestion
    with st.container(border=True):
        st.markdown("#### Data Ingestion")
        uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
        file_description = None
        
        if uploaded_file:
            file_description = st.text_input("Describe your document (required)", max_chars=300, placeholder="E.g. Engineering runbooks")
            if "uploaded_files" not in st.session_state:
                st.session_state.uploaded_files = {}

            file_key = f"{uploaded_file.name}_{file_description}"

            if file_description:
                if file_key not in st.session_state.uploaded_files:
                    upload_result = api_client.document_upload_rag(uploaded_file, file_description, st.session_state["jwt_token"])
                    upload_ok = upload_result.get("ok", False) if isinstance(upload_result, dict) else upload_result is True
                    if upload_ok:
                        st.success(f"Uploaded: {uploaded_file.name}")
                        st.session_state.uploaded_files[file_key] = True
                    else:
                        detail = upload_result.get("error", "Unknown upload error") if isinstance(upload_result, dict) else "Unknown upload error"
                        st.error(f"Document upload failed: {uploaded_file.name}. {detail}")
            else:
                st.warning("Please describe your document before uploading.")

    st.divider()

    # Inner Container 2: Context Controls
    with st.container(border=True):
        st.markdown("#### Context Controls")
        use_latest = st.checkbox("Force query on latest upload", value=False)

    st.divider()

    # Inner Container 3: Team Navigator Map
    with st.container(border=True):
        st.markdown("#### Team Navigator Map")
        try:
            nav_status = api_client.get_team_status()
        except Exception:
            nav_status = None

        if isinstance(nav_status, dict) and nav_status.get("navigator_loaded"):
            st.success(f"Team data active: {nav_status.get('member_count', 0)} SME(s)")
            members = nav_status.get("members", [])
            if members:
                with st.expander(f"View {len(members)} team member(s)", expanded=False):
                    for m in members:
                        st.markdown(f"- {m}")
        else:
            st.info("Team map uninitialized.")

    st.divider()

    # Inner Container 4: Persisted Docs
    with st.container(border=True):
        st.markdown("#### Document Picker")
        persisted_resp = api_client.get_persisted_docs()
        persisted_docs = persisted_resp.get("documents", []) if isinstance(persisted_resp, dict) and persisted_resp.get("ok") else []

        doc_choices = [f"{d['index']}: {d['snippet'][:80].replace(chr(10),' ')}" for d in persisted_docs]
        selected_doc_index = None
        if doc_choices:
            sel = st.selectbox("Select document", ["(none - use default search)"] + doc_choices)
            if sel and sel != "(none - use default search)":
                try:
                    selected_doc_index = int(sel.split(":", 1)[0])
                except Exception:
                    selected_doc_index = None
        else:
            st.caption("No persisted documents found.")

    st.divider()

    # Inner Container 5: BYOK (Bring Your Own Key) - Admin vs User Logic
    with st.container(border=True):
        st.markdown("#### Engine Configuration")
        if is_admin:
            # Admin gets a free pass to use the backend's default LLM
            st.info("Admin Mode: Using system default AI engine.")
            st.session_state.pop("user_openai_key", None) # Ensure no key is passed
        else:
            # Standard users MUST enter an OpenAI key
            openai_key_input = st.text_input("OpenAI API Key", type="password", help="Required to use the AI agent.")
            if openai_key_input:
                st.session_state["user_openai_key"] = openai_key_input
                st.success("Key active for this session.")
            else:
                st.session_state.pop("user_openai_key", None)

# --- Main Content Area ---
main_col, feed_col = st.columns([7, 3])

with main_col:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.container(border=True):
        st.markdown("### Conversation History Log")
        
        # Render chat history
        for i, item in enumerate(st.session_state.chat_history):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                role = item[0]
                text = item[1]
                source = item[2] if len(item) == 3 else None
                
                with st.chat_message(role):
                    st.markdown(text)
                    if source:
                        st.caption(f"📄 **Source:** `{source}`")
                    
                    # Add Feedback UI for Assistant messages
                    if role == "assistant":
                        # Show interactive feedback only for the LATEST assistant message
                        if i == len(st.session_state.chat_history) - 1:
                            feedback = streamlit_feedback(
                                feedback_type="thumbs",
                                optional_text_label="[Optional] Provide feedback to improve the AI",
                                key=f"feedback_{i}",
                                align="flex-start"
                            )
                            if feedback:
                                st.session_state.last_feedback = feedback
                                if feedback.get("score") in ["👍", "👎"]:
                                    st.toast("Feedback recorded! Thank you.", icon="📝")

    
    user_input = st.chat_input("Ask about legacy code, team structures, or documentation...", key="chat_input")

    if user_input:
        user_key = st.session_state.get("user_openai_key")
        
        
        if not is_admin and not user_key:
            st.warning(" Please enter your OpenAI API Key in the sidebar to use the agent.")
            st.stop()
            
        st.session_state.chat_history.append(("user", user_input))
        
        # Call the backend. If admin, user_key is None -> backend uses Mistral.
        # If user, user_key is passed -> backend uses OpenAI.
        response = api_client.query_backend(user_input, st.session_state["session_id"], st.session_state["jwt_token"], openai_api_key=user_key)
        
        if isinstance(response, str) and "Error" in response:
            st.session_state.chat_history.append(("assistant", response, None))
        else:
            match = re.search(r'\[Source:\s*(.*?)\]', response)
            source_file = match.group(1) if match else None
            clean_response = re.sub(r'\[Source:\s*.*?\]', '', response).strip()
            st.session_state.chat_history.append(("assistant", clean_response, source_file))

        st.rerun()

with feed_col:
    with st.container(border=True):
        st.markdown("### System Feed & Status")
        st.markdown(
            """
            <div style='text-align: center; margin-bottom: 12px;'>
                <span class='custom-badge'>SaaS System Nominal</span>
            </div>
            <p style='font-size: 0.9em; margin-bottom: 4px; font-weight: 600;'>New Update Released!</p>
            <p style='font-size: 0.85em; color: #696482; line-height: 1.4;'>
                The <b>Socratic Onboarding Engine</b> has been initialized. You can now verify and execute isolated code evaluations safely.
            </p>
            """, 
            unsafe_allow_html=True
        )