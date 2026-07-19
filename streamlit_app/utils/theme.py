"""
Shared theming for the LumanGuide Streamlit app.
Design System: 'GitHub Midnight' (Strict Dark, High Contrast, Purple Accents).
Typography: Space Grotesk (UI) & Space Mono (Code/Terminal).
"""

def get_custom_css(page: str = "home") -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-main: #0d1117;          /* Deep GitHub Black */
    --bg-card: #161b22;          /* Charcoal for cards & sidebar */
    --bg-input: #0d1117;         /* Deep black for inputs */
    
    --text-primary: #f0f6fc;     /* Eye-friendly Off-White */
    --text-secondary: #8b949e;   /* GitHub Secondary Grey */
    
    /* Font System */
    --font-main: 'Space Grotesk', sans-serif;
    --font-mono: 'Space Mono', monospace;
    
    --accent-purple: #a855f7;    /* Vibrant Purple */
    --accent-blue: #58a6ff;      /* GitHub Blue */
    --accent-green: #3fb950;     /* GitHub Green */
    --border-color: #30363d;     /* Slate Grey */
    --radius-lg: 16px;
    --radius-md: 8px;
}

/* 1. Global Canvas & Typography */
html, body, [class*="css"], .stApp {
    font-family: var(--font-main) !important; 
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
}

#MainMenu { display: none !important; }
footer { display: none !important; }
[data-testid="stLogo"], [data-testid="stLogoSmall"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* 2. Typography Hierarchy */
h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-main) !important; 
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

p, span, li, div {
    color: var(--text-primary) !important;
}

a {
    color: var(--accent-blue) !important; 
    text-decoration: none !important;
}

/* 3. Sidebar & Cards */
[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border-color) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
    padding: 2rem !important;
    margin-bottom: 1.5rem !important;
}

/* 4. Form Inputs */
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-color) !important;
    background-color: var(--bg-input) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-main) !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s ease;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent-purple) !important;
    box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.15) !important;
}

.stTextInput > label, .stTextArea > label {
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important; 
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    margin-bottom: 8px;
}

/* 5. Buttons */
.stButton button {
    background-color: #21262d !important; 
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-main) !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    padding: 10px 24px !important;
}

.stButton button:hover {
    border-color: var(--accent-purple) !important;
    color: var(--accent-purple) !important;
    background-color: #30363d !important;
}

button[kind="primary"] {
    background-color: var(--accent-purple) !important;
    color: #ffffff !important;
    border: none !important;
}

button[kind="primary"]:hover {
    background-color: #9333ea !important; 
    box-shadow: 0 4px 12px rgba(168, 85, 247, 0.3) !important;
}

/* 6. Chat Messages */
[data-testid="stChatMessage"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-lg) !important; 
    padding: 16px 20px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
}

[data-testid="stChatMessage"][data-testid*="user"] {
    border-left: 4px solid var(--accent-purple) !important;
}

[data-testid="stChatMessage"]:not([data-testid*="user"]) {
    border-left: 4px solid var(--accent-blue) !important;
}

[data-testid="stChatMessageContent"] p {
    color: var(--text-primary) !important;
    font-size: 0.95rem;
    line-height: 1.5;
}

/* 7. Eliminate White Boxes */
[data-testid="stChatInput"] {
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-lg) !important;
    background-color: var(--bg-card) !important;
    padding: 8px !important;
}
[data-testid="stChatInput"] textarea {
    background-color: var(--bg-input) !important;
    color: var(--text-primary) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-main) !important;
}

[data-testid="stAlert"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
}
[data-testid="stAlert"] * { color: var(--text-primary) !important; }

[data-testid="stExpander"] {
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stExpander"] details summary {
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: var(--bg-input) !important;
    border: 1px dashed var(--border-color) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent-purple) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: var(--text-secondary) !important;
    font-family: var(--font-main) !important;
}

/* 8. Code Blocks & Terminal */
[data-testid="stCodeBlock"] {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-color) !important;
    background-color: var(--bg-input) !important;
}
pre code {
    font-family: var(--font-mono) !important; 
    font-size: 0.9em !important;
}

/* 9. Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-purple); }

/* 10. Feedback UI */
div[data-testid="stFeedback"] { background-color: transparent !important; margin-top: 10px; }
div[data-testid="stFeedback"] button {
    background-color: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #f0f6fc !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stFeedback"] button:hover {
    border-color: #a855f7 !important;
    color: #a855f7 !important;
}
div[data-testid="stFeedback"] button svg { fill: #f0f6fc !important; }
div[data-testid="stFeedback"] button[aria-pressed="true"] {
    background-color: #a855f7 !important;
    border-color: #a855f7 !important;
    color: #ffffff !important;
}
div[data-testid="stFeedback"] button[aria-pressed="true"] svg { fill: #ffffff !important; }

/* --- SPLIT SCREEN & LANDING PAGE CUSTOM CSS --- */
.landing-grid {
    display: flex;
    height: 85vh;
    gap: 2rem;
    margin-top: 2rem;
}
.left-panel {
    flex: 1.2;
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 3rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.right-panel {
    flex: 0.8;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.brand-title {
    font-family: var(--font-main) !important;
    font-size: 3.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.04em !important;
}
.brand-subtitle {
    font-family: var(--font-mono) !important;
    color: var(--accent-purple) !important;
    font-size: 1.2rem !important;
    margin-bottom: 3rem !important;
}
.capabilities {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 3rem;
}
.cap-item {
    font-family: var(--font-main) !important;
    color: var(--text-primary) !important;
    font-size: 1.1rem !important;
    display: flex;
    align-items: center;
    gap: 12px;
}
.cap-dot {
    width: 8px;
    height: 8px;
    background-color: var(--accent-green);
    border-radius: 50%;
    display: inline-block;
}
.terminal-box {
    background-color: var(--bg-input);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 1.5rem;
    font-family: var(--font-mono) !important;
}
.terminal-header {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 0.5rem;
}
.terminal-line {
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
    margin-bottom: 0.5rem;
}
.ok-text { color: var(--accent-green) !important; }
.tech-pills {
    display: flex;
    gap: 12px;
    justify-content: center;
    margin-top: 2rem;
    flex-wrap: wrap;
}
.tech-pill {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    color: var(--text-secondary) !important;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-family: var(--font-mono) !important;
}
.auth-container {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 2.5rem;
}
</style>
"""