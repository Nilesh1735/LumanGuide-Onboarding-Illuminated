<div align="center">
<a href="https://github.com/Nilesh1735/LumanGuide-Onboarding-Illuminated">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:10b981,100:059669&height=120&section=header" width="100%" />
</a>
</div>

# LumanGuide - Onboarding, Illuminated

<div align="center">
<p>An enterprise-grade, intelligent Retrieval-Augmented Generation (RAG) system powered by a LangGraph state machine. Features dynamic LLM routing, real-time agent telemetry, and a premium SaaS UI.</p>

<img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
<img src="https://img.shields.io/badge/LangGraph-35495E?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph" />
<img src="https://img.shields.io/badge/FAISS-222222?style=for-the-badge&logo=facebook&logoColor=white" alt="FAISS" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
<img src="https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white" alt="MongoDB" />

</div>

## Overview

LumanGuide is a stateful, agentic AI platform that uses a LangGraph state machine to intelligently route queries to the most appropriate data source—indexed internal documents (FAISS), general LLM knowledge, or real-time web search (Tavily). 

Unlike standard RAG implementations, LumanGuide features a **Dynamic 3-Tier LLM Router** (Mistral → OpenAI → Gemini), a **Contextual Team Navigator** for SME routing, **MongoDB Atlas** for persistent cloud chat history, and a robust AppSec suite to prevent prompt injection and secret leakage. The frontend is built as a premium B2B SaaS application, featuring a split-screen landing page, live system telemetry, and an agent thought-process terminal.

## Tech Stack

<div align="center">

<table>
<tr>
<td><b>Layer</b></td>
<td><b>Technology</b></td>
</tr>
<tr>
<td>AI & GenAI</td>
<td>LangChain, LangGraph, Mistral AI, OpenAI, Google Gemini, Tavily, FAISS</td>
</tr>
<tr>
<td>Backend</td>
<td>FastAPI, Pydantic, MongoDB Atlas (Motor), JWT Bcrypt Auth</td>
</tr>
<tr>
<td>Frontend</td>
<td>Streamlit, Space Grotesk/Mono Typography, Custom CSS</td>
</tr>
<tr>
<td>Security & LLMOps</td>
<td>JWT RBAC, LangSmith, pip-audit, Regex Secret Scanner</td>
</tr>
</table>

<br>

<img src="https://skillicons.dev/icons?i=python" alt="Python" />
<img src="https://skillicons.dev/icons?i=fastapi" alt="FastAPI" />
<img src="https://skillicons.dev/icons?i=mongodb" alt="MongoDB" />
<img src="https://skillicons.dev/icons?i=sqlite" alt="SQLite" />
<img src="https://skillicons.dev/icons?i=aws" alt="AWS" />
<br><br>
<img src="https://skillicons.dev/icons?i=git" alt="Git" />
<img src="https://skillicons.dev/icons?i=github" alt="GitHub" />
<img src="https://skillicons.dev/icons?i=docker" alt="Docker" />

</div>

## Enterprise Architecture & UI Upgrades

This repository includes production-grade, enterprise-level architectural implementations:

1. **Dynamic 3-Tier LLM Router (`src/llms/openai.py`):** Automatically falls back from Mistral → OpenAI → Google Gemini based on API availability and rate limits, ensuring zero downtime.
2. **Premium SaaS UI (`streamlit_app/`):** A custom-built split-screen landing page with live system telemetry, Space Grotesk typography, and a Dark Emerald theme.
3. **Agent Telemetry Terminal:** Real-time UI logging that displays the LangGraph state machine's execution steps (e.g., *Synthesizing, Untangling, Crunching*) to the user.
4. **RAG Transparency:** Source citation badges and an "Inspect retrieved context" expander for every AI response, proving zero hallucination.
5. **AppSec Security Suite:**
   - **Security Headers Middleware:** Enforces CSP, HSTS, X-Frame-Options.
   - **Secret Scanner:** Scans uploaded documents for AWS keys, JWTs, and Slack tokens before embedding, redacting them as `[REDACTED_SECRET]`.
   - **Prompt Injection Guardrail:** A LangGraph node that blocks jailbreak attempts.
   - **RBAC (`src/security/rbac.py`):** JWT-based Role-Based Access Control (admin > contributor > viewer).

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Web Application                     │
│  • Split-Screen Auth • Live Telemetry • Agent Terminal UI       │
│  • Team Navigator Grid • Source Citation Badges                 │
└───────────────────────────────────┬─────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Async)                       │
│ ┌────────────────────┐ ┌────────────────────┐ ┌──────────────┐ │
│ │ Security Headers   │ │ RBAC (JWT verify)  │ │ Rate Limiter │ │
│ └────────────────────┘ └────────────────────┘ └──────────────┘ │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │                   REST API Endpoints                         │ │
│ │  POST /api/rag/query • POST /api/rag/documents/upload       │ │
│ └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────┬─────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestration                       │
│ ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐     │
│ │Guardrail│ -> │ Classify │ -> │ Router  │ -> │ Pipeline │     │
│ └─────────┘    └──────────┘    └─────────┘    └──────────┘     │
└───────────────────────────────────┬─────────────────────────────┘
          ↓                 ↓                 ↓
┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
│  Retriever      │ │  Dynamic LLM │ │  Web Search      │
│  (FAISS + Gemini│ │  Router      │ │  (Tavily)        │
│   Embeddings)   │ │  (Mistral/OAI│ │                  │
└─────────────────┘ └──────────────┘ └──────────────────┘
          ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MongoDB Atlas (Cloud)                         │
│       Persistent User Auth & Chat History Isolation             │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```text
LumanGuide/
├── data/
│   └── team_config.yaml         # Contextual Team Navigator SME list
├── src/
│   ├── main.py                  # FastAPI app entry & middleware setup
│   ├── api/
│   │   ├── routes.py            # RAG query, document, and team endpoints
│   │   └── auth.py              # Login, Signup, JWT & Bcrypt generation
│   ├── core/
│   │   ├── config.py            # Settings loader
│   │   ├── logger.py            # Logging setup
│   │   └── telemetry.py         # LangSmith LLMOps tracing
│   ├── db/
│   │   └── mongo_client.py      # MongoDB Atlas async client (Motor)
│   ├── llms/
│   │   ├── openai.py            # Dynamic 3-Tier LLM Router
│   │   └── router.py            # NVIDIA NIM fallback router
│   ├── rag/
│   │   ├── graph_builder.py     # LangGraph state machine construction
│   │   ├── guardrail_node.py    # Prompt injection defense
│   │   ├── multimodal_ingestor.py # GPT-4o Vision PDF processing
│   │   ├── retriever_setup.py   # FAISS vector store setup
│   │   ├── document_upload.py   # Document processing & secret scanning
│   │   └── reAct_agent.py       # ReAct agent setup
│   ├── security/
│   │   ├── headers_middleware.py# CSP, HSTS, X-Frame-Options
│   │   ├── rbac.py              # JWT RBAC (admin/contributor/viewer)
│   │   └── secret_scanner.py    # Regex secret detection & redaction
│   └── tools/
│       ├── common_tools.py      # Shared utilities
│       ├── graph_tools.py       # Graph routing logic
│       └── slack_tool.py        # Agentic Slack notification tool
├── streamlit_app/
│   ├── Home.py                  # Split-screen SaaS landing & auth page
│   ├── pages/
│   │   └── Chat.py              # Chat interface, telemetry terminal, UI
│   ├── components/
│   │   └── team_graph.py        # Interactive team graph logic
│   └── utils/
│       ├── api_client.py        # Backend HTTP client
│       └── theme.py             # Dark Emerald CSS & Space Grotesk fonts
├── tests/
│   └── eval_pipeline.py         # RAGAS automated evaluation script
├── .env.example                 # Environment variable template
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Getting Started

### Prerequisites
- **Python 3.12+** 
- API Keys: Google Gemini (AI Studio), Mistral AI, OpenAI, Tavily (for Search)
- MongoDB Atlas connection string (for cloud auth & history)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nilesh1735/LumanGuide-Onboarding-Illuminated.git
   cd LumanGuide-Onboarding-Illuminated
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Environment Configuration

Create a `.env` file in the project root. **Never commit this file.**

```env
# --- Core AI Configuration ---
GOOGLE_API_KEY=AIzaSyYOUR_GEMINI_KEY
MISTRAL_API_KEY=YOUR_MISTRAL_KEY
OPENAI_API_KEY=sk-YOUR_OPENAI_KEY
TAVILY_API_KEY=tvly-YOUR_TAVILY_KEY

# --- Database (MongoDB Atlas) ---
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority

# --- Security & RBAC ---
JWT_SECRET=YOUR_SUPER_SECRET_JWT_KEY_HERE
JWT_ALGORITHM=HS256

# --- LLMOps (LangSmith) ---
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_YOUR_LANGSMITH_KEY
LANGCHAIN_PROJECT=lumanguide
```

## Running the Application

You need two terminal windows (both with the virtual environment activated).

**Terminal 1: Start the FastAPI Backend**
```powershell
python -m uvicorn src.main:app --reload --port 8000
```
*Wait until you see `INFO: Application startup complete.`*

**Terminal 2: Start the Streamlit Frontend**
```powershell
python -m streamlit run streamlit_app/Home.py
```

Access the application at `http://localhost:8501`.

## Connect With Me

<div align="center">
<a href="https://www.linkedin.com/in/nilesh-raj-nr1735/">
<img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" />
</a>
<a href="mailto:nileshraj1735@gmail.com">
<img src="https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email" />
</a>
</div>

<br>

<div align="center">
<a href="https://github.com/Nilesh1735">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:10b981,100:059669&height=120&section=footer" width="100%" />
</a>
</div>
```