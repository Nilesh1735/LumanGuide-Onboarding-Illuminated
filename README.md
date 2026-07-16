<div align="center">
<a href="https://github.com/Nilesh1735/LumanGuide---Onboarding-Illuminated">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:7dd3fc,100:0ea5e9&height=120&section=header" width="100%" />
</a>
</div>

# LumanGuide - Onboarding, Illuminated

<div align="center">
<p>An enterprise-grade, intelligent Retrieval-Augmented Generation (RAG) system powered by agentic AI architecture, designed to streamline engineering onboarding and knowledge management.</p>

<img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
<img src="https://img.shields.io/badge/LangGraph-35495E?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph" />
<img src="https://img.shields.io/badge/FAISS-222222?style=for-the-badge&logo=facebook&logoColor=white" alt="FAISS" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />

</div>

## Overview

LumanGuide is a stateful, agentic AI platform that uses a LangGraph state machine to intelligently route queries to the most appropriate data source—indexed internal documents (FAISS), general LLM knowledge, or real-time web search (Tavily). 

Unlike standard RAG implementations, LumanGuide features a Resilient Fallback Architecture (NVIDIA NIM and SQLite), a Contextual Team Navigator for SME routing, LLMOps observability (LangSmith), and a robust AppSec suite to prevent prompt injection and secret leakage.

## Tech Stack

<div align="center">

<table>
<tr>
<td><b>Layer</b></td>
<td><b>Technology</b></td>
</tr>
<tr>
<td>AI & GenAI</td>
<td>LangChain, LangGraph, Google Gemini, OpenAI API, Tavily, FAISS</td>
</tr>
<tr>
<td>Backend</td>
<td>FastAPI, Pydantic, MongoDB (Motor), SQLite</td>
</tr>
<tr>
<td>Frontend</td>
<td>Streamlit, Custom CSS, streamlit-agraph</td>
</tr>
<tr>
<td>Security & LLMOps</td>
<td>JWT (RBAC), LangSmith, pip-audit, Regex Secret Scanner</td>
</tr>
</table>

<br>

<img src="https://skillicons.dev/icons?i=python" alt="Python" />
<img src="https://skillicons.dev/icons?i=fastapi" alt="FastAPI" />
<img src="https://skillicons.dev/icons?i=sqlite" alt="SQLite" />
<img src="https://skillicons.dev/icons?i=postgres" alt="PostgreSQL" />
<img src="https://skillicons.dev/icons?i=aws" alt="AWS" />
<br><br>
<img src="https://skillicons.dev/icons?i=scikitlearn" alt="Scikit-learn" />
<img src="https://skillicons.dev/icons?i=tensorflow" alt="Transformers/BERT" />
<br><br>
<img src="https://skillicons.dev/icons?i=git" alt="Git" />
<img src="https://skillicons.dev/icons?i=github" alt="GitHub" />
<img src="https://skillicons.dev/icons?i=docker" alt="Docker" />
<img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />

</div>

## Enterprise Architecture Upgrades

This repository includes production-grade, enterprise-level architectural implementations:

1. **LLMOps Observability (`src/core/telemetry.py`):** LangSmith integration with a `@trace_node` decorator that logs inputs, outputs, and token usage for every step in the LangGraph workflow.
2. **Agentic Slack Tool (`src/tools/slack_tool.py`):** A LangChain `BaseTool` allowing the AI agent to physically post messages to Slack channels to notify SMEs.
3. **Multimodal Vision Ingestion (`src/rag/multimodal_ingestor.py`):** Converts PDF diagrams to images and uses GPT-4o to transcribe them into Markdown.
4. **Automated RAG Evaluation (`tests/eval_pipeline.py`):** A CI/CD-ready script using RAGAS to calculate `faithfulness`, `answer_relevancy`, and `context_precision`.
5. **AppSec Security Suite:**
   - **Security Headers Middleware:** Enforces CSP, HSTS, X-Frame-Options.
   - **Secret Scanner:** Scans uploaded documents for AWS keys, JWTs, and Slack tokens before embedding, redacting them as `[REDACTED_SECRET]`.
   - **Prompt Injection Guardrail:** A LangGraph node that blocks jailbreak attempts and sanitizes NoSQL/template injection vectors.
   - **RBAC (`src/security/rbac.py`):** JWT-based Role-Based Access Control (admin > contributor > viewer).

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│ ┌─────────────────────────────────────────────────────────────┐ │
│               Streamlit Web Application (Custom Theme)         │ │
│  • Auth (Login/Signup) • Chat • Document Upload • Team Graph   │ │
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
│  Retriever      │ │  General LLM │ │  Web Search      │
│  (FAISS + Vision│ │  (Gemini/    │ │  (Tavily)        │
│   Multimodal)   │ │   OpenAI)    │ │                  │
└─────────────────┘ └──────────────┘ └──────────────────┘
```

## Project Structure

```text
LumanGuide/
├── .github/workflows/
│   └── dependency-audit.yml     # pip-audit CI/CD integration
├── data/
│   └── team_config.yaml         # Contextual Team Navigator SME list
├── src/
│   ├── main.py                  # FastAPI app entry & middleware setup
│   ├── api/
│   │   ├── routes.py            # RAG query, document, and team endpoints
│   │   └── auth.py              # Login, Signup, JWT generation
│   ├── core/
│   │   ├── config.py            # Settings loader
│   │   ├── logger.py            # Logging setup
│   │   └── telemetry.py         # LangSmith LLMOps tracing
│   ├── llms/
│   │   ├── openai.py            # Primary LLM (Gemini/OpenAI)
│   │   └── router.py            # NVIDIA NIM fallback router
│   ├── memory/
│   │   ├── chat_history_mongo.py# MongoDB persistent history
│   │   ├── chat_history_sqlite.py# SQLite persistent fallback
│   │   └── chathistory_in_memory.py
│   ├── ml_pipeline/
│   │   └── intent_classifier.py # BERT + TF + Random Forest router
│   ├── rag/
│   │   ├── graph_builder.py     # LangGraph state machine construction
│   │   ├── nodes.py             # Graph node implementations
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
│   ├── Home.py                  # Authentication and login page
│   ├── pages/
│   │   ├── Chat.py              # Chat interface and document upload
│   │   └── Admin_Telemetry.py   # LangSmith dashboard embed
│   └── components/
│       └── team_graph.py        # 3D interactive team graph (streamlit-agraph)
├── tests/
│   └── eval_pipeline.py         # RAGAS automated evaluation script
├── .env.example                 # Environment variable template
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Getting Started

### Prerequisites
- **Python 3.12+** (Do not use Python 3.14 due to `traceback.py` bugs)
- **Poppler** (for PDF image rasterization in multimodal ingestion)
- API Keys: Google Gemini (AI Studio), OpenAI (for Vision), Tavily (for Search)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nilesh1735/LumanGuide---Onboarding-Illuminated.git
   cd LumanGuide---Onboarding-Illuminated
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
OPENAI_API_KEY=sk-YOUR_OPENAI_KEY
TAVILY_API_KEY=tvly-YOUR_TAVILY_KEY

# --- LLM Fallback (NVIDIA NIM) ---
NVIDIA_NIM_BASE_URL=http://localhost:8000/v1
NVIDIA_NIM_MODEL=meta-llama/Llama-3-8B-Instruct

# --- Database (MongoDB - Optional) ---
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=lumanguide

# --- Security & RBAC ---
JWT_SECRET=YOUR_SUPER_SECRET_JWT_KEY_HERE
JWT_ALGORITHM=HS256
LUMANGUIDE_ADMIN_USER=admin
LUMANGUIDE_ADMIN_PASSWORD=ChangeThisPassword123!

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
*(Using `python -m streamlit` bypasses Windows Application Control policy blocks on `.exe` files).*

Access the application at `http://localhost:8501`.

## API Endpoints

### Query the RAG System
```http
POST /api/rag/query
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json

{
  "query": "What is the main topic of the document?",
  "session_id": "user_session_123"
}
```

### Upload Document (Requires `contributor` or `admin` role)
```http
POST /api/rag/documents/upload
Authorization: Bearer <JWT_TOKEN>
X-Description: Engineering Runbook 2024

Form Data:
- file: <PDF or TXT file>
```

## Security and RBAC

LumanGuide implements a strict security posture:
1. **Transport Security:** The FastAPI middleware injects `Content-Security-Policy`, `X-Frame-Options: DENY`, and `Strict-Transport-Security`.
2. **Secret Scanning:** Before any document is chunked and embedded into FAISS, `src/security/secret_scanner.py` regex-scans the text for AWS keys, Slack tokens, JWTs, and PEM private keys, replacing them with `[REDACTED_SECRET]`.
3. **Prompt Injection Defense:** The LangGraph `guardrail_node` intercepts queries attempting to override system prompts (e.g., "ignore previous instructions") and returns a `403 Forbidden`.
4. **Role-Based Access Control:** JWT-based authentication. Users are assigned `admin`, `contributor`, or `viewer` roles. Document ingestion requires `contributor` privileges.

## Testing and Evaluation

### Automated RAG Evaluation (RAGAS)
Run the RAGAS evaluation pipeline to test AI accuracy, faithfulness, and context precision against a synthetic golden dataset.

```bash
python -m tests.eval_pipeline
```

### Dependency Auditing
A GitHub Actions workflow (`.github/workflows/dependency-audit.yml`) runs `pip-audit` on every push to ensure no known high-severity CVEs are present in the supply chain.

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
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:7dd3fc,100:0ea5e9&height=120&section=footer" width="100%" />
</a>
</div>
```
