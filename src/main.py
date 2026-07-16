"""
Main FastAPI application entry point.
Using Lifespan context manager for startup/shutdown.
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.routes import router
from src.security.headers_middleware import SecurityHeadersMiddleware
from src.security.rbac import bootstrap_admin_from_env

async def initialize_components():
    """
    Simulate or perform the vectorstore initialization.
    """
    try:
        print("Starting component initialization...")
        # Add your heavy loading tasks here
        # If blocking, use: await asyncio.to_thread(load_func)
        print("Initialization complete.")
    except Exception as e:
        print(f"Error during initialization: {e}")


async def _auto_ingest_team_config():
    """
    On startup, attempt to auto-ingest the bundled data/team_config.yaml
    into the shared FAISS index so the Contextual Team Navigator works
    out of the box. Failures are non-fatal: they are logged and the app
    continues to start (the Navigator simply won't have data until a
    config is uploaded via POST /rag/team/upload).
    """
    try:
        # Run in a thread to avoid blocking the event loop during FAISS build
        await asyncio.to_thread(_ingest_team_config_sync)
    except Exception as exc:
        print(f"[Startup] Team config auto-ingest skipped: {exc}")


def _ingest_team_config_sync():
    """Synchronous team config ingestion helper."""
    import os
    from src.navigator.team_loader import ingest_team_config, DEFAULT_TEAM_CONFIG_PATH

    if not os.path.exists(DEFAULT_TEAM_CONFIG_PATH):
        print(f"[Startup] No team_config.yaml found at {DEFAULT_TEAM_CONFIG_PATH} — skipping Navigator auto-ingest.")
        return

    # Check whether team profiles are already in the index to avoid duplicates
    try:
        from src.rag.retriever_setup import _load_persisted_documents
        persisted = _load_persisted_documents()
        already_has_team = any(
            (d.metadata if hasattr(d, 'metadata') else d.get('metadata', {})).get("doc_type") == "team_profile"
            for d in persisted
        )
        if already_has_team:
            print("[Startup] Team profiles already present in index — skipping auto-ingest to avoid duplicates.")
            return
    except Exception:
        pass  # proceed with ingestion attempt

    try:
        count = ingest_team_config(tenant_id="default_tenant")
        print(f"[Startup] Team Navigator: auto-ingested {count} team document(s).")
    except Exception as exc:
        print(f"[Startup] Team config ingestion failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan context manager for startup and shutdown tasks.
    """
    # Startup logic
    print("Application startup triggered.")
    # Bootstrap a default admin account from the environment if configured.
    # Non-fatal if JWT_SECRET or admin creds are absent.
    try:
        bootstrap_admin_from_env()
    except Exception as exc:
        print(f"[Startup] Admin bootstrap skipped: {exc}")
    await initialize_components()
    await _auto_ingest_team_config()
    yield
    # Shutdown logic (if any)
    print("Application shutting down.")

# Initialize the FastAPI app with the lifespan
app = FastAPI(title="Adaptive RAG API", lifespan=lifespan)

# Security headers middleware. Registered first so it runs last in the
# outbound chain and can rewrite error responses on every path.
app.add_middleware(SecurityHeadersMiddleware)

# Include your routes
app.include_router(router, prefix="/api")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Adaptive RAG API is running"}