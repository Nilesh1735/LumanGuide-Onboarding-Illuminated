from fastapi import APIRouter, HTTPException, UploadFile, File, Header
from langchain_core.messages import HumanMessage, AIMessage
import asyncio
import logging
from src.api.auth import auth_router
from src.memory.chat_history_mongo import ChatHistory
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder
from src.rag.retriever_setup import get_retriever, get_all_documents
from src.navigator.team_loader import load_team_config, team_config_to_documents
import re

logger = logging.getLogger(__name__)

# Initialize router and include auth routes
router = APIRouter()
router.include_router(auth_router, tags=["auth"])

@router.post("/rag/query")
async def rag_query(req: QueryRequest):
    """
    Process a RAG query through the State-Driven Adaptive RAG pipeline.
    """
    chat_history = ChatHistory.get_session_history(req.session_id)

    try:
        asyncio.create_task(chat_history.add_message(HumanMessage(content=req.query)))
    except Exception as exc:
        logger.warning("Failed to schedule history write: %s", exc)

    try:
        messages = await asyncio.wait_for(chat_history.get_messages(), timeout=3.0)
        # FIX: Ensure the current user query is strictly the last message in the list
        if not messages:
            messages = [HumanMessage(content=req.query)]
        else:
            # Append the current query to guarantee it's the final HumanMessage
            messages.append(HumanMessage(content=req.query))
    except Exception as exc:
        logger.warning("Could not load chat history (falling back to single-message context): %s", exc)
        messages = [HumanMessage(content=req.query)]

    # Explicit file fallback
    try:
        trigger_phrase = bool(re.search(r"\b(from this file|provide answer from this file|answer from this file|from the uploaded file)\b", req.query, re.I))
        trigger_use_latest = getattr(req, 'use_latest', False)
        trigger_index = getattr(req, 'persisted_doc_index', None)

        if trigger_phrase or trigger_use_latest or (trigger_index is not None):
            persisted_docs = get_all_documents()
            if not persisted_docs:
                answer = "No persisted documents were found to answer from. Please upload a file first."
                try:
                    asyncio.create_task(chat_history.add_message(AIMessage(content=answer)))
                except Exception:
                    pass
                return {"result": {"content": answer, "diagnostics": {"docs_scanned": 0}}}

            chosen = None
            if trigger_use_latest:
                chosen = persisted_docs[-1]
            elif trigger_index is not None:
                try:
                    idx = int(trigger_index)
                    if idx < 0: idx = len(persisted_docs) + idx
                    if 0 <= idx < len(persisted_docs): chosen = persisted_docs[idx]
                except Exception:
                    chosen = None
            elif trigger_phrase:
                chosen = persisted_docs[-1]

            if chosen is None:
                answer = "Could not locate the requested persisted document (index may be out of range)."
                try:
                    asyncio.create_task(chat_history.add_message(AIMessage(content=answer)))
                except Exception:
                    pass
                return {"result": {"content": answer, "diagnostics": {"docs_scanned": len(persisted_docs)}}}

            text = chosen.page_content if hasattr(chosen, 'page_content') else str(chosen)
            clean_text = text.replace('\ufeff', '').strip()
            snippet = clean_text[:4000]
            reply = f"Answering from the selected persisted document:\n\n{snippet}"
            diagnostics = {"docs_scanned": 1, "sample_snippet": snippet[:200]}
            
            try:
                asyncio.create_task(chat_history.add_message(AIMessage(content=reply)))
            except Exception:
                pass
            return {"result": {"content": reply, "diagnostics": diagnostics}}
    except Exception as e:
        logger.exception("Explicit-file fallback failed: %s", e)

    # 4. Quick rule-based handlers for simple numeric queries
    try:
        numeric_q_pattern = re.compile(r"\b(max|biggest|largest|highest|min|minimum|lowest|smallest|how many|count|sum|total|multiply|product)\b", re.I)
        if numeric_q_pattern.search(req.query):
            logger.info("Numeric fallback triggered for query: %s", req.query)
            try:
                retriever = get_retriever()
                docs = []
                if retriever:
                    try:
                        invoke_res = retriever.invoke(req.query)  # type: ignore
                        docs = invoke_res if isinstance(invoke_res, list) else [invoke_res]
                    except: pass

                number_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
                found_numbers = []
                docs_searched = 0
                sample_snippet = None
                for d in docs:
                    text = d.page_content if hasattr(d, "page_content") else str(d)
                    if text:
                        docs_searched += 1
                        if sample_snippet is None: sample_snippet = text[:200]
                        for m in number_pattern.findall(text):
                            try: found_numbers.append(float(m))
                            except: continue

                if found_numbers:
                    q = req.query.lower()
                    if re.search(r"\b(min|minimum|lowest|smallest)\b", q): res_val, op_name = min(found_numbers), "lowest"
                    elif re.search(r"\b(sum|total)\b", q): res_val, op_name = sum(found_numbers), "sum"
                    elif re.search(r"\b(multiply|product)\b", q):
                        prod = 1.0
                        for n in found_numbers: prod *= n
                        res_val, op_name = prod, "product"
                    elif re.search(r"\b(count|how many)\b", q): res_val, op_name = len(found_numbers), "count"
                    else: res_val, op_name = max(found_numbers), "largest"

                    display_val = str(int(res_val)) if float(res_val).is_integer() else f"{res_val:.6g}"
                    answer = f"The {op_name} value found in documents is: {display_val}"
                    diagnostics = {"docs_scanned": docs_searched, "sample_snippet": sample_snippet}
                    asyncio.create_task(chat_history.add_message(AIMessage(content=answer)))
                    return {"result": {"content": answer, "diagnostics": diagnostics}}
            except Exception as e:
                logger.debug("Numeric fallback failed: %s", e)
    except Exception:
        pass

    # 5. Invoke the compiled graph
    try:
        result = builder.invoke({  # type: ignore
            "messages": messages,
            "latest_query": req.query,
            "consecutive_errors": 0
        })
    except Exception as exc:
        logger.error("Graph invocation failed: %s", exc)
        try:
            persisted_docs = get_all_documents()
            if persisted_docs:
                snippets = [ (d.page_content if hasattr(d,'page_content') else str(d)).strip()[:400] for d in persisted_docs[:3] ]
                if snippets:
                    reply = "I couldn't reach the configured LLM, but here are top snippets from your persisted documents:\n\n" + "\n\n---\n\n".join(snippets)
                    diagnostics = {"docs_scanned": len(persisted_docs), "sample_snippet": snippets[0]}
                    try: asyncio.create_task(chat_history.add_message(AIMessage(content=reply)))
                    except: pass
                    return {"result": {"content": reply, "diagnostics": diagnostics}}
        except Exception as ret_exc:
            logger.exception("Retrieval-only fallback also failed: %s", ret_exc)

        fallback_text = "The RAG pipeline encountered an error while processing your query. This usually means the configured LLM is unavailable."
        try: asyncio.create_task(chat_history.add_message(AIMessage(content=fallback_text)))
        except: pass
        return {"result": {"content": fallback_text}}

    output_message = result["messages"][-1]
    output_text = output_message.content

    try:
        asyncio.create_task(chat_history.add_message(AIMessage(content=output_text)))
    except Exception as exc:
        logger.warning("Failed to schedule assistant message write: %s", exc)

    # FIX: Return a clean dictionary with just the text string
    return {"result": {"content": output_text}}

@router.get("/rag/persisted_docs")
def list_persisted_docs():
    """Return a list of persisted documents with short snippets and metadata for the UI document picker."""
    try:
        docs = get_all_documents()
        out = []
        for i, d in enumerate(docs):
            text = d.page_content if hasattr(d, 'page_content') else str(d)
            snippet = (text.replace('\ufeff', '').strip()[:400]) if text else ""
            meta = d.metadata if hasattr(d, 'metadata') else {}
            out.append({"index": i, "snippet": snippet, "metadata": meta})
        return {"documents": out}
    except Exception as exc:
        logger.exception("Failed to list persisted docs: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/rag/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description")
):
    """Upload a document for RAG processing."""
    try:
        status_upload = documents(description, file)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document upload failed: {exc}") from exc
    return {"status": status_upload}

@router.post("/rag/team/upload")
async def upload_team_config_endpoint(file: UploadFile = File(...)):
    """Ingest a team_config.yaml file into the shared FAISS vector index."""
    import tempfile
    import os

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".yaml", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            team_data = load_team_config(path=tmp_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse YAML: {exc}") from exc

        docs = team_config_to_documents(team_data, tenant_id="default_tenant")
        if not docs:
            raise HTTPException(status_code=400, detail="Team config parsed but produced no documents.")

        from src.rag.retriever_setup import add_documents_to_retriever
        try:
            add_documents_to_retriever(docs, tenant_id="default_tenant")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"FAISS ingestion failed: {exc}") from exc

        return {"status": "ok", "members_ingested": len(team_data), "documents_ingested": len(docs)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.unlink(tmp_path)
            except: pass

@router.get("/rag/team/status")
def team_status():
    """Return whether team navigator data is loaded, plus the count of team profile documents."""
    try:
        docs = get_all_documents()
        team_docs = [
            d for d in docs
            if (d.metadata if hasattr(d, 'metadata') else {}).get("doc_type") == "team_profile"
        ]
        member_names = set()
        for d in team_docs:
            meta = d.metadata if hasattr(d, 'metadata') else {}
            member_names.add(meta.get("team_member_name", "Unknown"))

        return {
            "navigator_loaded": len(team_docs) > 0,
            "team_doc_count": len(team_docs),
            "member_count": len(member_names),
            "members": sorted(member_names),
        }
    except Exception as exc:
        logger.exception("Team status check failed: %s", exc)
        return {
            "navigator_loaded": False,
            "team_doc_count": 0,
            "member_count": 0,
            "members": [],
            "error": str(exc),
        }