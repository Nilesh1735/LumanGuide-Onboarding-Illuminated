from pathlib import Path
p=Path(r"c:\Users\LOQ\OneDrive\Desktop\Adaptive-Rag-main\streamlit_app\utils\api_client.py")
s=p.read_text(encoding='utf-8')
# Replace query_backend block
start = s.find('\ndef query_backend(')
if start!=-1:
    next_def = s.find('\ndef query_backend_diagnostic(', start+1)
    if next_def!=-1:
        new = '''\ndef query_backend(query: str, session_id: str, jwt_token: str, use_latest: bool = False, persisted_doc_index: int | None = None):
    """Send a query to the RAG backend."""
    headers = {"Authorization": f"******", "Content-Type": "application/json"}
    payload = {"query": query, "session_id": session_id}
    if use_latest:
        payload["use_latest"] = True
    if persisted_doc_index is not None:
        payload["persisted_doc_index"] = persisted_doc_index
    try:
        response = _post_with_fallback(
            "rag/query",
            payload,
            headers,
        )
        if response.status_code == 200:
            payload = _parse_response(response)
            return payload.get("result", {}).get("content", "No content found.")
        return f"Error: {_error_message(response)}"
    except Exception as exc:
        logger.error("Query request failed: %s", exc)
        return f"Connection error: {exc}"
'''
        s = s[:start] + new + s[next_def:]

# Replace query_backend_diagnostic block
start = s.find('\ndef query_backend_diagnostic(')
if start!=-1:
    # find following double newline after function end by locating next '\n\ndef '
    next_func = s.find('\n\ndef ', start+1)
    if next_func==-1:
        next_func = len(s)
    new = '''\ndef query_backend_diagnostic(query: str, session_id: str, jwt_token: str, use_latest: bool = False, persisted_doc_index: int | None = None):
    """Send a query to the RAG backend and return diagnostic info for the UI.

    Returns a dict with keys:
      - ok: bool
      - content: str (when ok)
      - error: str (when not ok)
      - attempts: list[str] (optional diagnostic lines)
    """
    headers = {"Authorization": f"******", "Content-Type": "application/json"}
    payload = {"query": query, "session_id": session_id}
    if use_latest:
        payload["use_latest"] = True
    if persisted_doc_index is not None:
        payload["persisted_doc_index"] = persisted_doc_index
    try:
        response = _post_with_fallback(
            "rag/query",
            payload,
            headers,
        )
        if response.status_code == 200:
            payload = _parse_response(response)
            result = payload.get("result", {})
            diagnostics = result.get("diagnostics")
            ret = {"ok": True, "content": result.get("content", "No content found.")}
            if diagnostics:
                ret["diagnostics"] = diagnostics
            return ret
        return {"ok": False, "error": _error_message(response), "attempts": []}
    except requests.HTTPError as http_err:
        msg = str(http_err)
        attempts: list[str] = []
        if "Attempts:\n" in msg:
            try:
                parts = msg.split("Attempts:\n", 1)[1]
                attempts = [line.strip() for line in parts.splitlines() if line.strip()]
            except Exception:
                attempts = []
        logger.error("Query HTTP error: %s", msg)
        return {"ok": False, "error": msg, "attempts": attempts}
    except requests.ConnectionError as conn_err:
        logger.error("Query connection error: %s", conn_err)
        return {"ok": False, "error": str(conn_err), "attempts": []}
    except Exception as exc:
        logger.error("Query request failed: %s", exc)
        return {"ok": False, "error": str(exc), "attempts": []}
'''
        s = s[:start] + new + s[next_func:]

p.write_text(s, encoding='utf-8')
print('patched functions')
