import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

# BACKEND_URL can override this. The README starts FastAPI on port 8000.
BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def _candidate_base_urls() -> list[str]:
    """Return candidate base URLs in preferred order.

    Preference: try local default port 8000 first, then any BACKEND_URL provided
    via env, then 8080 as a last fallback. This favors the default development
    server port while still allowing overrides.
    """
    urls: list[str] = []
    # Always try 8000 first
    default_8000 = "http://127.0.0.1:8000"
    if default_8000 not in urls:
        urls.append(default_8000)

    # Then include configured BASE_URL if it's different
    if BASE_URL not in urls:
        urls.append(BASE_URL)

    # Finally, try 8080 as a legacy fallback
    legacy_8080 = "http://127.0.0.1:8080"
    if legacy_8080 not in urls:
        urls.append(legacy_8080)

    return urls


def _parse_response(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def _error_message(response: requests.Response) -> str:
    payload = _parse_response(response)
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("message") or payload
    else:
        detail = payload
    return f"{response.status_code} at {response.url}: {detail}"


def _post_with_fallback(endpoint: str, json_data: dict, headers: dict, timeout: float = 180.0) -> requests.Response:
    """
    Try the mounted /api route first, then the direct route for older servers.
    Also checks ports 8000 and 8080 because this project has used both.

    If all reachable endpoints return 404, raise an HTTPError that includes the
    list of attempted URLs and their status codes. This makes it easier to
    diagnose misconfigured ports/prefixes from the UI while preserving the
    normal successful behavior.
    """
    cleaned_endpoint = endpoint.lstrip("/")
    urls_to_try = []
    for base_url in _candidate_base_urls():
        urls_to_try.extend(
            [
                f"{base_url}/api/{cleaned_endpoint}",
                f"{base_url}/{cleaned_endpoint}",
            ]
        )

    last_response = None
    connection_errors = []
    attempts = []  # collect (url, status_or_error)

    for url in urls_to_try:
        try:
            response = requests.post(url, json=json_data, headers=headers, timeout=timeout)
            attempts.append((url, response.status_code))
            if response.status_code != 404:
                return response
            last_response = response
        except requests.RequestException as exc:
            connection_errors.append(f"{url}: {exc}")
            attempts.append((url, str(exc)))

    # If at least one URL responded (but all with 404), raise a helpful HTTPError
    if last_response is not None:
        attempt_lines = [f"{u} -> {s}" for u, s in attempts]
        details = (
            "All backend endpoints returned 404 (Not Found). Attempts:\n" + "\n".join(attempt_lines)
        )
        raise requests.HTTPError(details)

    # No responses at all; raise a connection error with aggregated exceptions
    details = "; ".join(connection_errors) or "no backend URLs were reachable"
    raise requests.ConnectionError(details)


def get_api_token() -> str:
    """Return a placeholder token; the current backend auth routes do not require one."""
    return "mock_initial_token"


def create_user(username: str, password: str, api_token: str) -> dict:
    """Create a new user account and return a structured result for the UI."""
    headers = {"X-API-TOKEN": api_token, "Content-Type": "application/json"}
    try:
        response = _post_with_fallback(
            "signup",
            {"username": username, "password": password},
            headers,
        )
        if response.status_code == 200:
            logger.info("User created successfully.")
            return {"ok": True, "data": _parse_response(response)}

        return {"ok": False, "error": _error_message(response)}
    except Exception as exc:
        return {"ok": False, "error": f"Connection failure: {exc}"}


def login_user(username: str, password: str, api_token: str) -> dict:
    """Authenticate user login and return a structured result for the UI.

    Returns a dict with keys:
      - ok: bool
      - data: dict (on success)
      - error: str (on failure)
      - attempts: list[str] (optional diagnostic lines when fallback URLs were tried)
    """
    headers = {"X-API-TOKEN": api_token, "Content-Type": "application/json"}
    try:
        response = _post_with_fallback(
            "login",
            {"username": username, "password": password},
            headers,
        )
        if response.status_code == 200:
            return {"ok": True, "data": _parse_response(response)}

        return {"ok": False, "error": _error_message(response)}
    except requests.HTTPError as http_err:
        msg = str(http_err)
        attempts: list[str] = []
        if "Attempts:\n" in msg:
            try:
                parts = msg.split("Attempts:\n", 1)[1]
                attempts = [line.strip() for line in parts.splitlines() if line.strip()]
            except Exception:
                attempts = []
        return {"ok": False, "error": msg, "attempts": attempts}
    except requests.ConnectionError as conn_err:
        return {"ok": False, "error": str(conn_err), "attempts": []}
    except Exception as exc:
        return {"ok": False, "error": f"Connection failure: {exc}", "attempts": []}


def query_backend(query: str, session_id: str, jwt_token: str, openai_api_key: str = None):
    """Send a query to the RAG backend."""
    headers = {"Authorization": f"Bearer {jwt_token}", "Content-Type": "application/json"}
    
    # Pass the user's OpenAI key if provided
    if openai_api_key:
        headers["X-OpenAI-Key"] = openai_api_key
        
    try:
        response = _post_with_fallback(
            "rag/query",
            {"query": query, "session_id": session_id},
            headers,
            timeout=180.0
        )
        if response.status_code == 200:
            payload = _parse_response(response)
            return payload.get("result", {}).get("content", "No content found.")
        return f"Error: {_error_message(response)}"
    except Exception as exc:
        logger.error("Query request failed: %s", exc)
        return f"Connection error: {exc}"


def query_backend_diagnostic(query: str, session_id: str, jwt_token: str):
    """Send a query to the RAG backend and return diagnostic info for the UI.

    Returns a dict with keys:
      - ok: bool
      - content: str (when ok)
      - error: str (when not ok)
      - attempts: list[str] (optional diagnostic lines)
    """
    headers = {"Authorization": f"******", "Content-Type": "application/json"}
    try:
        response = _post_with_fallback(
            "rag/query",
            {"query": query, "session_id": session_id},
            headers,
            timeout=180.0  # Increased timeout to 180s
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


def get_persisted_docs():
    """Fetch preview list of persisted documents for the UI document picker.
    Returns a dict: {"ok": True, "documents": [ {index, snippet, metadata}, ... ]} or error dict.
    """
    for base_url in _candidate_base_urls():
        for path in ("api/rag/persisted_docs", "rag/persisted_docs"):
            try:
                r = requests.get(f"{base_url}/{path}", timeout=10)
                if r.status_code != 404:
                    if r.status_code == 200:
                        try:
                            return {"ok": True, "documents": _parse_response(r).get("documents", [])}
                        except Exception:
                            return {"ok": False, "error": "Failed to parse persisted docs response"}
                    else:
                        return {"ok": False, "error": _error_message(r)}
            except requests.RequestException as exc:
                continue
    return {"ok": False, "error": "No persisted-docs endpoint was reachable."}


def get_team_status():
    """Fetch the Team Navigator status (whether team data is loaded + member list).

    Returns a dict with keys:
      - ok: bool
      - navigator_loaded: bool
      - member_count: int
      - members: list[str]
    """
    for base_url in _candidate_base_urls():
        for path in ("api/rag/team/status", "rag/team/status"):
            try:
                r = requests.get(f"{base_url}/{path}", timeout=10)
                if r.status_code != 404:
                    if r.status_code == 200:
                        try:
                            data = _parse_response(r)
                            return {
                                "ok": True,
                                "navigator_loaded": data.get("navigator_loaded", False),
                                "team_doc_count": data.get("team_doc_count", 0),
                                "member_count": data.get("member_count", 0),
                                "members": data.get("members", []),
                            }
                        except Exception:
                            return {"ok": False, "navigator_loaded": False, "members": []}
                    else:
                        return {"ok": False, "navigator_loaded": False, "members": []}
            except requests.RequestException:
                continue
    return {"ok": False, "navigator_loaded": False, "members": []}

def document_upload_rag(file, description: str, jwt_token: str) -> dict:
    """Upload a document to the RAG system."""
    headers = {"X-Description": description, "Authorization": f"Bearer {jwt_token}"}
    files = {"file": (file.name, file, file.type)}

    for base_url in _candidate_base_urls():
        for path in ("api/rag/documents/upload", "rag/documents/upload"):
            try:
                file.seek(0)
                response = requests.post(
                    f"{base_url}/{path}",
                    files=files,
                    headers=headers,
                    timeout=120,  # Increased timeout for large embeddings
                )
                if response.status_code != 404:
                    if response.status_code == 200:
                        try:
                            payload = _parse_response(response)
                            # Be very permissive with the success check
                            if isinstance(payload, dict) and payload.get("status") is not None:
                                return {"ok": bool(payload.get("status")), "data": payload}
                            elif isinstance(payload, dict) and payload.get("message"):
                                return {"ok": True, "data": payload}
                            else:
                                # If status is missing but 200 OK, assume success
                                return {"ok": True, "data": {"message": "Upload successful"}}
                        except ValueError:
                            # JSON parse error, but 200 OK
                            return {"ok": True, "data": {"message": "Upload successful"}}
                    else:
                        return {"ok": False, "error": _error_message(response)}
            except requests.RequestException as exc:
                logger.error("Upload failed for %s/%s: %s", base_url, path, exc)

    return {"ok": False, "error": "No upload endpoint was reachable."}