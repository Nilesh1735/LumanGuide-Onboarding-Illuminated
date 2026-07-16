import logging
import os
import streamlit as st
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Safely resolve the backend URL.
# 1. Checks Streamlit Cloud Secrets (for production)
# 2. Checks local environment variables (for local dev)
# 3. Falls back to localhost:8000
try:
    BASE_URL = st.secrets.get("BACKEND_URL", os.getenv("BACKEND_URL", "http://127.0.0.1:8000"))
except Exception:
    # Fallback if running outside of a Streamlit context or no secrets file exists
    BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

BASE_URL = BASE_URL.rstrip("/")


def _candidate_base_urls() -> list[str]:
    """Return the resolved backend URL.
    
    We no longer hardcode localhost ports here because on Streamlit Cloud, 
    trying localhost causes 5-10 second timeouts on every request.
    """
    return [BASE_URL]


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
    Try the mounted /api route first, then the direct route.
    Uses the resolved BASE_URL.
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
    """Authenticate user login and return a structured result for the UI."""
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
    """Send a query to the RAG backend and return diagnostic info for the UI."""
    headers = {"Authorization": f"Bearer {jwt_token}", "Content-Type": "application/json"}
    try:
        response = _post_with_fallback(
            "rag/query",
            {"query": query, "session_id": session_id},
            headers,
            timeout=180.0
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
    """Fetch preview list of persisted documents for the UI document picker."""
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
    """Fetch the Team Navigator status (whether team data is loaded + member list)."""
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