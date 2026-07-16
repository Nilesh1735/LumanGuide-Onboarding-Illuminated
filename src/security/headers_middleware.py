"""
Strict transport and header security middleware for the LumanGuide FastAPI
application.

This middleware injects defensive HTTP response headers on every outbound
response to mitigate common web vulnerabilities: clickjacking (X-Frame-Options),
MIME-sniffing (X-Content-Type-Options), transport layer downgrade
(Strict-Transport-Security), cross-site scripting (Content-Security-Policy),
and referrer leakage (Referrer-Policy). It also suppresses the default
Starlette/FastAPI debug error pages when running in production so internal
stack traces are never leaked to clients.

Usage::

    from src.security.headers_middleware import SecurityHeadersMiddleware

    app = FastAPI(...)
    app.add_middleware(SecurityHeadersMiddleware)
"""

from __future__ import annotations

import os
import logging
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# HSTS max-age in seconds. 1 year (31536000) is the recommendation from
# the OWASP cheat sheet for pre-production testing. For production behind
# a load balancer, start with a shorter value (e.g. 300) and increase
# incrementally.
HSTS_MAX_AGE = int(os.getenv("HSTS_MAX_AGE", "31536000"))

# When True, include the "includeSubDomains" directive in the HSTS header.
# Enable only after the entire domain is served over HTTPS; otherwise
# subdomains that are still HTTP-only will become inaccessible.
HSTS_INCLUDE_SUBDOMAINS = os.getenv(
    "HSTS_INCLUDE_SUBDOMAINS", "false"
).strip().lower() == "true"

# When True, include the "preload" directive. This is required for inclusion
# in the browser HSTS preload list (see https://hstspreload.org).
HSTS_PRELOAD = os.getenv("HSTS_PRELOAD", "false").strip().lower() == "true"

# Content-Security-Policy. This baseline policy restricts resource loading to
# the same origin and whitelists the Streamlit WebSocket endpoint. Adjust
# script-src and connect-src to match your deployment.
CSP_POLICY = os.getenv(
    "CSP_POLICY",
    (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "connect-src 'self' ws: wss: http://127.0.0.1:* http://localhost:*; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "object-src 'none'; "
    ),
)

# X-Frame-Options value. "DENY" prevents all framing. "SAMEORIGIN" allows
# framing by the same origin (required if the app embeds itself).
X_FRAME_OPTIONS = os.getenv("X_FRAME_OPTIONS", "DENY").upper()

# Referrer-Policy. "strict-origin-when-cross-origin" sends the origin only
# on cross-origin requests, protecting navigation context.
REFERRER_POLICY = os.getenv("REFERRER_POLICY", "strict-origin-when-cross-origin")

# Additional headers that callers can inject via environment variables.
# Format: "Header-Name: value" (semicolon-separated for multiple).
_EXTRA_HEADERS_RAW = os.getenv("EXTRA_SECURITY_HEADERS", "")


def _build_hsts_value() -> str:
    """Compose the Strict-Transport-Security header value.

    Returns:
        A string of the form ``max-age=...; includeSubDomains; preload``.
    """
    parts = [f"max-age={HSTS_MAX_AGE}"]
    if HSTS_INCLUDE_SUBDOMAINS:
        parts.append("includeSubDomains")
    if HSTS_PRELOAD:
        parts.append("preload")
    return "; ".join(parts)


def _parse_extra_headers() -> dict[str, str]:
    """Parse EXTRA_SECURITY_HEADERS into a name-value mapping.

    Returns:
        A dictionary of header name to header value.
    """
    headers: dict[str, str] = {}
    if not _EXTRA_HEADERS_RAW:
        return headers
    for entry in _EXTRA_HEADERS_RAW.split(";"):
        entry = entry.strip()
        if ":" not in entry:
            continue
        name, _, value = entry.partition(":")
        name = name.strip()
        value = value.strip()
        if name and value:
            headers[name] = value
    return headers


# Pre-compute values at module load so per-request overhead is a single dict
# merge rather than string concatenation.
_HSTS_VALUE = _build_hsts_value()
_EXTRA_HEADERS = _parse_extra_headers()

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that injects security headers on every response.

    When the environment variable ``APP_ENV`` is set to ``production``, the
    middleware also intercepts unhandled 500 errors from Starlette and
    replaces the default HTML error page with a JSON body that does not
    include stack traces or internal details.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)

        # --- Inject security headers on every response. ---
        response.headers["Content-Security-Policy"] = CSP_POLICY
        response.headers["X-Frame-Options"] = X_FRAME_OPTIONS
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = REFERRER_POLICY
        response.headers["Strict-Transport-Security"] = _HSTS_VALUE
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"

        for name, value in _EXTRA_HEADERS.items():
            # Allow callers to override the defaults via env vars.
            response.headers[name] = value

        # --- Suppress debug error pages in production. ---
        is_production = os.getenv("APP_ENV", "development").strip().lower() == "production"
        if (
            is_production
            and response.status_code >= 500
            and getattr(response, "media_type", None) == "text/html"
        ):
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error."},
            )
            # Re-apply headers on the replaced response.
            response.headers["Content-Security-Policy"] = CSP_POLICY
            response.headers["X-Frame-Options"] = X_FRAME_OPTIONS
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = REFERRER_POLICY
            response.headers["Strict-Transport-Security"] = _HSTS_VALUE

        return response


__all__ = ["SecurityHeadersMiddleware"]
