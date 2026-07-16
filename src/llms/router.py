"""
Resilient LLM router with automatic fallback to a local NVIDIA NIM endpoint.

Primary provider is OpenAI. If a request to the primary fails with a
connection error, timeout, or rate-limit error, the router transparently
retries the request against a self-hosted open-source model (for example,
meta-llama/Llama-3-8B) served through NVIDIA NIM via the
``langchain_nvidia_ai_endpoints`` package.

The router is implemented with LangChain's native ``with_fallbacks``
mechanism, so the returned object remains a fully fledged Runnable. It
therefore supports ``invoke``, ``with_structured_output``, ``bind_tools``
and chain composition (``prompt | llm``), making it a drop-in replacement
for the single-provider ``llm`` exposed by ``src.llms.openai``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def _build_primary_llm():
    """Construct the primary OpenAI-backed chat model.

    Reads credentials from the environment. Raises a descriptive error if
    the required configuration is missing so the caller can decide whether
    to skip the primary provider entirely.

    Returns:
        A configured ``ChatOpenAI`` instance.

    Raises:
        RuntimeError: If ``OPENAI_API_KEY`` is not set in the environment.
    """
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured in the environment.")

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    request_timeout = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "30"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

    logger.info("Primary LLM configured: provider=openai model=%s", model_name)
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        timeout=request_timeout,
        max_retries=max_retries,
    )


def _build_fallback_llm():
    """Construct the NVIDIA NIM fallback chat model.

    The model is expected to be served by a local (or private) NVIDIA NIM
    deployment. Both the base URL of the NIM server and the model identifier
    are configurable through environment variables so this file never hard
    codes infrastructure details.

    Returns:
        A configured ``ChatNVIDIA`` instance.

    Raises:
        RuntimeError: If the NIM base URL is not configured.
    """
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    base_url = os.getenv("NVIDIA_NIM_BASE_URL") or os.getenv("NVIDIA_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "NVIDIA_NIM_BASE_URL is not configured; cannot initialize the "
            "NVIDIA NIM fallback provider."
        )

    api_key = os.getenv("NVIDIA_API_KEY", "local-nim-no-auth")
    model_name = os.getenv("NVIDIA_NIM_MODEL", "meta-llama/Llama-3-8B-Instruct")
    temperature = float(os.getenv("NVIDIA_NIM_TEMPERATURE", "0.2"))
    request_timeout = float(os.getenv("NVIDIA_NIM_TIMEOUT", "60"))

    logger.info("Fallback LLM configured: provider=nvidia_nim model=%s", model_name)
    return ChatNVIDIA(
        base_url=base_url,
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        request_timeout=request_timeout,
    )


def _fallback_exception_types() -> List[type]:
    """Return the exception types that should trigger a fallback.

    These cover the explicit failure modes requested: connection errors,
    timeouts, and rate limits. Generic ``Exception`` is intentionally kept
    as a last resort so that any unexpected provider fault still degrades
    gracefully rather than surfacing a 500 to the caller. Missing optional
    exception classes are skipped silently so import never fails when an
    SDK version differs.
    """
    types: List[type] = []

    candidate_paths = (
        ("openai", "APITimeoutError"),
        ("openai", "APIConnectionError"),
        ("openai", "RateLimitError"),
        ("openai", "APIError"),
        ("httpcore", "ConnectError"),
        ("httpx", "ConnectTimeout"),
        ("httpx", "ReadTimeout"),
    )
    for module_name, attr in candidate_paths:
        try:
            module = __import__(module_name)
            exc = getattr(module, attr, None)
            if isinstance(exc, type) and issubclass(exc, BaseException):
                types.append(exc)
        except Exception:  # noqa: BLE001 - optional dependency, ignore silently
            continue

    types.append(Exception)
    return types


def get_llm() -> Any:
    """Build and return the resilient router LLM.

    The returned object is a ``RunnableWithFallbacks`` built from
    ``primary.with_fallbacks([fallback])``. It preserves the full Runnable
    interface, including ``invoke``, ``with_structured_output``,
    ``bind_tools`` and the ``|`` composition operator.

    If only the primary provider is available, that provider is returned
    directly. If neither is available, ``None`` is returned and the caller
    is expected to fall back to its own retrieval-only / mock path.

    Returns:
        A Runnable chat model with fallback behaviour, the primary model
        alone, or ``None`` if no provider could be configured.
    """
    primary: Optional[Any] = None
    fallback: Optional[Any] = None

    try:
        primary = _build_primary_llm()
    except Exception as exc:  # noqa: BLE001 - configuration/runtime fault
        logger.warning("Primary LLM unavailable: %s", exc)

    try:
        fallback = _build_fallback_llm()
    except Exception as exc:  # noqa: BLE001 - configuration/runtime fault
        logger.warning("NVIDIA NIM fallback unavailable: %s", exc)

    if primary is not None and fallback is not None:
        exceptions = _fallback_exception_types()
        logger.info(
            "LLM router active: primary=openai fallback=nvidia_nim "
            "fallback_exceptions=%d",
            len(exceptions),
        )
        return primary.with_fallbacks(
            fallbacks=[fallback],
            exceptions_to_handle=exceptions,
        )

    if primary is not None:
        logger.info("LLM router active: primary=openai only (no fallback configured).")
        return primary

    if fallback is not None:
        logger.info("LLM router active: nvidia_nim only (primary unavailable).")
        return fallback

    logger.error(
        "LLM router could not configure any provider; returning None. "
        "Set OPENAI_API_KEY and/or NVIDIA_NIM_BASE_URL."
    )
    return None


# Module-level instance for drop-in compatibility with
# ``from src.llms.openai import llm``. Import failures are swallowed so that
# importing this module never crashes the application; callers that need a
# guaranteed handle should call ``get_llm()`` directly.
try:
    llm = get_llm()
except Exception as exc:  # noqa: BLE001 - never let module import fail
    logger.exception("Failed to initialize LLM router at import time: %s", exc)
    llm = None


__all__ = ["get_llm", "llm"]
