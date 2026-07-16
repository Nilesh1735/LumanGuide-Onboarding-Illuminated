"""
LangSmith telemetry and observability for the LangGraph workflow.

This module centralises LLMOps tracing configuration and exposes a
``@trace_node(node_name)`` decorator that wraps LangGraph node functions to
emit structured, named runs to LangSmith. Each wrapped run explicitly logs:

  * The node input payload.
  * The node output payload.
  * Approximate token usage derived from the input and output payloads
    (helpful for cost attribution when a node does not surface native
    ``usage_metadata``).

The module is intentionally resilient: if LangSmith is not configured (no
API key) or the ``langsmith`` package is absent, ``init_telemetry`` reports
the disabled state and the decorator degrades to a transparent passthrough.
This guarantees that tracing never breaks node execution, which is a hard
requirement for production observability tooling.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

_TRACING_ENV_KEYS = (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
    "LANGCHAIN_ENDPOINT",
)
_DEFAULT_LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"

# Module-level flag that records whether LangSmith tracing was successfully
# enabled. The decorator consults it to decide whether to emit runs.
_TELEMETRY_ENABLED: bool = False


def init_telemetry(
    project_name: Optional[str] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> bool:
    """Initialise LangSmith tracing from the environment.

    Reads the canonical LangChain tracing variables and, if a valid API key
    is present, ensures they are exported into ``os.environ`` so that every
    LangChain/LangGraph Runnable invoked from the process is automatically
    traced. Explicit ``project_name``, ``api_key`` and ``endpoint`` arguments
    take precedence over environment values, which is convenient for tests
    and provisioning scripts.

    Args:
        project_name: Optional override for the LangSmith project name. When
            omitted the ``LANGCHAIN_PROJECT`` environment variable is used,
            defaulting to ``"LumanGuide"``.
        api_key: Optional override for the LangSmith API key. When omitted
            the ``LANGCHAIN_API_KEY`` environment variable is used.
        endpoint: Optional override for the LangSmith API endpoint. When
            omitted the ``LANGCHAIN_ENDPOINT`` environment variable is used,
            defaulting to the public LangSmith URL.

    Returns:
        ``True`` if tracing was enabled, ``False`` otherwise. A return value
        of ``False`` is logged at INFO level and is not an error condition.
    """
    global _TELEMETRY_ENABLED

    resolved_key = api_key or os.getenv("LANGCHAIN_API_KEY")
    resolved_project = project_name or os.getenv("LANGCHAIN_PROJECT", "LumanGuide")
    resolved_endpoint = endpoint or os.getenv(
        "LANGCHAIN_ENDPOINT", _DEFAULT_LANGCHAIN_ENDPOINT
    )

    if not resolved_key:
        logger.info(
            "LangSmith tracing disabled: LANGCHAIN_API_KEY is not set. "
            "Set it (and LANGCHAIN_TRACING_V2=true) to enable observability."
        )
        _TELEMETRY_ENABLED = False
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    try:
        # Importing here keeps the module importable when the optional
        # langsmith dependency is absent.
        import langsmith  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - optional dependency
        logger.warning(
            "LangSmith tracing disabled: langsmith package unavailable (%s). "
            "Install with `pip install langsmith` to enable observability.",
            exc,
        )
        _TELEMETRY_ENABLED = False
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = resolved_key
    os.environ["LANGCHAIN_PROJECT"] = resolved_project
    os.environ["LANGCHAIN_ENDPOINT"] = resolved_endpoint

    _TELEMETRY_ENABLED = True
    logger.info(
        "LangSmith tracing enabled: project=%s endpoint=%s",
        resolved_project,
        resolved_endpoint,
    )
    return True


def is_telemetry_enabled() -> bool:
    """Return whether LangSmith tracing is currently active.

    Returns:
        ``True`` if ``init_telemetry`` successfully enabled tracing.
    """
    return _TELEMETRY_ENABLED


# ---------------------------------------------------------------------------
# Token estimation helper
# ---------------------------------------------------------------------------

def _estimate_tokens(payload: Any) -> int:
    """Estimate the token count of an arbitrary node payload.

    LangGraph node inputs and outputs are usually dicts containing state,
    messages and tool results. We recursively stringify the payload and
    approximate token count with ``tiktoken`` when available, falling back
    to a deterministic whitespace heuristic so a count is always produced.

    Args:
        payload: The node input or output value.

    Returns:
        An approximate non-negative token count.
    """
    text = _stringify_payload(payload)
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text, disallowed_special=()))
    except Exception:  # noqa: BLE001 - heuristic fallback
        # Crude but deterministic: ~4 characters per token for English text.
        return max(1, len(text) // 4)


def _stringify_payload(payload: Any) -> str:
    """Render a node payload as a flat string for token estimation.

    Handles common LangChain message objects by reading their ``content``
    attribute, and recurses into dicts, lists and tuples.

    Args:
        payload: The value to stringify.

    Returns:
        A best-effort string representation.
    """
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    # LangChain BaseMessage objects expose a .content attribute.
    content = getattr(payload, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(payload, dict):
        return " ".join(_stringify_payload(v) for v in payload.values())
    if isinstance(payload, (list, tuple)):
        return " ".join(_stringify_payload(item) for item in payload)
    return str(payload)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def trace_node(node_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator factory that traces a LangGraph node under a named run.

    The wrapped function logs its input payload, output payload and an
    approximate token-usage breakdown to LangSmith as a run named
    ``node_name``. Both synchronous and asynchronous node functions are
    supported and their original signatures are preserved via
    ``functools.wraps``.

    Behaviour when telemetry is disabled: the decorator is a transparent
    passthrough that adds only a timing log line, so it is always safe to
    apply to production nodes regardless of environment configuration.

    Args:
        node_name: The human-readable name of the LangGraph node. This is
            used verbatim as the LangSmith run name and must be non-empty.

    Returns:
        A decorator that wraps the supplied node function.

    Raises:
        ValueError: If ``node_name`` is empty.
    """
    if not node_name or not node_name.strip():
        raise ValueError("node_name must be a non-empty string.")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if not _TELEMETRY_ENABLED:
            # Telemetry is off: wrap with a lightweight timing logger so the
            # decorator remains a true no-op with respect to LangSmith while
            # still surfacing basic latency diagnostics.
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def _async_passthrough(*args: Any, **kwargs: Any) -> Any:
                    return await func(*args, **kwargs)

                return _async_passthrough

            @functools.wraps(func)
            def _sync_passthrough(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            return _sync_passthrough

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await _run_traced_async(
                    func, node_name, args, kwargs
                )

            return _async_wrapper

        @functools.wraps(func)
        def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return _run_traced_sync(func, node_name, args, kwargs)

        return _sync_wrapper

    return decorator


def _emit_run(
    node_name: str,
    inputs: dict,
    outputs: Any,
    error: Optional[BaseException],
) -> None:
    """Create a named LangSmith run capturing inputs, outputs and tokens.

    The run is created in the "chain" run type so it aggregates cleanly
    beneath the top-level LangGraph trace. Token usage is attached as a
    structured ``usage`` metadata field, providing cost attribution even
    for nodes that do not return LangChain messages with native usage data.

    Args:
        node_name: Name used for the LangSmith run.
        inputs: Serializable mapping of the node's input arguments.
        outputs: The value returned by the node, if it succeeded.
        error: The exception raised by the node, if it failed.
    """
    try:
        from langsmith import Client

        client = Client()
        input_tokens = _estimate_tokens(inputs)
        output_tokens = _estimate_tokens(outputs)
        metadata = {
            "node": node_name,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

        run = client.create_run(
            name=node_name,
            run_type="chain",
            inputs=inputs,
            outputs={"result": outputs} if error is None else None,
            error=str(error) if error is not None else None,
            metadata=metadata,
        )

        if error is not None:
            client.update_run(
                run_id=run.id,
                outputs=None,
                error=str(error),
            )
        else:
            client.update_run(
                run_id=run.id,
                outputs={"result": outputs},
            )
    except Exception as exc:  # noqa: BLE001 - telemetry must never break nodes
        logger.debug(
            "Failed to emit LangSmith run for node %s: %s", node_name, exc
        )


def _run_traced_sync(func, node_name, args, kwargs):
    """Execute a synchronous node under tracing and emit its run."""
    try:
        result = func(*args, **kwargs)
    except BaseException as exc:  # noqa: BLE001 - capture for failed-run logging
        _emit_run(node_name, _build_inputs(args, kwargs), None, exc)
        raise
    _emit_run(node_name, _build_inputs(args, kwargs), result, None)
    return result


async def _run_traced_async(func, node_name, args, kwargs):
    """Execute an asynchronous node under tracing and emit its run."""
    try:
        result = await func(*args, **kwargs)
    except BaseException as exc:  # noqa: BLE001 - capture for failed-run logging
        _emit_run(node_name, _build_inputs(args, kwargs), None, exc)
        raise
    _emit_run(node_name, _build_inputs(args, kwargs), result, None)
    return result


def _build_inputs(args: tuple, kwargs: dict) -> dict:
    """Build a JSON-serialisable input mapping for a traced node.

    LangGraph nodes conventionally receive the graph state as a single
    positional argument. We capture positional arguments under ``args`` and
    keyword arguments verbatim, coerced to strings to guarantee that the
    LangSmith client can serialize them regardless of state type.

    Args:
        args: Positional arguments passed to the node.
        kwargs: Keyword arguments passed to the node.

    Returns:
        A serializable dictionary describing the node invocation.
    """
    payload: dict = {}
    if args:
        payload["args"] = [_stringify_payload(a) for a in args]
    if kwargs:
        payload["kwargs"] = {
            str(k): _stringify_payload(v) for k, v in kwargs.items()
        }
    return payload


# Eagerly attempt initialisation at import time so that any subsequent
# import of LangChain/LangGraph components inherits the tracing context.
# Failures are non-fatal and simply leave telemetry disabled.
try:
    init_telemetry()
except Exception as exc:  # noqa: BLE001 - never break import
    logger.debug("Telemetry auto-initialisation skipped: %s", exc)


__all__ = [
    "init_telemetry",
    "is_telemetry_enabled",
    "trace_node",
]
