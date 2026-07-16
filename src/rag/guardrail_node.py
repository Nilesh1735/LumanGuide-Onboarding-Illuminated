"""
Prompt injection guardrail for the LumanGuide LangGraph state machine.

This module provides a first-line defence against prompt injection and
query-level injection attempts. It runs as a dedicated LangGraph node
*before* the classifier sees the query, and it also exposes an
async helper for the FastAPI layer so that clearly malicious requests
can be rejected with HTTP 403 without paying the cost of graph
invocation.

Defence layers implemented here:

  1. Prompt-injection phrase detection. A curated, case-insensitive
     pattern list catches the most common jailbreak attempts
     ("ignore previous instructions", "reveal the system prompt", etc.).
  2. Query sanitisation. The user query is normalised and stripped of
     characters that are dangerous in downstream contexts: NoSQL
     operator injection (MongoDB ``$where``, ``$ne``), vector-store
     filter injection, and control-character smuggling.
  3. LangGraph integration. ``guardrail_node`` mutates the workflow
     state so that blocked queries never reach the classifier, instead
     terminating with a safe canned response.

The detection logic is intentionally heuristic and conservative: it
prioritises false-positive avoidance for legitimate business queries
while blocking the well-known attack templates.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Tuple

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Canned response returned when a query is blocked. Kept generic so it
# does not leak which rule fired or hint at the system prompt structure.
_BLOCKED_RESPONSE = (
    "I am unable to process that request. Your query was flagged by the "
    "content security policy. Please rephrase your question without "
    "attempting to override instructions or access internal configuration."
)

# Flag set in the workflow state when the guardrail blocks a query. The
# classifier and downstream nodes check this to short-circuit to END.
_BLOCKED_FLAG = "guardrail_blocked"

# ---------------------------------------------------------------------------
# Prompt-injection detection patterns
# ---------------------------------------------------------------------------

# Each entry is a compiled regex. Patterns are evaluated case-insensitively
# and word-boundary aware where appropriate. The list is derived from
# published prompt-injection corpora and common jailbreak templates.
_INJECTION_PATTERNS: Tuple[re.Pattern[str], ...] = (
    # Direct instruction override attempts.
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(the\s+)?(previous|prior|above)\s+(instructions?|rules?|prompts?)", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|your\s+(previous|prior)\s+instructions?)", re.IGNORECASE),
    re.compile(r"override\s+(your\s+)?(system|safety|content)\s+(prompt|instructions?|rules?)", re.IGNORECASE),
    re.compile(r"(do not|don't|dont)\s+follow\s+(your|the)\s+(rules|instructions)", re.IGNORECASE),

    # System prompt extraction attempts.
    re.compile(r"(reveal|show|print|display|repeat|output|leak|expose)\s+(me\s+)?(your|the)\s+(system\s+)?prompt", re.IGNORECASE),
    re.compile(r"what\s+(is|are)\s+your\s+(system\s+)?(instructions?|prompt|rules?)", re.IGNORECASE),
    re.compile(r"(initial|original|base)\s+(system\s+)?prompt", re.IGNORECASE),

    # Role / identity manipulation.
    re.compile(r"(you\s+are\s+now|act\s+as|pretend\s+(to\s+be|you\s+are)|from\s+now\s+on\s+you\s+are)\s+(a|an)?\s*(DAN|developer|root|admin|jailbreak|unrestricted)", re.IGNORECASE),
    re.compile(r"(enter|enable|activate|switch\s+to)\s+(developer|debug|root|god|jailbreak)\s+mode", re.IGNORECASE),
    re.compile(r"\bDAN\b.*\bjailbreak\b", re.IGNORECASE),

    # Instruction-separator smuggling (attempts to inject fake system messages).
    re.compile(r"<\s*/?\s*system\s*>", re.IGNORECASE),
    re.compile(r"<\s*/?\s*(im_start|im_end)\s*>", re.IGNORECASE),
    re.compile(r"#\s*system\s*(prompt|instruction)", re.IGNORECASE),
    re.compile(r"\[SYSTEM\]|\[INST\]|\[/INST\]", re.IGNORECASE),

    # Token-separator / delimiter injection used to break chat templates.
    re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>", re.IGNORECASE),
    re.compile(r"<<SYS>>|<</SYS>>", re.IGNORECASE),

    # Privilege escalation phrasing toward the agent.
    re.compile(r"(grant|give)\s+(me|yourself)\s+(admin|root|sudo|elevated)\s+(access|privileges?)", re.IGNORECASE),
    re.compile(r"execute\s+(arbitrary|any)\s+(code|command|python)", re.IGNORECASE),
)

# ---------------------------------------------------------------------------
# Query sanitisation
# ---------------------------------------------------------------------------

# NoSQL / MongoDB operator injection. Matches leading-dollar operators
# that, if passed unfiltered into a Mongo query, could alter the query
# semantics (e.g. {"$ne": null} to bypass filters).
_NOSQL_OPERATOR_PATTERN = re.compile(
    r"\$(?:where|ne|gt|gte|lt|lte|in|nin|exists|regex|mod|elemMatch|not|size|all|or|and|nor)\b",
    re.IGNORECASE,
)

# Vector-store filter injection. FAISS / Qdrant metadata filters use JSON
# paths; braces and brackets can alter filter logic.
_FILTER_INJECTION_PATTERN = re.compile(r"[\{\}\[\]]")

# Control characters that could be used for log poisoning or template
# smuggling. Keep \n and \t because legitimate text may contain them.
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Template/format string injection. A query like "{{config}}" or
# "${os.environ}" must not be passed into f-string or .format() contexts
# downstream. We neutralise the delimiters rather than the words.
_TEMPLATE_INJECTION_PATTERN = re.compile(r"\{\{|\}\}|\$\{|%\{|\}")

# Maximum allowed query length. Extremely long queries are a common
# vector for context-window exhaustion and embedding-pipeline abuse.
MAX_QUERY_LENGTH = 2000


def sanitize_query(query: str) -> str:
    """Normalise and sanitise a user query for safe downstream use.

    The query is:

      * Truncated to ``MAX_QUERY_LENGTH`` characters.
      * Stripped of NoSQL operator tokens (``$ne``, ``$where``, ...).
      * Stripped of vector-store filter delimiters (``{ } [ ]``).
      * Stripped of non-printable control characters.
      * Stripped of template/format-string delimiters (``{{ }} ${ }``).

    Args:
        query: The raw user-supplied query string.

    Returns:
        A sanitised string safe to embed into prompts, pass to a vector
        store retriever, or log. The original semantic content is
        preserved as far as possible; only dangerous delimiters and
        operators are removed.
    """
    if not query:
        return ""

    text = query
    if len(text) > MAX_QUERY_LENGTH:
        logger.warning(
            "Query truncated from %d to %d characters (potential context abuse).",
            len(text),
            MAX_QUERY_LENGTH,
        )
        text = text[:MAX_QUERY_LENGTH]

    text = _NOSQL_OPERATOR_PATTERN.sub("", text)
    text = _FILTER_INJECTION_PATTERN.sub("", text)
    text = _CONTROL_CHAR_PATTERN.sub("", text)
    text = _TEMPLATE_INJECTION_PATTERN.sub("", text)
    return text.strip()


def detect_prompt_injection(query: str) -> Tuple[bool, str]:
    """Evaluate a query against prompt-injection detection patterns.

    Args:
        query: The raw user query. Sanitisation is applied internally so
            callers do not need to pre-clean the input.

    Returns:
        A tuple ``(is_injection, reason)``. ``is_injection`` is ``True``
        if any pattern matched. ``reason`` is a short human-readable
        description of the first matched rule, suitable for logging.
    """
    if not query:
        return False, ""

    # Detection runs on the raw query: injection phrases are semantic and
    # must be evaluated before sanitisation strips delimiters.
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(query)
        if match:
            return True, f"matched injection pattern: {pattern.pattern!r}"

    return False, ""


def is_query_safe(query: str) -> Tuple[bool, str]:
    """Combined safety check: injection detection + structural sanitisation.

    This is the canonical pre-flight check used by both the LangGraph
    node and the FastAPI layer. It returns a single verdict so callers
    have one function to call.

    Args:
        query: The raw user query.

    Returns:
        A tuple ``(is_safe, reason)``. ``is_safe`` is ``False`` if the
        query was flagged as a prompt-injection attempt; ``True``
        otherwise.
    """
    if not query:
        return True, ""

    if len(query) > MAX_QUERY_LENGTH:
        return False, (
            f"query exceeds maximum length of {MAX_QUERY_LENGTH} characters"
        )

    is_injection, reason = detect_prompt_injection(query)
    if is_injection:
        return False, reason

    return True, ""


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def guardrail_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node that enforces prompt-injection defences.

    This node is designed to be inserted as the first node after ``START``
    so that every query passes through it before classification. It
    performs three duties:

      1. Detect prompt-injection attempts and, if found, set the
         ``guardrail_blocked`` flag and replace the pending message with
         a safe canned response.
      2. Sanitise the latest query against NoSQL / vector-store / template
         injection regardless of the verdict, so downstream nodes always
         receive clean input.
      3. Return an updated state slice that the conditional edge inspects
         to route blocked queries directly to ``END``.

    The node reads the latest user message from ``state["messages"]`` and
    writes back to ``state["latest_query"]`` and
    ``state["guardrail_blocked"]``.

    Args:
        state: The current LangGraph workflow state. Must contain a
            non-empty ``messages`` list whose last entry is the user
            query.

    Returns:
        A state-update dictionary with keys:
          * ``messages``: a list containing the canned AIMessage if
            blocked, otherwise unchanged.
          * ``latest_query``: the sanitised query.
          * ``guardrail_blocked``: ``"yes"`` if blocked, ``"no"`` otherwise.
    """
    messages = state.get("messages") or []
    if not messages:
        return {
            "messages": messages,
            "latest_query": "",
            _BLOCKED_FLAG: "no",
        }

    raw_query = getattr(messages[-1], "content", str(messages[-1]))
    if isinstance(raw_query, list):
        # LangChain may deliver content as a list of blocks.
        raw_query = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw_query
        )

    is_safe, reason = is_query_safe(raw_query)
    cleaned_query = sanitize_query(raw_query)

    if not is_safe:
        logger.warning(
            "Prompt-injection guardrail blocked a query. Reason: %s. "
            "Query preview: %r",
            reason,
            raw_query[:120],
        )
        return {
            "messages": [AIMessage(content=_BLOCKED_RESPONSE)],
            "latest_query": cleaned_query,
            _BLOCKED_FLAG: "yes",
        }

    return {
        "messages": messages,
        "latest_query": cleaned_query,
        _BLOCKED_FLAG: "no",
    }


def guardrail_router(state: Dict[str, Any]) -> str:
    """Conditional-edge router used after the guardrail node.

    Returns the literal ``"block"`` when the guardrail flagged the query,
    and ``"proceed"`` otherwise. Wire it into the state machine as::

        graph.add_conditional_edges(
            "guardrail",
            guardrail_router,
            {"block": END, "proceed": "evaluate_query_intent"},
        )

    Args:
        state: The workflow state after ``guardrail_node`` has run.

    Returns:
        ``"block"`` or ``"proceed"``.
    """
    if state.get(_BLOCKED_FLAG) == "yes":
        return "block"
    return "proceed"


# ---------------------------------------------------------------------------
# FastAPI integration helper
# ---------------------------------------------------------------------------


async def enforce_guardrail_http(query: str) -> str:
    """Async pre-flight check for the FastAPI layer.

    Returns the sanitised query when the input is safe, or raises an
    ``HTTPException(403)`` when it is flagged as a prompt-injection
    attempt. Designed to be awaited before graph invocation in the
    ``/rag/query`` route.

    Args:
        query: The raw user query from the request body.

    Returns:
        The sanitised query string.

    Raises:
        fastapi.HTTPException: 403 Forbidden when the query is flagged.
    """
    from fastapi import HTTPException

    is_safe, reason = is_query_safe(query)
    if not is_safe:
        logger.warning(
            "HTTP 403 raised by guardrail. Reason: %s. Query preview: %r",
            reason,
            query[:120],
        )
        raise HTTPException(
            status_code=403,
            detail=(
                "Query blocked by content security policy. "
                "Potential prompt-injection attempt detected."
            ),
        )
    return sanitize_query(query)


__all__ = [
    "MAX_QUERY_LENGTH",
    "sanitize_query",
    "detect_prompt_injection",
    "is_query_safe",
    "guardrail_node",
    "guardrail_router",
    "enforce_guardrail_http",
    "_BLOCKED_FLAG",
]
