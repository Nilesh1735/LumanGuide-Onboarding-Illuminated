"""
Slack notification tool for the agentic workflow.

Exposes ``SlackNotifyTool``, a LangChain ``BaseTool`` that the ReAct agent
can invoke to notify a Subject Matter Expert (SME) on Slack. The tool wraps
the ``slack_sdk`` ``WebClient`` and posts a formatted message to a channel
identified by a Slack channel ID (or, optionally, a channel name).

Design goals:

  * Validated inputs: the channel identifier and message text are validated
    before any network call, so malformed invocations fail fast with a
    descriptive error rather than an opaque Slack API error.
  * Graceful auth handling: Slack auth errors are caught and surfaced as a
    structured tool error string, never as an exception, so the ReAct agent
    can reason about the failure and continue.
  * Configurable transport: the ``client`` parameter lets callers inject a
    pre-configured or mocked ``WebClient``, which keeps the tool testable
    without hitting the Slack API.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

# A Slack channel ID starts with a leading prefix letter (C for channels,
# D for DMs, G for group DMs, etc.) followed by 8+ base32-ish characters.
_CHANNEL_ID_PATTERN = re.compile(r"^[CDG][A-Z0-9]{8,}$")
# A channel name is lowercase, may contain lowercase letters, digits, hyphens
# and underscores, and must start with a letter.
_CHANNEL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,79}$")

_MAX_MESSAGE_LENGTH = 4000  # Slack's documented text limit is 40k; we cap tighter.


class SlackNotifyInput(BaseModel):
    """Validated input schema for ``SlackNotifyTool``.

    Exposing the schema as a Pydantic model gives the ReAct agent a precise
    contract for argument parsing and yields clear validation errors.
    """

    channel: str = Field(
        ...,
        description=(
            "Slack channel identifier. Accepts a channel ID (e.g. C012AB3CD) "
            "or a channel name without the leading # (e.g. eng-onboarding)."
        ),
    )
    message: str = Field(
        ...,
        description="The message text to post to the channel.",
    )
    mention: Optional[str] = Field(
        default=None,
        description=(
            "Optional Slack handle to mention at the start of the message "
            "(e.g. '@nilesh.y')."
        ),
    )

    @field_validator("channel")
    @classmethod
    def _validate_channel(cls, value: str) -> str:
        """Validate that the channel identifier is well-formed.

        Args:
            value: The raw channel identifier supplied by the agent.

        Returns:
            The validated, stripped channel identifier.

        Raises:
            ValueError: If the identifier matches neither a Slack channel ID
                nor a channel name.
        """
        if value is None:
            raise ValueError("channel is required.")
        cleaned = value.strip().lstrip("#")
        if not cleaned:
            raise ValueError("channel must not be empty.")
        if not (
            _CHANNEL_ID_PATTERN.match(cleaned) or _CHANNEL_NAME_PATTERN.match(cleaned)
        ):
            raise ValueError(
                f"Invalid Slack channel identifier: {value!r}. Provide a "
                "channel ID (e.g. C012AB3CD) or a channel name "
                "(e.g. eng-onboarding)."
            )
        return cleaned

    @field_validator("message")
    @classmethod
    def _validate_message(cls, value: str) -> str:
        """Validate that the message is non-empty and within size limits.

        Args:
            value: The raw message text.

        Returns:
            The validated, stripped message text.

        Raises:
            ValueError: If the message is empty or exceeds the size cap.
        """
        if value is None or not value.strip():
            raise ValueError("message must not be empty.")
        if len(value) > _MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"message length {len(value)} exceeds the maximum of "
                f"{_MAX_MESSAGE_LENGTH} characters."
            )
        return value.strip()


class SlackNotifyTool(BaseTool):
    """LangChain tool that posts a formatted message to a Slack channel.

    The token is read from the ``SLACK_BOT_TOKEN`` environment variable by
    default. A custom ``WebClient`` can be injected via ``client`` for
    testing or for sharing a connection pool.
    """

    name: str = "slack_notify_sme"
    description: str = (
        "Use this tool to notify a Subject Matter Expert on Slack. Provide a "
        "channel ID or channel name and the message text to send. Returns a "
        "confirmation string, or a human-readable error description if the "
        "message could not be sent."
    )
    args_schema: Type[BaseModel] = SlackNotifyInput

    # Configuration fields (Pydantic v2 model with arbitrary types allowed so
    # we can store the slack_sdk WebClient instance directly).
    bot_token: Optional[str] = None
    channel_fallback: Optional[str] = None
    client: Any = None

    def __init__(
        self,
        bot_token: Optional[str] = None,
        channel_fallback: Optional[str] = None,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the Slack notification tool.

        Args:
            bot_token: Slack bot token (``xoxb-...``). Defaults to the
                ``SLACK_BOT_TOKEN`` environment variable.
            channel_fallback: Optional default channel used when the agent
                does not supply one. Useful for SME-alert workflows.
            client: Optional pre-configured ``slack_sdk.WebClient``. When
                provided, ``bot_token`` is ignored.
            **kwargs: Forwarded to ``BaseTool``.
        """
        super().__init__(**kwargs)
        resolved_token = bot_token or os.getenv("SLACK_BOT_TOKEN")
        self.bot_token = resolved_token
        self.channel_fallback = channel_fallback or os.getenv("SLACK_DEFAULT_CHANNEL")
        self.client = client or self._build_default_client(resolved_token)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_default_client(token: Optional[str]) -> Any:
        """Construct a ``slack_sdk.WebClient`` if a token is available.

        Args:
            token: The Slack bot token.

        Returns:
            A configured ``WebClient`` or ``None`` if no token is present.
        """
        if not token:
            logger.info(
                "SlackNotifyTool initialised without a bot token; calls will "
                "return a configuration error until SLACK_BOT_TOKEN is set."
            )
            return None
        try:
            from slack_sdk import WebClient
        except Exception as exc:  # noqa: BLE001 - optional dependency
            logger.warning(
                "slack_sdk is not installed; SlackNotifyTool is inactive. "
                "Install with `pip install slack_sdk`. Detail: %s",
                exc,
            )
            return None
        return WebClient(token=token)

    def _format_message(self, message: str, mention: Optional[str]) -> str:
        """Compose the final message text, including an optional mention.

        Args:
            message: The validated message body.
            mention: Optional Slack handle to prepend.

        Returns:
            The formatted message ready for posting.
        """
        prefix = ""
        if mention:
            cleaned_mention = mention.strip()
            if not cleaned_mention.startswith("@"):
                cleaned_mention = "@" + cleaned_mention
            prefix = f"{cleaned_mention} "
        return f"{prefix}{message}"

    # ------------------------------------------------------------------
    # BaseTool contract
    # ------------------------------------------------------------------

    def _run(self, channel: str, message: str, mention: Optional[str] = None) -> str:
        """Synchronous execution path invoked by the ReAct agent.

        Args:
            channel: Slack channel ID or name.
            message: Message body to post.
            mention: Optional Slack handle to prepend.

        Returns:
            A human-readable result string. On failure the string describes
            the error so the agent can reason about it.
        """
        return self._dispatch(channel, message, mention)

    async def _arun(
        self,
        channel: str,
        message: Optional[str] = None,
        mention: Optional[str] = None,
    ) -> str:
        """Asynchronous execution path.

        The ``slack_sdk`` ``WebClient`` is synchronous; we offload the call
        to a worker thread via ``asyncio.to_thread`` so the event loop is
        never blocked.

        Args:
            channel: Slack channel ID or name.
            message: Message body to post.
            mention: Optional Slack handle to prepend.

        Returns:
            A human-readable result string.
        """
        import asyncio

        # ``channel`` is positional and required; ``message`` and ``mention``
        # may arrive as None if the agent omits them.
        return await asyncio.to_thread(self._dispatch, channel, message, mention)

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        channel: Optional[str],
        message: Optional[str],
        mention: Optional[str],
    ) -> str:
        """Validate inputs, post the message, and normalise the result.

        Args:
            channel: Slack channel ID or name (may be ``None`` to fall back
                to ``channel_fallback``).
            message: Message body to post.
            mention: Optional Slack handle to prepend.

        Returns:
            A human-readable result string.
        """
        resolved_channel = channel or self.channel_fallback
        if resolved_channel is None and channel is None:
            return (
                "SlackNotifyTool error: no channel provided and no "
                "SLACK_DEFAULT_CHANNEL fallback is configured."
            )

        try:
            payload = SlackNotifyInput(
                channel=resolved_channel,
                message=message,
                mention=mention,
            )
        except Exception as exc:  # noqa: BLE001 - surface validation cleanly
            return f"SlackNotifyTool validation error: {exc}"

        if self.client is None:
            return (
                "SlackNotifyTool configuration error: SLACK_BOT_TOKEN is not "
                "set or slack_sdk is unavailable. Configure credentials to "
                "enable Slack notifications."
            )

        text = self._format_message(payload.message, payload.mention)
        return self._post_message(payload.channel, text)

    def _post_message(self, channel: str, text: str) -> str:
        """Send the message to Slack and translate the API response.

        Args:
            channel: The validated channel identifier.
            text: The formatted message body.

        Returns:
            A human-readable result string.
        """
        from slack_sdk.errors import SlackApiError, SlackRequestError

        try:
            response = self.client.chat_postMessage(channel=channel, text=text)
        except SlackApiError as exc:
            return self._describe_api_error(exc)
        except SlackRequestError as exc:
            logger.warning("Slack request error: %s", exc)
            return f"SlackNotifyTool request error: {exc}"
        except Exception as exc:  # noqa: BLE001 - never raise to the agent
            logger.exception("Unexpected Slack error")
            return f"SlackNotifyTool unexpected error: {exc}"

        if not response.get("ok", False):
            return (
                "SlackNotifyTool error: Slack API returned ok=false. "
                f"Response: {response}"
            )

        channel_ref = (
            response.get("channel")
            or response.get("message", {}).get("channel")
            or channel
        )
        ts = response.get("ts", "n/a")
        logger.info(
            "SlackNotifyTool posted message to channel=%s ts=%s",
            channel_ref,
            ts,
        )
        return (
            f"Slack notification sent to channel {channel_ref} "
            f"(message ts={ts})."
        )

    @staticmethod
    def _describe_api_error(exc: Any) -> str:
        """Translate a ``SlackApiError`` into a user-facing message.

        Auth-related errors are detected from the Slack error code and
        described with actionable guidance.

        Args:
            exc: The ``SlackApiError`` raised by the WebClient.

        Returns:
            A human-readable description of the failure.
        """
        response = getattr(exc, "response", None) or {}
        api_error = {}
        try:
            api_error = response.get("data", {}) or response.get("error", {})
        except Exception:  # noqa: BLE001 - defensive
            api_error = {}

        error_code = ""
        error_message = ""
        if isinstance(api_error, dict):
            error_code = str(api_error.get("error", ""))
            error_message = str(api_error.get("error", ""))

        if not error_code:
            error_code = str(getattr(exc, "message", "") or "unknown_error")

        auth_codes = {"invalid_auth", "not_authed", "account_inactive", "token_revoked"}
        if error_code in auth_codes:
            return (
                "SlackNotifyTool authentication error: the bot token is "
                f"invalid or revoked ({error_code}). Verify SLACK_BOT_TOKEN."
            )
        if error_code == "channel_not_found":
            return (
                f"SlackNotifyTool error: channel not found ({channel!s}). "
                "Ensure the bot is a member of the target channel."
            ).replace("{channel!s}", "")
        return f"SlackNotifyTool API error: {error_message or error_code}."


def get_slack_notify_tool(
    bot_token: Optional[str] = None,
    channel_fallback: Optional[str] = None,
    client: Any = None,
) -> SlackNotifyTool:
    """Factory returning a configured ``SlackNotifyTool`` instance.

    Args:
        bot_token: Optional Slack bot token override.
        channel_fallback: Optional default channel override.
        client: Optional pre-configured ``WebClient``.

    Returns:
        A configured ``SlackNotifyTool``.
    """
    return SlackNotifyTool(
        bot_token=bot_token,
        channel_fallback=channel_fallback,
        client=client,
    )


__all__ = [
    "SlackNotifyTool",
    "SlackNotifyInput",
    "get_slack_notify_tool",
]
