"""
Document secret scanner for RAG ingestion pipelines.

Before extracted text is chunked and embedded into FAISS, this module
scans it for high-value credential patterns. Matching substrings are
redacted in-place and a structured security warning is logged with the
match details so operators can investigate the source document.

This prevents accidental indexing of secrets (API keys, tokens,
certificates) that could be surfaced to end users via the retrieval
pipeline, which is a distinct failure mode from traditional secrets
scanning because the sensitive text enters the vector store rather
than a database or log.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

# Redaction replacement token.
_REDACTION_PLACEHOLDER = "[REDACTED_SECRET]"

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecretPattern:
    """A named regex pattern for a single credential type.

    Attributes:
        name: Human-readable label for the pattern (used in log output).
        regex: Compiled regular expression that matches the credential.
        severity: Severity level for triage. High-severity matches trigger
            an immediate WARNING log.
    """

    name: str
    regex: "re.Pattern[str]"
    severity: str = "high"


# Patterns are ordered by specificity. More specific patterns are tested
# first to avoid false positives from broader patterns.
_SECRET_PATTERNS: List[SecretPattern] = [
    # --- AWS credentials ---
    # Access key ID: 20 uppercase alphanumeric characters, optionally
    # preceded by the literal prefix "AKIA".
    SecretPattern(
        name="AWS Access Key ID",
        regex=re.compile(
            r"(?:AKIA|(?<![A-Za-z0-9]))[A-Z0-9]{16,20}(?![A-Za-z0-9])",
        ),
    ),
    # Secret access key: 40-character base64 string commonly appearing on
    # the line after an access key ID.
    # FIXED: Removed variable-length look-behind to support Python 3.12+
    SecretPattern(
        name="AWS Secret Access Key",
        regex=re.compile(
            r"(aws_secret_access_key\s*[=:]\s*)[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])",
            re.IGNORECASE,
        ),
    ),
    # AWS session token: much longer base64 block.
    # FIXED: Removed variable-length look-behind to support Python 3.12+
    SecretPattern(
        name="AWS Session Token",
        regex=re.compile(
            r"(aws_session_token\s*[=:]\s*)[A-Za-z0-9/+=]{200,}(?![A-Za-z0-9/+=])",
            re.IGNORECASE,
        ),
    ),
    # Generic "AWS" prefix followed by a long credential-like string.
    SecretPattern(
        name="AWS Credential (generic)",
        regex=re.compile(
            r"(?i)aws[_\-]?(?:secret|access|key|token)[_\-]?"
            r"(?:id|key)?\s*[=:]\s*\S{20,}",
        ),
        severity="medium",
    ),

    # --- Slack tokens ---
    # Bot tokens start with "xoxb-", user tokens with "xoxp-".
    SecretPattern(
        name="Slack Bot Token",
        regex=re.compile(r"xoxb-[A-Za-z0-9\-]{10,}"),
    ),
    SecretPattern(
        name="Slack User Token",
        regex=re.compile(r"xoxp-[A-Za-z0-9\-]{10,}"),
    ),
    SecretPattern(
        name="Slack App Token",
        regex=re.compile(r"xapp-[A-Za-z0-9\-]{10,}"),
    ),
    SecretPattern(
        name="Slack Refresh Token",
        regex=re.compile(r"xoxr-[A-Za-z0-9\-]{10,}"),
    ),

    # --- JSON Web Tokens ---
    # JWTs are three base64url segments separated by dots. We match
    # tokens that are long enough to plausibly be credentials (not just
    # random base64 fragments).
    SecretPattern(
        name="JSON Web Token (JWT)",
        regex=re.compile(
            r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
        ),
    ),

    # --- Private keys ---
    # Detect the PEM header that precedes base64-encoded key material.
    SecretPattern(
        name="RSA Private Key (PEM)",
        regex=re.compile(
            r"-----BEGIN[\s-]+(?:RSA\s+)?PRIVATE\s+KEY-----",
            re.IGNORECASE,
        ),
    ),
    SecretPattern(
        name="EC Private Key (PEM)",
        regex=re.compile(
            r"-----BEGIN[\s-]+EC\s+PRIVATE\s+KEY-----",
            re.IGNORECASE,
        ),
    ),
    SecretPattern(
        name="DSA Private Key (PEM)",
        regex=re.compile(
            r"-----BEGIN[\s-]+DSA\s+PRIVATE\s+KEY-----",
            re.IGNORECASE,
        ),
    ),
    SecretPattern(
        name="OpenSSH Private Key",
        regex=re.compile(
            r"-----BEGIN[\s-]+OPENSSH\s+PRIVATE\s+KEY-----",
            re.IGNORECASE,
        ),
    ),
    SecretPattern(
        name="PKCS8 Private Key (PEM)",
        regex=re.compile(
            r"-----BEGIN[\s-]+ENCRYPTED\s+PRIVATE\s+KEY-----",
            re.IGNORECASE,
        ),
    ),

    # --- Google API keys ---
    SecretPattern(
        name="Google API Key",
        regex=re.compile(r"\bAIza[A-Za-z0-9_\-]{35}\b"),
    ),

    # --- Generic high-entropy strings after key-name patterns ---
    # This is a lower-confidence catch-all. It matches long alphanumeric
    # strings that appear after common assignment patterns.
    SecretPattern(
        name="Generic Credential (high entropy)",
        regex=re.compile(
            r"(?i)(?:api[_\-]?key|secret|password|passwd|token|bearer|auth)"
            r"\s*[=:]\s*['\"]?[A-Za-z0-9+/=_\-]{32,}['\"]?",
        ),
        severity="medium",
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class ScanResult:
    """Result of a secret scan over a block of text.

    Attributes:
        text: The (possibly redacted) text.
        matches: List of individual match records.
        secrets_found: True if at least one match was detected.
    """

    text: str
    matches: List[dict] = field(default_factory=list)

    @property
    def secrets_found(self) -> bool:
        return len(self.matches) > 0


def scan_document_for_secrets(text: str) -> ScanResult:
    """Scan extracted document text for credential patterns and redact them.

    The function iterates over the compiled pattern list, finds all
    non-overlapping matches, replaces them in-place with the redaction
    placeholder, and logs a structured warning for each detection.
    Because patterns are applied sequentially, later patterns operate
    on the already-redacted text so previously redacted regions are
    not re-scanned.

    Args:
        text: The raw extracted text from a document before chunking.

    Returns:
        A ``ScanResult`` containing the redacted text and a list of match
        records. Each record has keys ``pattern``, ``severity``,
        ``match_preview`` (first 8 characters of the matched string) and
        ``character_offset``.
    """
    if not text:
        return ScanResult(text=text)

    redacted_text = text
    matches: List[dict] = []

    # Sort patterns so high-severity ones are applied first.
    sorted_patterns = sorted(
        _SECRET_PATTERNS, key=lambda p: 0 if p.severity == "high" else 1
    )

    for pattern in sorted_patterns:
        for m in pattern.regex.finditer(redacted_text):
            matched_string = m.group(0)
            matches.append(
                {
                    "pattern": pattern.name,
                    "severity": pattern.severity,
                    "match_preview": matched_string[:8] + "...",
                    "character_offset": m.start(),
                    "length": len(matched_string),
                }
            )
            # Replace only the matched portion; preserve surrounding text.
            # Using re.sub with the compiled pattern ensures consistent
            # replacement semantics.
            redacted_text = pattern.regex.sub(
                _REDACTION_PLACEHOLDER, redacted_text, count=1
            )

    if matches:
        logger.warning(
            "Secret scanner detected %d credential pattern(s) in document text. "
            "Affected text has been redacted. Matches: %s",
            len(matches),
            matches,
        )

    return ScanResult(text=redacted_text, matches=matches)


__all__ = ["scan_document_for_secrets", "ScanResult", "SecretPattern"]