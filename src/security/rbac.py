"""
Role-Based Access Control (RBAC) for the LumanGuide FastAPI application.

This module provides JWT-based authentication and authorisation via a
``verify_role(required_role)`` dependency factory. It is designed as a
drop-in upgrade for the current in-memory ``user_db`` auth module, which
stores plaintext passwords and issues no tokens. Here we introduce:

  * Password hashing with bcrypt via ``passlib``.
  * JWT issuance and verification using ``PyJWT`` and a symmetric secret.
  * A role hierarchy (admin > contributor > viewer) enforced through a
    FastAPI ``Depends`` callable.
  * Optional bootstrap of a default admin account from environment
    variables so the system is usable on first launch.

All public functions are async-compatible so they can be used directly as
FastAPI dependencies or awaited from async route handlers.
"""

from __future__ import annotations

import logging
import os
import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Symmetric secret used to sign JWTs. MUST be set in production. The
# fallback is intentionally non-functional so a missing secret fails loud.
JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ISSUER = os.getenv("JWT_ISSUER", "lumanguide")
# Access-token lifetime in minutes.
ACCESS_TOKEN_TTL_MINUTES = int(os.getenv("JWT_TTL_MINUTES", "60"))

# Role hierarchy. A user with a higher role implicitly satisfies any
# ``required_role`` that is lower in the hierarchy. Order matters: index 0
# is the most privileged.
_ROLE_HIERARCHY: List[str] = ["admin", "contributor", "viewer"]
_ROLE_RANK: Dict[str, int] = {role: idx for idx, role in enumerate(_ROLE_HIERARCHY)}

# OAuth2 scheme. tokenUrl points at the login endpoint so Swagger UI can
# perform the interactive OAuth2 flow. auto_error=False lets us return a
# custom 401 instead of the scheme's default.
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="api/login",
    auto_error=False,
)

# Optional in-process directory of users keyed by username. This mirrors
# the existing ``user_db`` shape but adds ``hashed_password`` and ``role``.
# In production this should be replaced by a persistent store (MongoDB).
_user_store: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def _get_password_hasher():
    """Return a configured bcrypt password-hashing context.

    The import is deferred so the module remains importable in
    environments where ``passlib`` is not yet installed.
    """
    from passlib.context import CryptContext

    return CryptContext(schemes=["bcrypt"], deprecated="auto")


_pwd_context = None


def hash_password(plain: str) -> str:
    """Hash a plaintext password using bcrypt.

    Args:
        plain: The plaintext password.

    Returns:
        A bcrypt hash string.
    """
    global _pwd_context
    if _pwd_context is None:
        _pwd_context = _get_password_hasher()
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash.

    Args:
        plain: The plaintext password supplied at login.
        hashed: The stored bcrypt hash.

    Returns:
        ``True`` if the password matches the hash.
    """
    global _pwd_context
    if _pwd_context is None:
        _pwd_context = _get_password_hasher()
    try:
        return _pwd_context.verify(plain, hashed)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# JWT issuance / verification
# ---------------------------------------------------------------------------


def _require_secret() -> str:
    """Return the configured JWT secret or raise.

    Raises:
        RuntimeError: If ``JWT_SECRET`` is not configured.
    """
    if not JWT_SECRET:
        raise RuntimeError(
            "JWT_SECRET is not set. Generate a strong secret (e.g. "
            "`python -c \"import secrets; print(secrets.token_urlsafe(48))\"`) "
            "and set it in the environment."
        )
    return JWT_SECRET


def create_access_token(
    username: str,
    role: str,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """Issue a signed JWT for an authenticated user.

    Args:
        username: The subject identifier.
        role: The role claim. Must be one of ``_ROLE_HIERARCHY``.
        extra_claims: Optional additional claims merged into the payload.

    Returns:
        A compact JWT string.

    Raises:
        ValueError: If ``role`` is not a recognised role.
        RuntimeError: If ``JWT_SECRET`` is not configured.
    """
    if role not in _ROLE_RANK:
        raise ValueError(
            f"Unknown role {role!r}. Valid roles: {_ROLE_HIERARCHY}"
        )

    import jwt

    now = datetime.datetime.now(datetime.timezone.utc)
    payload: Dict[str, Any] = {
        "sub": username,
        "role": role,
        "iss": JWT_ISSUER,
        "iat": int(now.timestamp()),
        "exp": int((now + datetime.timedelta(minutes=ACCESS_TOKEN_TTL_MINUTES)).timestamp()),
    }
    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(payload, _require_secret(), algorithm=JWT_ALGORITHM)


async def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT.

    Args:
        token: The compact JWT string from the Authorization header.

    Returns:
        The decoded claims dictionary.

    Raises:
        HTTPException(401): If the token is missing, malformed, expired,
            or has an invalid signature.
    """
    import jwt

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(
            token,
            _require_secret(),
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            options={"require": ["exp", "iat", "sub", "role"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidIssuerError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token issuer is invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is invalid or malformed.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


# ---------------------------------------------------------------------------
# Role helpers
# ---------------------------------------------------------------------------


def _role_satisfies(user_role: str, required_role: str) -> bool:
    """Return True if ``user_role`` satisfies ``required_role``.

    Satisfaction is hierarchical: an admin satisfies any lower role, a
    contributor satisfies ``contributor`` and ``viewer``, a viewer
    satisfies only ``viewer``.

    Args:
        user_role: The role extracted from the JWT.
        required_role: The role required by the endpoint.

    Returns:
        ``True`` if the user role is sufficient.
    """
    if user_role not in _ROLE_RANK or required_role not in _ROLE_RANK:
        return False
    return _ROLE_RANK[user_role] <= _ROLE_RANK[required_role]


def roles_for_user(username: str) -> Set[str]:
    """Return the set of roles granted to a user.

    Args:
        username: The subject identifier.

    Returns:
        A set of role names, or an empty set if the user is unknown.
    """
    record = _user_store.get(username)
    if not record:
        return set()
    return {record.get("role", "viewer")}


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """FastAPI dependency that resolves the authenticated user claims.

    Args:
        token: The JWT extracted from the Authorization header by the
            OAuth2 scheme.

    Returns:
        The decoded JWT claims dictionary.

    Raises:
        HTTPException(401): On any authentication failure.
    """
    return await decode_token(token)


def verify_role(
    required_role: str,
) -> Callable[..., Any]:
    """Build a FastAPI dependency that enforces a minimum role.

    Usage::

        @router.post("/rag/documents/upload",
                     dependencies=[Depends(verify_role("contributor"))])
        async def upload(...): ...

    The dependency decodes the JWT from the ``Authorization: Bearer ...``
    header, checks the ``role`` claim against the hierarchy, and raises
    HTTP 403 if the user's role is insufficient. Authentication failures
    (missing/invalid token) raise HTTP 401.

    Args:
        required_role: The minimum role required to access the endpoint.
            Must be one of ``admin``, ``contributor``, ``viewer``.

    Returns:
        An async callable suitable for ``Depends(...)`` that returns the
        decoded claims on success.

    Raises:
        ValueError: If ``required_role`` is not a recognised role.
    """
    if required_role not in _ROLE_RANK:
        raise ValueError(
            f"Unknown required_role {required_role!r}. "
            f"Valid roles: {_ROLE_HIERARCHY}"
        )

    async def _dependency(
        token: str = Depends(oauth2_scheme),
    ) -> Dict[str, Any]:
        claims = await decode_token(token)
        user_role = str(claims.get("role", ""))

        if not _role_satisfies(user_role, required_role):
            logger.warning(
                "Authorisation denied: user=%s role=%s required=%s",
                claims.get("sub"),
                user_role,
                required_role,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Insufficient permissions. Required role: "
                    f"{required_role}. Your role: {user_role}."
                ),
            )

        return claims

    return _dependency


# ---------------------------------------------------------------------------
# User store helpers (bootstrap / registration)
# ---------------------------------------------------------------------------


def register_user(username: str, password: str, role: str = "viewer") -> None:
    """Register a new user in the in-process store.

    Args:
        username: Unique username.
        password: Plaintext password; hashed before storage.
        role: Role to assign. Defaults to the least-privileged role.

    Raises:
        ValueError: If the username already exists or the role is invalid.
    """
    if role not in _ROLE_RANK:
        raise ValueError(f"Invalid role {role!r}.")
    if username in _user_store:
        raise ValueError(f"User {username!r} already exists.")

    _user_store[username] = {
        "username": username,
        "hashed_password": hash_password(password),
        "role": role,
    }
    logger.info("Registered user %s with role %s.", username, role)


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate a user against the in-process store.

    Args:
        username: The username to authenticate.
        password: The plaintext password.

    Returns:
        The user record on success, or ``None`` if credentials are invalid.
    """
    record = _user_store.get(username)
    if not record:
        return None
    if not verify_password(password, record["hashed_password"]):
        return None
    return record


def bootstrap_admin_from_env() -> None:
    """Create a default admin account from environment variables.

    Reads ``LUMANGUIDE_ADMIN_USER`` and ``LUMANGUIDE_ADMIN_PASSWORD`` and,
    if both are set and the user does not already exist, registers an
    admin. This is safe to call at startup.
    """
    admin_user = os.getenv("LUMANGUIDE_ADMIN_USER", "").strip()
    admin_pass = os.getenv("LUMANGUIDE_ADMIN_PASSWORD", "").strip()
    if not admin_user or not admin_pass:
        return
    if admin_user in _user_store:
        return
    try:
        register_user(admin_user, admin_pass, role="admin")
        logger.info("Bootstrapped admin account %s from environment.", admin_user)
    except Exception as exc:
        logger.warning("Failed to bootstrap admin account: %s", exc)


__all__ = [
    "verify_role",
    "get_current_user",
    "create_access_token",
    "decode_token",
    "hash_password",
    "verify_password",
    "register_user",
    "authenticate_user",
    "bootstrap_admin_from_env",
    "roles_for_user",
]
