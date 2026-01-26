# tests/e2e/fixtures/code_sample.py
"""
Authentication module for the TechCorp application.

This module provides user authentication, session management,
and authorization functionality.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta


class AuthenticationError(Exception):
    """Raised when authentication fails."""

    pass


class SessionExpiredError(Exception):
    """Raised when a session has expired."""

    pass


@dataclass
class User:
    """Represents an authenticated user."""

    user_id: str
    username: str
    email: str
    role: str
    created_at: datetime


@dataclass
class Session:
    """Represents an active user session."""

    token: str
    user_id: str
    created_at: datetime
    expires_at: datetime

    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return datetime.utcnow() > self.expires_at


class UserAuth:
    """
    Handles user authentication and session management.

    This class provides methods for:
    - User login with username/password
    - Session token generation and validation
    - User logout and session cleanup
    - Password hashing and verification

    Example:
        auth = UserAuth(secret_key="my-secret-key")
        token = auth.login("username", "password")
        user = auth.validate_session(token)
        auth.logout(token)
    """

    def __init__(self, secret_key: str, session_duration_hours: int = 24):
        """
        Initialize the authentication handler.

        Args:
            secret_key: Secret key for token generation
            session_duration_hours: How long sessions remain valid
        """
        self.secret_key = secret_key
        self.session_duration = timedelta(hours=session_duration_hours)
        self._sessions: dict[str, Session] = {}
        self._users: dict[str, tuple[str, User]] = {}  # username -> (password_hash, User)

    def register_user(self, username: str, password: str, email: str, role: str = "user") -> User:
        """
        Register a new user.

        Args:
            username: Unique username
            password: Plain text password (will be hashed)
            email: User email address
            role: User role (default: "user")

        Returns:
            The created User object

        Raises:
            ValueError: If username already exists
        """
        if username in self._users:
            raise ValueError(f"Username '{username}' already exists")

        user_id = f"U{secrets.token_hex(8)}"
        password_hash = self._hash_password(password)
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            created_at=datetime.utcnow(),
        )
        self._users[username] = (password_hash, user)
        return user

    def login(self, username: str, password: str) -> str:
        """
        Authenticate user and create a new session.

        Args:
            username: User's username
            password: User's password

        Returns:
            Session token string

        Raises:
            AuthenticationError: If credentials are invalid
        """
        if username not in self._users:
            raise AuthenticationError("Invalid credentials")

        password_hash, user = self._users[username]
        if not self._verify_password(password, password_hash):
            raise AuthenticationError("Invalid credentials")

        token = self._generate_token()
        session = Session(
            token=token,
            user_id=user.user_id,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + self.session_duration,
        )
        self._sessions[token] = session
        return token

    def logout(self, token: str) -> None:
        """
        Invalidate a session token.

        Args:
            token: Session token to invalidate
        """
        self._sessions.pop(token, None)

    def validate_session(self, token: str) -> User:
        """
        Validate a session token and return the associated user.

        Args:
            token: Session token to validate

        Returns:
            The authenticated User object

        Raises:
            AuthenticationError: If token is invalid
            SessionExpiredError: If session has expired
        """
        session = self._sessions.get(token)
        if session is None:
            raise AuthenticationError("Invalid session token")

        if session.is_expired():
            self._sessions.pop(token, None)
            raise SessionExpiredError("Session has expired")

        # Find user by user_id
        for _, (_, user) in self._users.items():
            if user.user_id == session.user_id:
                return user

        raise AuthenticationError("User not found")

    def refresh_session(self, token: str) -> str:
        """
        Refresh a session, extending its expiration.

        Args:
            token: Current session token

        Returns:
            New session token

        Raises:
            AuthenticationError: If token is invalid
        """
        user = self.validate_session(token)
        self.logout(token)

        new_token = self._generate_token()
        session = Session(
            token=new_token,
            user_id=user.user_id,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + self.session_duration,
        )
        self._sessions[new_token] = session
        return new_token

    def _generate_token(self) -> str:
        """Generate a secure random token."""
        return secrets.token_urlsafe(32)

    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256 with salt."""
        salt = self.secret_key.encode()
        return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000).hex()

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        return self._hash_password(password) == password_hash


class RoleAuthorizer:
    """
    Handles role-based authorization.

    Defines what actions each role can perform.
    """

    ROLE_PERMISSIONS = {
        "admin": ["read", "write", "delete", "manage_users"],
        "manager": ["read", "write", "delete"],
        "user": ["read", "write"],
        "guest": ["read"],
    }

    def can_perform(self, user: User, action: str) -> bool:
        """
        Check if a user can perform an action.

        Args:
            user: The user to check
            action: The action to perform

        Returns:
            True if allowed, False otherwise
        """
        permissions = self.ROLE_PERMISSIONS.get(user.role, [])
        return action in permissions
