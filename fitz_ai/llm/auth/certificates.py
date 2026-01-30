# fitz_ai/llm/auth/certificates.py
"""
Certificate validation utilities for startup-time verification.

Provides user-friendly error messages for certificate problems instead of
cryptic SSL errors at runtime. Validates PEM certificates and private keys
with actionable guidance on how to fix issues.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives.serialization import load_pem_private_key

logger = logging.getLogger(__name__)


class CertificateError(Exception):
    """User-friendly certificate error with actionable message."""

    pass


def validate_certificate_file(path: str, cert_type: str) -> None:
    """
    Validate a certificate file with user-friendly error messages.

    Checks that the file exists, is a valid PEM certificate, and has not
    expired. Logs a warning if the certificate expires within 7 days.

    Args:
        path: Path to the certificate file.
        cert_type: Human-readable type for error messages (e.g., "CA certificate",
            "Client certificate").

    Raises:
        CertificateError: With actionable error message describing the problem
            and how to fix it.
    """
    file_path = Path(path)

    # Check file exists
    if not file_path.exists():
        raise CertificateError(
            f"{cert_type} file not found: {path}\n"
            f"Please verify the path is correct and the file exists."
        )

    # Check path is a file, not a directory
    if not file_path.is_file():
        raise CertificateError(
            f"{cert_type} path is a directory, not a file: {path}\n"
            f"Please provide a path to a certificate file, not a directory."
        )

    # Try to read and parse the certificate
    try:
        cert_data = file_path.read_bytes()
        cert = x509.load_pem_x509_certificate(cert_data)
    except ValueError as e:
        raise CertificateError(
            f"{cert_type} file is not a valid PEM certificate: {path}\n"
            f"Error: {e}\n"
            f"Please ensure the file is a PEM-encoded certificate "
            f"(starts with -----BEGIN CERTIFICATE-----)."
        )

    # Check expiration
    now = datetime.now(timezone.utc)
    if cert.not_valid_after_utc < now:
        raise CertificateError(
            f"{cert_type} has expired: {path}\n"
            f"Expired on: {cert.not_valid_after_utc.isoformat()}\n"
            f"Please obtain a new certificate."
        )

    # Warn if expiring soon (within 7 days)
    days_until_expiry = (cert.not_valid_after_utc - now).days
    if days_until_expiry < 7:
        logger.warning(
            f"{cert_type} will expire in {days_until_expiry} days: {path}"
        )


def validate_key_file(
    path: str, key_type: str, password: str | None = None
) -> None:
    """
    Validate a private key file with user-friendly error messages.

    Checks that the file exists, is a valid PEM private key, and can be
    decrypted if password-protected.

    Args:
        path: Path to the private key file.
        key_type: Human-readable type for error messages (e.g., "Client key").
        password: Optional password for encrypted private keys.

    Raises:
        CertificateError: With actionable error message describing the problem
            and how to fix it.
    """
    file_path = Path(path)

    # Check file exists
    if not file_path.exists():
        raise CertificateError(
            f"{key_type} file not found: {path}\n"
            f"Please verify the path is correct and the file exists."
        )

    # Check path is a file, not a directory
    if not file_path.is_file():
        raise CertificateError(
            f"{key_type} path is a directory, not a file: {path}\n"
            f"Please provide a path to a key file, not a directory."
        )

    # Try to load the private key
    try:
        key_data = file_path.read_bytes()
        password_bytes = password.encode() if password else None
        load_pem_private_key(key_data, password=password_bytes)
    except ValueError as e:
        error_msg = str(e).lower()
        if "password" in error_msg or "encrypted" in error_msg:
            raise CertificateError(
                f"{key_type} is encrypted but no password was provided: {path}\n"
                f"Please configure 'client_key_password' with the key password."
            )
        raise CertificateError(
            f"{key_type} file is not a valid PEM private key: {path}\n"
            f"Error: {e}\n"
            f"Please ensure the file is a PEM-encoded private key "
            f"(starts with -----BEGIN PRIVATE KEY----- or similar)."
        )
    except TypeError as e:
        # Wrong password provided
        raise CertificateError(
            f"{key_type} password is incorrect: {path}\n"
            f"Error: {e}\n"
            f"Please verify the 'client_key_password' configuration."
        )
