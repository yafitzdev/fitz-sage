# tests/unit/llm/test_certificates.py
"""
Unit tests for certificate validation utilities and mTLS configuration.

Tests cover:
- CertificateError exception class
- validate_certificate_file() for various error conditions
- validate_key_file() for key validation
- M2MAuth mTLS configuration through get_request_kwargs()
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from fitz_ai.llm.auth import M2MAuth
from fitz_ai.llm.auth.certificates import (
    CertificateError,
    validate_certificate_file,
    validate_key_file,
)


class TestCertificateError:
    """Test CertificateError exception class."""

    def test_certificate_error_is_exception(self) -> None:
        """CertificateError inherits from Exception."""
        assert issubclass(CertificateError, Exception)

    def test_certificate_error_message(self) -> None:
        """CertificateError preserves error message."""
        error = CertificateError("test message")
        assert str(error) == "test message"


class TestValidateCertificateFile:
    """Test validate_certificate_file() function."""

    def test_validate_certificate_file_not_found(self) -> None:
        """Missing certificate file raises CertificateError with path."""
        with pytest.raises(CertificateError) as exc_info:
            validate_certificate_file("/nonexistent/cert.pem", "Test certificate")

        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg
        assert "/nonexistent/cert.pem" in str(exc_info.value)

    def test_validate_certificate_file_is_directory(self, tmp_path: Path) -> None:
        """Directory instead of file raises CertificateError with guidance."""
        # tmp_path is already a directory
        with pytest.raises(CertificateError) as exc_info:
            validate_certificate_file(str(tmp_path), "Test certificate")

        error_msg = str(exc_info.value).lower()
        assert "directory" in error_msg

    def test_validate_certificate_file_invalid_pem(self, tmp_path: Path) -> None:
        """Invalid PEM format raises CertificateError explaining expected format."""
        invalid_cert = tmp_path / "invalid.pem"
        invalid_cert.write_text("not a certificate")

        with pytest.raises(CertificateError) as exc_info:
            validate_certificate_file(str(invalid_cert), "Test certificate")

        error_msg = str(exc_info.value)
        # Should mention PEM format or the expected header
        assert "PEM" in error_msg or "-----BEGIN CERTIFICATE-----" in error_msg

    def test_validate_certificate_file_expired(self, tmp_path: Path) -> None:
        """Expired certificate raises CertificateError with expiry date."""
        # Generate an expired certificate
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(timezone.utc) - timedelta(days=365))
            .not_valid_after(datetime.now(timezone.utc) - timedelta(days=1))  # Expired yesterday
            .sign(key, hashes.SHA256())
        )

        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        expired_cert = tmp_path / "expired.pem"
        expired_cert.write_bytes(cert_pem)

        with pytest.raises(CertificateError) as exc_info:
            validate_certificate_file(str(expired_cert), "Test certificate")

        error_msg = str(exc_info.value).lower()
        assert "expired" in error_msg

    def test_validate_certificate_file_valid(self, tmp_path: Path) -> None:
        """Valid certificate does NOT raise."""
        # Generate a valid certificate (expires in 365 days)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(timezone.utc))
            .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
            .sign(key, hashes.SHA256())
        )

        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        valid_cert = tmp_path / "valid.pem"
        valid_cert.write_bytes(cert_pem)

        # Should not raise
        validate_certificate_file(str(valid_cert), "Test certificate")


class TestValidateKeyFile:
    """Test validate_key_file() function."""

    def test_validate_key_file_not_found(self) -> None:
        """Missing key file raises CertificateError with path."""
        with pytest.raises(CertificateError) as exc_info:
            validate_key_file("/nonexistent/key.pem", "Test key")

        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg
        assert "/nonexistent/key.pem" in str(exc_info.value)

    def test_validate_key_file_invalid_format(self, tmp_path: Path) -> None:
        """Invalid key format raises CertificateError."""
        invalid_key = tmp_path / "invalid.key"
        invalid_key.write_text("not a key")

        with pytest.raises(CertificateError) as exc_info:
            validate_key_file(str(invalid_key), "Test key")

        error_msg = str(exc_info.value)
        # Should mention PEM format or expected header
        assert "PEM" in error_msg or "-----BEGIN" in error_msg

    def test_validate_key_file_valid(self, tmp_path: Path) -> None:
        """Valid private key does NOT raise."""
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        key_pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )

        valid_key = tmp_path / "valid.key"
        valid_key.write_bytes(key_pem)

        # Should not raise
        validate_key_file(str(valid_key), "Test key")


class TestM2MAuthCertificates:
    """Test M2MAuth mTLS configuration."""

    def test_m2m_auth_validates_invalid_cert_path(self) -> None:
        """M2MAuth validates certificates at init and raises CertificateError."""
        with pytest.raises(CertificateError) as exc_info:
            M2MAuth(
                token_url="https://auth.example.com/token",
                client_id="my-client",
                client_secret="my-secret",
                client_cert_path="/nonexistent/cert.pem",
            )

        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg

    def test_m2m_auth_get_request_kwargs_with_client_cert(
        self, temp_certificate: tuple[str, str]
    ) -> None:
        """M2MAuth.get_request_kwargs returns cert tuple with cert and key paths."""
        cert_path, key_path = temp_certificate

        auth = M2MAuth(
            token_url="https://auth.example.com/token",
            client_id="my-client",
            client_secret="my-secret",
            client_cert_path=cert_path,
            client_key_path=key_path,
        )

        kwargs = auth.get_request_kwargs()
        assert "cert" in kwargs
        assert kwargs["cert"] == (cert_path, key_path)

    def test_m2m_auth_get_request_kwargs_with_password(self, tmp_path: Path) -> None:
        """M2MAuth.get_request_kwargs includes password in cert tuple."""
        # Generate an encrypted key
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        password = b"test-password"
        key_pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.BestAvailableEncryption(password),
        )

        # Generate matching certificate
        subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(timezone.utc))
            .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
            .sign(key, hashes.SHA256())
        )
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)

        cert_file = tmp_path / "cert.pem"
        key_file = tmp_path / "key.pem"
        cert_file.write_bytes(cert_pem)
        key_file.write_bytes(key_pem)

        auth = M2MAuth(
            token_url="https://auth.example.com/token",
            client_id="my-client",
            client_secret="my-secret",
            client_cert_path=str(cert_file),
            client_key_path=str(key_file),
            client_key_password="test-password",
        )

        kwargs = auth.get_request_kwargs()
        assert kwargs["cert"] == (str(cert_file), str(key_file), "test-password")

    def test_m2m_auth_get_request_kwargs_combined_pem(
        self, temp_certificate: tuple[str, str]
    ) -> None:
        """M2MAuth.get_request_kwargs returns single path for combined PEM."""
        cert_path, _ = temp_certificate

        auth = M2MAuth(
            token_url="https://auth.example.com/token",
            client_id="my-client",
            client_secret="my-secret",
            client_cert_path=cert_path,
            # No key_path - combined PEM scenario
        )

        kwargs = auth.get_request_kwargs()
        # Single path, not a tuple
        assert kwargs["cert"] == cert_path
        assert isinstance(kwargs["cert"], str)
