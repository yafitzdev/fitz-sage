# fitz_ai/cloud/crypto.py
"""Client-side encryption for Fitz Cloud cache.

CRITICAL: The org_key NEVER leaves this machine. The server cannot decrypt stored data.
"""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass
class EncryptedBlob:
    """Encrypted cache blob with metadata."""

    ciphertext: bytes  # AES-256-GCM encrypted data
    timestamp: str  # ISO 8601 timestamp used in AAD
    org_id: str  # Organization ID (for AAD)


class CacheEncryption:
    """
    AES-256-GCM encryption for cache entries.

    Additional Authenticated Data (AAD) includes:
    - org_id
    - timestamp (ISO 8601)
    This ensures the blob can only be decrypted with matching AAD.
    """

    def __init__(self, org_key: str):
        """
        Initialize encryption with organization key.

        Args:
            org_key: 32-byte hex string or base64-encoded key
        """
        # Derive 32-byte key from org_key
        if len(org_key) == 64:
            # Hex string
            key_bytes = bytes.fromhex(org_key)
        elif len(org_key) == 44:
            # Base64
            key_bytes = base64.b64decode(org_key)
        else:
            # Arbitrary string - derive key
            key_bytes = hashlib.sha256(org_key.encode()).digest()

        if len(key_bytes) != 32:
            raise ValueError("org_key must derive to 32 bytes for AES-256")

        self.aesgcm = AESGCM(key_bytes)

    def encrypt(self, plaintext: str, org_id: str) -> EncryptedBlob:
        """
        Encrypt plaintext with AES-256-GCM.

        Args:
            plaintext: Data to encrypt (JSON-serialized Answer)
            org_id: Organization ID (included in AAD)

        Returns:
            EncryptedBlob with ciphertext, timestamp, and org_id
        """
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Additional Authenticated Data
        aad = f"{org_id}:{timestamp}".encode()

        # Generate 96-bit (12 byte) nonce
        nonce = os.urandom(12)

        # Encrypt
        ciphertext = self.aesgcm.encrypt(nonce, plaintext.encode(), aad)

        # Prepend nonce to ciphertext (standard practice)
        blob = nonce + ciphertext

        return EncryptedBlob(
            ciphertext=blob,
            timestamp=timestamp,
            org_id=org_id,
        )

    def decrypt(self, blob: EncryptedBlob) -> str:
        """
        Decrypt ciphertext with AES-256-GCM.

        Args:
            blob: EncryptedBlob with ciphertext, timestamp, org_id

        Returns:
            Decrypted plaintext (JSON-serialized Answer)

        Raises:
            ValueError: If decryption fails (wrong key, tampered data, or mismatched AAD)
        """
        # Extract nonce (first 12 bytes) and ciphertext
        nonce = blob.ciphertext[:12]
        ciphertext = blob.ciphertext[12:]

        # Reconstruct AAD
        aad = f"{blob.org_id}:{blob.timestamp}".encode()

        try:
            plaintext_bytes = self.aesgcm.decrypt(nonce, ciphertext, aad)
            return plaintext_bytes.decode()
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")


def generate_org_key() -> str:
    """
    Generate a new random org_key.

    Returns:
        64-character hex string (32 bytes)
    """
    return os.urandom(32).hex()
