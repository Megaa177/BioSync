"""
biosync/crypto.py

Cryptographic primitives for BioSync, a federated fingerprint recognition
system. This module never handles raw fingerprint images — only opaque
byte payloads (e.g. extracted feature templates or model update tensors
serialized upstream). Nothing here assumes a particular dataset.

Three responsibilities:
  1. Template encryption at rest / in transit      -> AES-256-GCM
  2. Ephemeral key agreement between federation nodes -> X25519 + HKDF
  3. Signing & verifying model updates from clients  -> Ed25519

Design notes:
  - AES-GCM is used (not CBC) so we get authenticated encryption for free;
    a tampered ciphertext fails to decrypt rather than silently corrupting.
  - Keys are never derived from passwords here — that's out of scope for
    this module. Feed it real key material (os.urandom, an HSM, a KMS, etc).
  - All public functions operate on bytes in / bytes out so they're easy
    to unit test without a real fingerprint dataset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric import x25519, ed25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.exceptions import InvalidSignature, InvalidTag

AES_KEY_SIZE = 32       # 256-bit key
NONCE_SIZE = 12         # 96-bit nonce, standard for GCM


# --------------------------------------------------------------------------
# 1. Template encryption (AES-256-GCM)
# --------------------------------------------------------------------------

def generate_aes_key() -> bytes:
    """Generate a fresh random 256-bit AES key."""
    return AESGCM.generate_key(bit_length=256)


def encrypt_template(key: bytes, plaintext: bytes, associated_data: bytes | None = None) -> bytes:
    """
    Encrypt a fingerprint template (or any payload) with AES-256-GCM.

    Returns nonce || ciphertext (ciphertext includes the GCM auth tag).
    `associated_data` is authenticated but not encrypted — useful for
    binding a ciphertext to metadata like a device_id or node_id so it
    can't be replayed elsewhere.
    """
    if len(key) != AES_KEY_SIZE:
        raise ValueError(f"AES key must be {AES_KEY_SIZE} bytes, got {len(key)}")
    aesgcm = AESGCM(key)
    nonce = os.urandom(NONCE_SIZE)
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)
    return nonce + ciphertext


def decrypt_template(key: bytes, blob: bytes, associated_data: bytes | None = None) -> bytes:
    """
    Decrypt a blob produced by encrypt_template.
    Raises InvalidTag if the ciphertext was tampered with or the key/AAD is wrong.
    """
    if len(blob) < NONCE_SIZE:
        raise ValueError("Ciphertext blob too short to contain a nonce")
    aesgcm = AESGCM(key)
    nonce, ciphertext = blob[:NONCE_SIZE], blob[NONCE_SIZE:]
    return aesgcm.decrypt(nonce, ciphertext, associated_data)


# --------------------------------------------------------------------------
# 2. Node-to-node key agreement (X25519 + HKDF)
# --------------------------------------------------------------------------

@dataclass
class NodeKeyPair:
    """An ephemeral X25519 keypair for a single federation node/session."""
    private_key: x25519.X25519PrivateKey
    public_key: x25519.X25519PublicKey

    @classmethod
    def generate(cls) -> "NodeKeyPair":
        priv = x25519.X25519PrivateKey.generate()
        return cls(private_key=priv, public_key=priv.public_key())

    def public_bytes(self) -> bytes:
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )


def derive_shared_key(
    my_private_key: x25519.X25519PrivateKey,
    their_public_bytes: bytes,
    info: bytes = b"biosync-federation-session",
) -> bytes:
    """
    Perform X25519 ECDH with a peer's raw public key bytes, then run the
    shared secret through HKDF-SHA256 to get a uniformly random 256-bit
    session key suitable for AES-GCM.
    """
    their_public_key = x25519.X25519PublicKey.from_public_bytes(their_public_bytes)
    shared_secret = my_private_key.exchange(their_public_key)
    return HKDF(
        algorithm=hashes.SHA256(),
        length=AES_KEY_SIZE,
        salt=None,
        info=info,
    ).derive(shared_secret)


# --------------------------------------------------------------------------
# 3. Signing & verifying federated model updates (Ed25519)
# --------------------------------------------------------------------------

@dataclass
class SigningIdentity:
    """A long-lived Ed25519 identity for a client/node that submits model updates."""
    private_key: ed25519.Ed25519PrivateKey
    public_key: ed25519.Ed25519PublicKey

    @classmethod
    def generate(cls) -> "SigningIdentity":
        priv = ed25519.Ed25519PrivateKey.generate()
        return cls(private_key=priv, public_key=priv.public_key())

    def public_bytes(self) -> bytes:
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def sign(self, payload: bytes) -> bytes:
        return self.private_key.sign(payload)


def verify_update(public_bytes: bytes, payload: bytes, signature: bytes) -> bool:
    """
    Verify a signed model update. Returns True/False instead of raising,
    since callers typically just want to accept-or-reject a submission.
    """
    public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_bytes)
    try:
        public_key.verify(signature, payload)
        return True
    except InvalidSignature:
        return False


__all__ = [
    "generate_aes_key",
    "encrypt_template",
    "decrypt_template",
    "NodeKeyPair",
    "derive_shared_key",
    "SigningIdentity",
    "verify_update",
    "InvalidTag",
]