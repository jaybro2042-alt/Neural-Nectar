"""
AES-GCM encryption helpers for tools.memory. Uses PyCryptodome if available.
Provides encrypt_bytes(key_bytes, plaintext_bytes) -> base64 str
and decrypt_bytes(key_bytes, b64str) -> plaintext_bytes

Key must be 16/24/32 bytes for AES-128/192/256.
"""
import base64

try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    _crypto_available = True
except Exception:
    _crypto_available = False


def _ensure_available():
    if not _crypto_available:
        raise RuntimeError('PyCryptodome not available. Install package "pycryptodome" to use encryption.')


def encrypt_bytes(key: bytes, plaintext: bytes) -> str:
    _ensure_available()
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError('key must be bytes')
    # AES-GCM
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    payload = nonce + tag + ciphertext
    return base64.b64encode(payload).decode('ascii')


def decrypt_bytes(key: bytes, b64payload: str) -> bytes:
    _ensure_available()
    payload = base64.b64decode(b64payload)
    if len(payload) < 12 + 16:
        raise ValueError('payload too short')
    nonce = payload[:12]
    tag = payload[12:28]
    ciphertext = payload[28:]
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    plaintext = cipher.decrypt_and_verify(ciphertext, tag)
    return plaintext

