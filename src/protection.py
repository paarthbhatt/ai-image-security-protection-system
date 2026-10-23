"""Hashing, authenticated encryption, and audit records for the prototype."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: str | Path, expected: str) -> bool:
    return sha256_file(path) == expected


def encrypt_bytes(plaintext: bytes, key: bytes) -> tuple[bytes, bytes]:
    """Encrypt bytes using AES-GCM; nonce is returned separately."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if len(key) not in (16, 24, 32):
        raise ValueError("AES-GCM keys must be 128, 192, or 256 bits")
    nonce = os.urandom(12)
    return nonce, AESGCM(key).encrypt(nonce, plaintext, None)


def decrypt_bytes(nonce: bytes, ciphertext: bytes, key: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    return AESGCM(key).decrypt(nonce, ciphertext, None)


def run_protection_checks(sample_path: str | Path, output_dir: Path) -> dict[str, object]:
    """Run deterministic hash and authenticated-encryption demonstrations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = Path(sample_path)
    payload = path.read_bytes()
    original_hash = sha256_file(path)
    tampered_path = output_dir / "tampered_copy.bin"
    mutated = bytearray(payload)
    mutated[len(mutated) // 2] ^= 0x01
    tampered_path.write_bytes(mutated)
    hash_check = verify_hash(path, original_hash)
    hash_rejects_mutation = not verify_hash(tampered_path, original_hash)

    encryption_ok = False
    tamper_rejected = False
    error = None
    try:
        key = hashlib.sha256(b"presentation-demo-key-not-a-secret").digest()
        nonce, ciphertext = encrypt_bytes(payload, key)
        recovered = decrypt_bytes(nonce, ciphertext, key)
        encryption_ok = recovered == payload
        modified_ciphertext = bytearray(ciphertext)
        modified_ciphertext[-1] ^= 0x01
        try:
            decrypt_bytes(nonce, bytes(modified_ciphertext), key)
        except Exception:
            tamper_rejected = True
    except Exception as exc:  # pragma: no cover - useful in minimal environments
        error = str(exc)

    result = {
        "sample_file": str(path),
        "sha256": original_hash,
        "reference_hash_matches": hash_check,
        "mutated_file_rejected": hash_rejects_mutation,
        "aes_gcm_round_trip": encryption_ok,
        "aes_gcm_modified_ciphertext_rejected": tamper_rejected,
        "encryption_error": error,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "audit_record.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
