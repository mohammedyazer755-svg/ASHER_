"""Small reference implementation of ASHER companion protocol v1.

This module is a test/mock adapter, not a production transport. It mirrors the
wire formulas in ``app/src/main/.../protocol`` so a desktop implementation can
be tested without an Android SDK.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import struct
import time
import re
from dataclasses import dataclass
from typing import Any

try:  # Optional for environments that only run the Kotlin/manifest checks.
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
except ImportError:  # pragma: no cover - exercised only on minimal installs
    hashes = serialization = ec = AESGCM = HKDF = None  # type: ignore[assignment]

VERSION = 1
CODE_COMMITMENT_DOMAIN = "asher/companion/pairing-code/v1"
CODE_PROOF_DOMAIN = "asher/companion/pairing-proof/v1"
HKDF_INFO = b"asher/companion/channel/v1"


def b64e(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _require_crypto() -> None:
    if ec is None or AESGCM is None or HKDF is None:
        raise RuntimeError("cryptography is required for the PC protocol mock")


def b64d(value: str) -> bytes:
    if not value or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for ch in value):
        raise ValueError("non-canonical base64")
    decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    if b64e(decoded) != value:
        raise ValueError("non-canonical base64")
    return decoded


def normalize_code(value: str) -> str:
    value = "".join(ch for ch in value if ch.isalnum()).upper()
    if not 6 <= len(value) <= 32 or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" for ch in value):
        raise ValueError("pairing code must contain 6-32 ASCII letters/digits")
    return value


def _token(value: str) -> str:
    if not value or len(value) > 128 or re.fullmatch(r"[A-Za-z0-9._~:-]+", value) is None:
        raise ValueError("invalid protocol token")
    return value


def _length_prefixed(*values: str) -> bytes:
    joined = "|".join(values).encode("utf-8")
    return struct.pack(">I", len(joined)) + joined


def code_commitment(
    pairing_id: str,
    nonce: bytes,
    pc_public_key: bytes,
    expires_at: int,
    code: str,
) -> bytes:
    return hashlib.sha256(
        _length_prefixed(
            CODE_COMMITMENT_DOMAIN,
            pairing_id,
            b64e(nonce),
            b64e(pc_public_key),
            str(expires_at),
            normalize_code(code),
        )
    ).digest()


def code_proof(pairing_id: str, nonce: bytes, code: str) -> bytes:
    return hashlib.sha256(
        _length_prefixed(CODE_PROOF_DOMAIN, pairing_id, b64e(nonce), normalize_code(code))
    ).digest()


def invitation_json(invitation: dict[str, Any]) -> str:
    """Canonical invitation ordering used by ``CryptoPrimitives.invitationDigest``."""
    ordered = {
        "code_commitment": invitation["code_commitment"],
        "expires_at": invitation["expires_at"],
        "nonce": invitation["nonce"],
        "pairing_id": invitation["pairing_id"],
        "pc_public_key": invitation["pc_public_key"],
        "v": invitation.get("v", VERSION),
    }
    return json.dumps(ordered, separators=(",", ":"), ensure_ascii=False)


def invitation_digest(invitation: dict[str, Any]) -> bytes:
    canonical = invitation_json(invitation)
    return hashlib.sha256(f"asher/companion/pairing-code/v1|{canonical}".encode()).digest()


@dataclass
class PcInvitation:
    payload: dict[str, Any]
    private_key: ec.EllipticCurvePrivateKey
    code: str

    @property
    def json(self) -> str:
        return invitation_json(self.payload)

    @property
    def deep_link(self) -> str:
        return f"asher://pair?payload={b64e(self.json.encode())}"


def make_invitation(
    pairing_id: str = "test-pairing",
    code: str = "ASHER42",
    expires_at: int | None = None,
) -> PcInvitation:
    _require_crypto()
    _token(pairing_id)
    expiry = expires_at if expires_at is not None else int(time.time()) + 300
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    nonce = os.urandom(32)
    payload = {
        "v": VERSION,
        "pairing_id": pairing_id,
        "pc_public_key": b64e(public_key),
        "nonce": b64e(nonce),
        "code_commitment": b64e(code_commitment(pairing_id, nonce, public_key, expiry, code)),
        "expires_at": expiry,
    }
    return PcInvitation(payload, private_key, normalize_code(code))


def _derive(private_key: ec.EllipticCurvePrivateKey, peer_public_der: str, nonce: bytes) -> bytes:
    _require_crypto()
    peer = serialization.load_der_public_key(b64d(peer_public_der))
    if not isinstance(peer, ec.EllipticCurvePublicKey):
        raise ValueError("peer key is not EC")
    shared = private_key.exchange(ec.ECDH(), peer)
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=nonce, info=HKDF_INFO).derive(shared)


def complete_request(invitation: PcInvitation, request: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    if request.get("v") != VERSION or request.get("pairing_id") != invitation.payload["pairing_id"]:
        raise ValueError("request is not bound to invitation")
    if request.get("nonce") != invitation.payload["nonce"]:
        raise ValueError("nonce mismatch")
    expected_proof = code_proof(
        invitation.payload["pairing_id"], b64d(invitation.payload["nonce"]), invitation.code
    )
    if not hmac.compare_digest(b64d(request["code_proof"]), expected_proof):
        raise ValueError("code proof mismatch")
    expected_digest = invitation_digest(invitation.payload)
    if not hmac.compare_digest(b64d(request["invitation_digest"]), expected_digest):
        raise ValueError("invitation digest mismatch")

    channel_key = _derive(invitation.private_key, request["mobile_public_key"], b64d(invitation.payload["nonce"]))
    response_without_mac = {
        "nonce": invitation.payload["nonce"],
        "pairing_id": invitation.payload["pairing_id"],
        "pc_public_key": invitation.payload["pc_public_key"],
        "v": VERSION,
    }
    response_canonical = json.dumps(response_without_mac, separators=(",", ":"), ensure_ascii=False)
    request_canonical = json.dumps(request, separators=(",", ":"), ensure_ascii=False)
    mac = hmac.new(channel_key, f"{request_canonical}\n{response_canonical}".encode(), hashlib.sha256).digest()
    response = dict(response_without_mac, transcript_mac=b64e(mac))
    return response, channel_key


def encrypt_frame(key: bytes, session_id: str, sequence: int, message_type: str, payload: str, iv: bytes | None = None) -> dict[str, Any]:
    _require_crypto()
    _token(session_id)
    if sequence < 0:
        raise ValueError("negative sequence")
    iv = iv or os.urandom(12)
    header = {"seq": sequence, "sid": session_id, "t": message_type, "v": VERSION}
    aad = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    ciphertext = AESGCM(key).encrypt(iv, payload.encode(), aad)
    return {
        "ct": b64e(ciphertext),
        "iv": b64e(iv),
        "seq": sequence,
        "sid": session_id,
        "t": message_type,
        "v": VERSION,
    }


def decrypt_frame(key: bytes, frame: dict[str, Any]) -> str:
    _require_crypto()
    header = {"seq": frame["seq"], "sid": frame["sid"], "t": frame["t"], "v": frame["v"]}
    aad = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    return AESGCM(key).decrypt(b64d(frame["iv"]), b64d(frame["ct"]), aad).decode()


@dataclass
class PcSecureChannel:
    """Tiny stateful counterpart to Android ``SecureChannel`` for tests."""

    key: bytes
    session_id: str
    send_sequence: int = 0
    receive_sequence: int = 0

    def encrypt(self, message_type: str, payload: str, *, iv: bytes | None = None) -> dict[str, Any]:
        frame = encrypt_frame(
            self.key,
            self.session_id,
            self.send_sequence,
            message_type,
            payload,
            iv=iv,
        )
        self.send_sequence += 1
        return frame

    def decrypt(self, frame: dict[str, Any]) -> str:
        if frame.get("sid") != self.session_id:
            raise ValueError("session mismatch")
        if frame.get("seq") != self.receive_sequence:
            raise ValueError("replayed or out-of-order frame")
        payload = decrypt_frame(self.key, frame)
        self.receive_sequence += 1
        return payload
