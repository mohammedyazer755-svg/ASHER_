"""Environment-independent checks for the Android companion wire contract.

These tests exercise the PC reference mock, not an Android emulator. They keep
the pairing transcript and AEAD framing deterministic enough to catch accidental
wire-format drift before an Android toolchain is available.
"""

from __future__ import annotations

import copy
import unittest

from android.pc_mock.protocol import (
    VERSION,
    b64d,
    b64e,
    code_proof,
    complete_request,
    decrypt_frame,
    encrypt_frame,
    invitation_digest,
    make_invitation,
    PcSecureChannel,
)
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:  # pragma: no cover - test is skipped on minimal installs
    serialization = ec = None  # type: ignore[assignment]
    CRYPTOGRAPHY_AVAILABLE = False


class AndroidProtocolTests(unittest.TestCase):
    @unittest.skipUnless(CRYPTOGRAPHY_AVAILABLE, "cryptography is not installed")
    def test_pc_invitation_and_request_transcript(self) -> None:
        invitation = make_invitation(pairing_id="pair-1", code="ASHER-42")
        self.assertEqual(VERSION, invitation.payload["v"])
        self.assertEqual(invitation_digest(invitation.payload), invitation_digest(invitation.payload))

        mobile_private = ec.generate_private_key(ec.SECP256R1())
        mobile_public = mobile_private.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        request = {
            "code_proof": "",  # filled using the PC's public helper below
            "invitation_digest": "",
            "mobile_public_key": b64e(mobile_public),
            "nonce": invitation.payload["nonce"],
            "pairing_id": invitation.payload["pairing_id"],
            "v": VERSION,
        }
        request["code_proof"] = b64e(code_proof(invitation.payload["pairing_id"], b64d(request["nonce"]), invitation.code))
        request["invitation_digest"] = b64e(invitation_digest(invitation.payload))
        response, channel_key = complete_request(invitation, request)
        self.assertEqual(invitation.payload["pc_public_key"], response["pc_public_key"])
        self.assertEqual(32, len(channel_key))

        altered = copy.deepcopy(invitation.payload)
        altered["expires_at"] += 1
        # A desktop must regenerate the commitment when any invitation field
        # changes; Android therefore rejects this altered payload.
        from android.pc_mock.protocol import code_commitment

        self.assertNotEqual(
            invitation.payload["code_commitment"],
            b64e(
                code_commitment(
                    altered["pairing_id"],
                    b64d(altered["nonce"]),
                    b64d(altered["pc_public_key"]),
                    altered["expires_at"],
                    invitation.code,
                )
            ),
        )

    @unittest.skipUnless(CRYPTOGRAPHY_AVAILABLE, "cryptography is not installed")
    def test_frame_authentication_and_tamper_detection(self) -> None:
        key = b"K" * 32
        frame = encrypt_frame(key, "session", 0, "command", '{"ok":true}', iv=b"I" * 12)
        self.assertEqual('{"ok":true}', decrypt_frame(key, frame))
        sender = PcSecureChannel(key, "session")
        receiver = PcSecureChannel(key, "session")
        self.assertEqual('{"ok":true}', receiver.decrypt(sender.encrypt("command", '{"ok":true}', iv=b"J" * 12)))
        with self.assertRaises(ValueError):
            receiver.decrypt(frame)
        tampered = copy.deepcopy(frame)
        tampered["ct"] = tampered["ct"][:-1] + ("A" if tampered["ct"][-1] != "A" else "B")
        with self.assertRaises(Exception):
            decrypt_frame(key, tampered)


if __name__ == "__main__":
    unittest.main()
