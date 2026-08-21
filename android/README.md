# ASHER Android companion

This directory is a standalone Android/Kotlin companion scaffold. It is
deliberately transport-agnostic: a future Wi-Fi, Bluetooth, or USB adapter can
implement the small `SecureChannel` interface without moving keys into the UI.

## Build

From this directory, with Android SDK (API 35), JDK 17, and Gradle 8.7+:

```text
gradle :app:testDebugUnitTest
gradle :app:assembleDebug
```

The current development machine does not have Java, Gradle, ADB, or an Android
SDK, so emulator, biometric, and APK verification remain explicitly
unperformed. The project files use standard Android Gradle Plugin conventions
and are ready to build once that toolchain is installed.

## Pairing protocol (version 1)

1. The PC creates a short-lived invitation containing `pairing_id`, a P-256
   public key, a random nonce, an expiry, and a code commitment. It displays a
human-checkable code next to the QR/deep link (`asher://pair?...`). The
commitment covers the pairing id, key, nonce, expiry, and code.
2. The Android user scans/pastes the invitation and enters the code visible on
   the PC. Android recomputes the commitment and refuses a mismatch or expired
   invitation. This comparison is the out-of-band MITM check; accepting a QR
   without comparing the code is not safe.
3. Android creates a non-exportable P-256 identity key in `AndroidKeyStore` and
   sends a `pairing_request` containing its public key, invitation digest, and
   a code proof. The private key never enters JSON, logs, preferences, or the
   transport.
4. The PC responds with its invitation key and a transcript MAC. Both sides
   derive `HKDF-SHA-256(ECDH, nonce, "asher/companion/channel/v1")` and retain
   only public peer metadata plus the Keystore alias on Android.
5. Each transport connection gets a fresh session id and a `SecureChannel`.
   Payloads are encrypted with `AES-256-GCM`; the canonical frame header is
   authenticated as AAD. A four-byte big-endian length prefix frames JSON.
   Sequence numbers start at zero and must increase exactly by one. Replays,
   gaps, wrong sessions, oversized frames, bad tags, and unsupported versions
   are rejected without advancing the receive counter.

Wire keys are intentionally short and stable:

| Object | Keys |
| --- | --- |
| Invitation | `v`, `pairing_id`, `pc_public_key`, `nonce`, `code_commitment`, `expires_at` |
| Request | `v`, `pairing_id`, `mobile_public_key`, `nonce`, `invitation_digest`, `code_proof` |
| Response | `v`, `pairing_id`, `pc_public_key`, `nonce`, `transcript_mac` |
| Frame | `v`, `t`, `sid`, `seq`, `iv`, `ct` |

The canonical AAD/header ordering is:

```json
{"seq":0,"sid":"session","t":"command","v":1}
```

Implementations must use URL-safe, unpadded Base64 for all binary fields and
UTF-8 JSON. Unknown message types and protocol versions fail closed.

## Authorization model

Pairing is local and code-confirmed. Low-risk status/conversation messages are
available only after pairing. Private memory, revocation, and all consequential
remote actions require a fresh Android biometric or device-credential result;
commands additionally require an exact local confirmation. A successful prompt
returns a one-shot, 60-second grant token; consuming it is required before a
command is sent. Requests marked as voice-only are denied. Android runtime permissions (Bluetooth, notifications,
and calling) are separate from these capability grants and are requested only
when the user opts into the relevant feature.

`EncryptedSharedPreferences` stores pairing metadata and capability grants;
private key material remains in `AndroidKeyStore`. `network_security_config.xml`
disables cleartext traffic and trusts system certificates only. Revoke pairing
deletes the Keystore key and encrypted metadata.

## Host transport contract

The app does not open a socket implicitly. A host adapter should:

- establish a TLS 1.3 transport with certificate/public-key pinning where the PC
  endpoint supports it;
- feed exactly one `FrameCodec` length-prefixed frame at a time to
  `SecureChannel.decrypt`;
- create a new channel/session after reconnect (never reset sequence numbers on
  a live channel);
- stop forwarding commands when `PermissionPolicy` returns a denial; and
- erase channel state on logout, revoke, or authentication failure.

The protocol layer is intentionally usable with a PC mock for deterministic
tests without granting the Android app network or device-control side effects.
