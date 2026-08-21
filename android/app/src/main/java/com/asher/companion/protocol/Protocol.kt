package com.asher.companion.protocol

import android.util.Base64
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

/**
 * Wire-level constants and strict, dependency-light message models.
 *
 * The PC implementation should treat every field as untrusted.  Models reject
 * an unsupported version, malformed base64, oversized values, and impossible
 * sequence numbers before any cryptographic operation is attempted.
 */
object Protocol {
    const val VERSION: Int = 1
    const val MAX_FRAME_BYTES: Int = 1_048_576
    const val MAX_PAYLOAD_BYTES: Int = 524_288
    const val GCM_IV_BYTES: Int = 12
    const val GCM_TAG_BITS: Int = 128
    const val ECDH_PUBLIC_KEY_MIN_BYTES: Int = 50
    const val ECDH_PUBLIC_KEY_MAX_BYTES: Int = 256
    const val PAIRING_CODE_MIN_LENGTH: Int = 6
    const val PAIRING_CODE_MAX_LENGTH: Int = 32
    const val HKDF_INFO: String = "asher/companion/channel/v1"
    const val CODE_COMMITMENT_DOMAIN: String = "asher/companion/pairing-code/v1"
    const val CODE_PROOF_DOMAIN: String = "asher/companion/pairing-proof/v1"

    fun encodeBase64(value: ByteArray): String =
        Base64.encodeToString(value, Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING)

    fun decodeBase64(value: String, field: String): ByteArray {
        if (value.isEmpty() || value.length > MAX_FRAME_BYTES ||
            !value.matches(Regex("[A-Za-z0-9_-]+"))
        ) {
            throw ProtocolException("$field has an invalid length")
        }
        return try {
            val decoded = Base64.decode(value, Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING)
            // Reject alternate encodings (padding/ignored characters) so the
            // same transcript cannot have multiple byte representations.
            if (encodeBase64(decoded) != value) throw ProtocolException("$field is not canonical base64")
            decoded
        } catch (error: IllegalArgumentException) {
            throw ProtocolException("$field is not valid base64", error)
        }
    }

    fun requireVersion(version: Int) {
        if (version != VERSION) throw ProtocolException("Unsupported protocol version: $version")
    }

    fun normalizePairingCode(value: String): String {
        // Spaces and separators are presentation-only. Restrict the alphabet
        // so a copied code cannot smuggle control characters into transcripts.
        val normalized = value.filter { it.isLetterOrDigit() }.uppercase()
        if (normalized.length !in PAIRING_CODE_MIN_LENGTH..PAIRING_CODE_MAX_LENGTH ||
            normalized.any { it !in 'A'..'Z' && it !in '0'..'9' }
        ) {
            throw ProtocolException("Pairing code must contain 6–32 letters or digits")
        }
        return normalized
    }

    fun constantTimeEquals(left: ByteArray, right: ByteArray): Boolean =
        MessageDigest.isEqual(left, right)

    fun utf8(value: String): ByteArray = value.toByteArray(StandardCharsets.UTF_8)

    fun boundedUtf8(value: String, field: String, maxBytes: Int = MAX_PAYLOAD_BYTES): ByteArray {
        val bytes = utf8(value)
        if (bytes.isEmpty() || bytes.size > maxBytes) {
            throw ProtocolException("$field exceeds the protocol size limit")
        }
        return bytes
    }
}

class ProtocolException(message: String, cause: Throwable? = null) : Exception(message, cause)

enum class MessageType(val wireValue: String) {
    PAIRING_REQUEST("pairing_request"),
    PAIRING_RESPONSE("pairing_response"),
    COMMAND("command"),
    COMMAND_RESULT("command_result"),
    EVENT("event"),
    PING("ping"),
    PONG("pong"),
    REVOKE("revoke");

    companion object {
        fun fromWire(value: String): MessageType = entries.firstOrNull { it.wireValue == value }
            ?: throw ProtocolException("Unknown message type")
    }
}

/** A short-lived invitation displayed by the desktop during pairing. */
data class PairingInvitation(
    val pairingId: String,
    val pcPublicKey: String,
    val nonce: String,
    val codeCommitment: String,
    val expiresAtEpochSeconds: Long,
    val version: Int = Protocol.VERSION,
) {
    init {
        Protocol.requireVersion(version)
        requireToken(pairingId, "pairing_id", 128)
        requireEncodedKey(pcPublicKey, "pc_public_key")
        requireEncodedBytes(nonce, "nonce", 16, 64)
        requireEncodedBytes(codeCommitment, "code_commitment", 16, 64)
        if (expiresAtEpochSeconds <= 0L) throw ProtocolException("Pairing invitation expiry is invalid")
    }

    fun canonicalWithoutCode(): String = buildString {
        append('{')
        append("\"code_commitment\":").append(quote(codeCommitment)).append(',')
        append("\"expires_at\":").append(expiresAtEpochSeconds).append(',')
        append("\"nonce\":").append(quote(nonce)).append(',')
        append("\"pairing_id\":").append(quote(pairingId)).append(',')
        append("\"pc_public_key\":").append(quote(pcPublicKey)).append(',')
        append("\"v\":").append(version)
        append('}')
    }

    fun toJson(): String = canonicalWithoutCode()

    fun toDeepLink(): String = "asher://pair?payload=${Protocol.encodeBase64(Protocol.utf8(toJson()))}"

    companion object {
        fun fromJson(json: String): PairingInvitation = try {
            val objectValue = org.json.JSONObject(json)
            PairingInvitation(
                pairingId = objectValue.getString("pairing_id"),
                pcPublicKey = objectValue.getString("pc_public_key"),
                nonce = objectValue.getString("nonce"),
                codeCommitment = objectValue.getString("code_commitment"),
                expiresAtEpochSeconds = objectValue.getLong("expires_at"),
                version = objectValue.getInt("v"),
            )
        } catch (error: ProtocolException) {
            throw error
        } catch (error: Exception) {
            throw ProtocolException("Malformed pairing invitation", error)
        }
    }
}

/** First message sent by Android after the human compares the pairing code. */
data class PairingRequest(
    val pairingId: String,
    val mobilePublicKey: String,
    val nonce: String,
    val invitationDigest: String,
    val codeProof: String,
    val version: Int = Protocol.VERSION,
) {
    init {
        Protocol.requireVersion(version)
        requireToken(pairingId, "pairing_id", 128)
        requireEncodedKey(mobilePublicKey, "mobile_public_key")
        requireEncodedBytes(nonce, "nonce", 16, 64)
        requireEncodedBytes(invitationDigest, "invitation_digest", 16, 64)
        requireEncodedBytes(codeProof, "code_proof", 16, 64)
    }

    fun canonicalWithoutProof(): String = buildString {
        append('{')
        append("\"invitation_digest\":").append(quote(invitationDigest)).append(',')
        append("\"mobile_public_key\":").append(quote(mobilePublicKey)).append(',')
        append("\"nonce\":").append(quote(nonce)).append(',')
        append("\"pairing_id\":").append(quote(pairingId)).append(',')
        append("\"v\":").append(version)
        append('}')
    }

    fun toJson(): String = buildString {
        append('{')
        append("\"code_proof\":").append(quote(codeProof)).append(',')
        append("\"invitation_digest\":").append(quote(invitationDigest)).append(',')
        append("\"mobile_public_key\":").append(quote(mobilePublicKey)).append(',')
        append("\"nonce\":").append(quote(nonce)).append(',')
        append("\"pairing_id\":").append(quote(pairingId)).append(',')
        append("\"v\":").append(version)
        append('}')
    }

    companion object {
        fun fromJson(json: String): PairingRequest = try {
            val value = org.json.JSONObject(json)
            PairingRequest(
                pairingId = value.getString("pairing_id"),
                mobilePublicKey = value.getString("mobile_public_key"),
                nonce = value.getString("nonce"),
                invitationDigest = value.getString("invitation_digest"),
                codeProof = value.getString("code_proof"),
                version = value.getInt("v"),
            )
        } catch (error: ProtocolException) {
            throw error
        } catch (error: Exception) {
            throw ProtocolException("Malformed pairing request", error)
        }
    }
}

/** Desktop response proves possession of the private desktop key. */
data class PairingResponse(
    val pairingId: String,
    val pcPublicKey: String,
    val nonce: String,
    val transcriptMac: String,
    val version: Int = Protocol.VERSION,
) {
    init {
        Protocol.requireVersion(version)
        requireToken(pairingId, "pairing_id", 128)
        requireEncodedKey(pcPublicKey, "pc_public_key")
        requireEncodedBytes(nonce, "nonce", 16, 64)
        requireEncodedBytes(transcriptMac, "transcript_mac", 32, 64)
    }

    fun canonicalWithoutMac(): String = buildString {
        append('{')
        append("\"nonce\":").append(quote(nonce)).append(',')
        append("\"pairing_id\":").append(quote(pairingId)).append(',')
        append("\"pc_public_key\":").append(quote(pcPublicKey)).append(',')
        append("\"v\":").append(version)
        append('}')
    }

    fun toJson(): String = buildString {
        append('{')
        append("\"nonce\":").append(quote(nonce)).append(',')
        append("\"pairing_id\":").append(quote(pairingId)).append(',')
        append("\"pc_public_key\":").append(quote(pcPublicKey)).append(',')
        append("\"transcript_mac\":").append(quote(transcriptMac)).append(',')
        append("\"v\":").append(version)
        append('}')
    }

    companion object {
        fun fromJson(json: String): PairingResponse = try {
            val value = org.json.JSONObject(json)
            PairingResponse(
                pairingId = value.getString("pairing_id"),
                pcPublicKey = value.getString("pc_public_key"),
                nonce = value.getString("nonce"),
                transcriptMac = value.getString("transcript_mac"),
                version = value.getInt("v"),
            )
        } catch (error: ProtocolException) {
            throw error
        } catch (error: Exception) {
            throw ProtocolException("Malformed pairing response", error)
        }
    }
}

data class FrameHeader(
    val type: MessageType,
    val sessionId: String,
    val sequence: Long,
    val version: Int = Protocol.VERSION,
) {
    init {
        Protocol.requireVersion(version)
        requireToken(sessionId, "session_id", 128)
        if (sequence < 0L) throw ProtocolException("Negative frame sequence")
    }

    /** Deterministic AAD representation shared by Android and the PC mock. */
    fun canonical(): String = buildString {
        append('{')
        append("\"seq\":").append(sequence).append(',')
        append("\"sid\":").append(quote(sessionId)).append(',')
        append("\"t\":").append(quote(type.wireValue)).append(',')
        append("\"v\":").append(version)
        append('}')
    }
}

data class EncryptedFrame(
    val header: FrameHeader,
    val iv: String,
    val ciphertext: String,
) {
    init {
        val ivBytes = Protocol.decodeBase64(iv, "iv")
        if (ivBytes.size != Protocol.GCM_IV_BYTES) throw ProtocolException("Invalid GCM IV length")
        val encrypted = Protocol.decodeBase64(ciphertext, "ciphertext")
        if (encrypted.size < Protocol.GCM_TAG_BITS / 8 || encrypted.size > Protocol.MAX_FRAME_BYTES) {
            throw ProtocolException("Invalid ciphertext length")
        }
    }

    fun toJson(): String = buildString {
        append('{')
        append("\"ct\":").append(quote(ciphertext)).append(',')
        append("\"iv\":").append(quote(iv)).append(',')
        append("\"seq\":").append(header.sequence).append(',')
        append("\"sid\":").append(quote(header.sessionId)).append(',')
        append("\"t\":").append(quote(header.type.wireValue)).append(',')
        append("\"v\":").append(header.version)
        append('}')
    }

    fun encodedSize(): Int = toJson().toByteArray(StandardCharsets.UTF_8).size

    companion object {
        fun fromJson(json: String): EncryptedFrame {
            if (json.toByteArray(StandardCharsets.UTF_8).size > Protocol.MAX_FRAME_BYTES) {
                throw ProtocolException("Encrypted frame is too large")
            }
            return try {
                val value = org.json.JSONObject(json)
                EncryptedFrame(
                    header = FrameHeader(
                        type = MessageType.fromWire(value.getString("t")),
                        sessionId = value.getString("sid"),
                        sequence = value.getLong("seq"),
                        version = value.getInt("v"),
                    ),
                    iv = value.getString("iv"),
                    ciphertext = value.getString("ct"),
                )
            } catch (error: ProtocolException) {
                throw error
            } catch (error: Exception) {
                throw ProtocolException("Malformed encrypted frame", error)
            }
        }
    }
}

private fun quote(value: String): String = org.json.JSONObject.quote(value)

private fun requireToken(value: String, field: String, maxLength: Int) {
    if (value.isBlank() || value.length > maxLength ||
        !value.matches(Regex("[A-Za-z0-9._~:-]+"))
    ) throw ProtocolException("Invalid $field")
}

private fun requireEncodedBytes(value: String, field: String, min: Int, max: Int) {
    val bytes = Protocol.decodeBase64(value, field)
    if (bytes.size !in min..max) throw ProtocolException("Invalid $field length")
}

private fun requireEncodedKey(value: String, field: String) =
    requireEncodedBytes(value, field, Protocol.ECDH_PUBLIC_KEY_MIN_BYTES, Protocol.ECDH_PUBLIC_KEY_MAX_BYTES)
