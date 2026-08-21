package com.asher.companion.security

import com.asher.companion.protocol.Protocol
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets
import java.security.KeyFactory
import java.security.KeyPair
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.MessageDigest
import java.security.PrivateKey
import java.security.PublicKey
import java.security.SecureRandom
import java.security.spec.ECGenParameterSpec
import java.security.spec.X509EncodedKeySpec
import javax.crypto.Cipher
import javax.crypto.KeyAgreement
import javax.crypto.Mac
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties

/** Small crypto facade so protocol code has one audited implementation. */
object CryptoPrimitives {
    private const val KEYSTORE = "AndroidKeyStore"
    private const val CURVE = "secp256r1"
    private const val HASH = "SHA-256"
    private const val HMAC = "HmacSHA256"
    private val secureRandom = SecureRandom()

    fun randomBytes(size: Int): ByteArray {
        require(size > 0) { "size must be positive" }
        return ByteArray(size).also(secureRandom::nextBytes)
    }

    /** Generate or retrieve a non-exportable identity key in Android Keystore. */
    @Synchronized
    fun getOrCreateIdentityKey(alias: String): KeyPair {
        require(alias.matches(Regex("[A-Za-z0-9._-]{1,120}"))) { "Invalid keystore alias" }
        val store = KeyStore.getInstance(KEYSTORE).apply { load(null) }
        val existing = if (store.containsAlias(alias)) {
            val privateKey = store.getKey(alias, null) as? PrivateKey
            val certificate = store.getCertificate(alias)
            if (privateKey != null && certificate != null) KeyPair(certificate.publicKey, privateKey) else null
        } else null
        if (existing != null) return existing

        val generator = KeyPairGenerator.getInstance(KeyProperties.KEY_ALGORITHM_EC, KEYSTORE)
        generator.initialize(
            KeyGenParameterSpec.Builder(alias, KeyProperties.PURPOSE_AGREE_KEY)
                .setAlgorithmParameterSpec(ECGenParameterSpec(CURVE))
                .setDigests(KeyProperties.DIGEST_SHA256, KeyProperties.DIGEST_SHA512)
                .build(),
        )
        return generator.generateKeyPair()
    }

    fun getIdentityKey(alias: String): KeyPair? {
        val store = KeyStore.getInstance(KEYSTORE).apply { load(null) }
        if (!store.containsAlias(alias)) return null
        val privateKey = store.getKey(alias, null) as? PrivateKey ?: return null
        val certificate = store.getCertificate(alias) ?: return null
        return KeyPair(certificate.publicKey, privateKey)
    }

    fun deleteIdentityKey(alias: String) {
        val store = KeyStore.getInstance(KEYSTORE).apply { load(null) }
        if (store.containsAlias(alias)) store.deleteEntry(alias)
    }

    fun generateEphemeralKeyPair(): KeyPair =
        KeyPairGenerator.getInstance("EC").apply {
            initialize(ECGenParameterSpec(CURVE), secureRandom)
        }.generateKeyPair()

    fun decodePublicKey(encoded: ByteArray): PublicKey {
        if (encoded.size !in Protocol.ECDH_PUBLIC_KEY_MIN_BYTES..Protocol.ECDH_PUBLIC_KEY_MAX_BYTES) {
            throw SecurityException("Peer public key has an invalid size")
        }
        return try {
            KeyFactory.getInstance("EC").generatePublic(X509EncodedKeySpec(encoded))
        } catch (error: Exception) {
            throw SecurityException("Peer public key is invalid", error)
        }
    }

    fun deriveSharedSecret(privateKey: PrivateKey, peerPublicKey: PublicKey): ByteArray {
        return try {
            KeyAgreement.getInstance("ECDH").apply {
                init(privateKey)
                doPhase(peerPublicKey, true)
            }.generateSecret()
        } catch (error: Exception) {
            throw SecurityException("ECDH negotiation failed", error)
        }
    }

    /** RFC 5869 HKDF-SHA-256. */
    fun hkdfSha256(ikm: ByteArray, salt: ByteArray, info: ByteArray, length: Int): ByteArray {
        require(length in 1..255 * 32) { "HKDF output length is invalid" }
        val actualSalt = if (salt.isEmpty()) ByteArray(32) else salt
        val prk = hmac(actualSalt, ikm)
        val result = ByteArray(length)
        var previous = ByteArray(0)
        var offset = 0
        var counter = 1
        while (offset < length) {
            previous = hmac(prk, previous + info + byteArrayOf(counter.toByte()))
            val copyLength = minOf(previous.size, length - offset)
            previous.copyInto(result, offset, 0, copyLength)
            offset += copyLength
            counter += 1
        }
        prk.fill(0)
        previous.fill(0)
        return result
    }

    fun deriveChannelKey(sharedSecret: ByteArray, nonce: ByteArray): ByteArray = hkdfSha256(
        ikm = sharedSecret,
        salt = nonce,
        info = Protocol.utf8(Protocol.HKDF_INFO),
        length = 32,
    )

    fun hmac(key: ByteArray, message: ByteArray): ByteArray =
        Mac.getInstance(HMAC).apply { init(SecretKeySpec(key, HMAC)) }.doFinal(message)

    fun sha256(message: ByteArray): ByteArray = MessageDigest.getInstance(HASH).digest(message)

    fun invitationDigest(canonicalInvitation: String): ByteArray =
        sha256(Protocol.utf8("${Protocol.CODE_COMMITMENT_DOMAIN}|$canonicalInvitation"))

    fun codeCommitment(
        pairingId: String,
        nonce: ByteArray,
        pcPublicKey: ByteArray,
        expiresAtEpochSeconds: Long,
        code: String,
    ): ByteArray {
        require(expiresAtEpochSeconds > 0L) { "Pairing expiry must be positive" }
        val normalized = Protocol.normalizePairingCode(code)
        val input = buildBytes(
            Protocol.CODE_COMMITMENT_DOMAIN,
            pairingId,
            Protocol.encodeBase64(nonce),
            Protocol.encodeBase64(pcPublicKey),
            expiresAtEpochSeconds.toString(),
            normalized,
        )
        return sha256(input)
    }

    fun codeProof(pairingId: String, nonce: ByteArray, code: String): ByteArray {
        val normalized = Protocol.normalizePairingCode(code)
        return sha256(buildBytes(Protocol.CODE_PROOF_DOMAIN, pairingId, Protocol.encodeBase64(nonce), normalized))
    }

    fun transcriptMac(channelKey: ByteArray, request: String, responseWithoutMac: String): ByteArray =
        hmac(channelKey, Protocol.utf8("$request\n$responseWithoutMac"))

    fun fingerprint(encodedPublicKey: ByteArray): String =
        sha256(encodedPublicKey).joinToString(":") { "%02x".format(it) }

    fun encrypt(key: ByteArray, iv: ByteArray, aad: ByteArray, plaintext: ByteArray): ByteArray {
        require(iv.size == Protocol.GCM_IV_BYTES) { "Invalid GCM IV" }
        return Cipher.getInstance("AES/GCM/NoPadding").apply {
            init(Cipher.ENCRYPT_MODE, SecretKeySpec(key, "AES"), GCMParameterSpec(Protocol.GCM_TAG_BITS, iv))
            updateAAD(aad)
        }.doFinal(plaintext)
    }

    fun decrypt(key: ByteArray, iv: ByteArray, aad: ByteArray, ciphertext: ByteArray): ByteArray {
        require(iv.size == Protocol.GCM_IV_BYTES) { "Invalid GCM IV" }
        return try {
            Cipher.getInstance("AES/GCM/NoPadding").apply {
                init(Cipher.DECRYPT_MODE, SecretKeySpec(key, "AES"), GCMParameterSpec(Protocol.GCM_TAG_BITS, iv))
                updateAAD(aad)
            }.doFinal(ciphertext)
        } catch (error: Exception) {
            // Do not reveal whether authentication or parsing failed to a peer.
            throw SecurityException("Encrypted frame authentication failed")
        }
    }

    fun wipe(vararg values: ByteArray?) {
        values.forEach { it?.fill(0) }
    }

    private fun buildBytes(vararg values: String): ByteArray {
        val output = values.joinToString("|").toByteArray(StandardCharsets.UTF_8)
        // Prefixing the length prevents concatenation ambiguity if a future
        // field is allowed to contain separators.
        return ByteBuffer.allocate(4 + output.size).putInt(output.size).put(output).array()
    }
}
