package com.asher.companion.pairing

import android.content.Context
import com.asher.companion.protocol.PairingInvitation
import com.asher.companion.protocol.PairingRequest
import com.asher.companion.protocol.PairingResponse
import com.asher.companion.protocol.Protocol
import com.asher.companion.protocol.SecureChannel
import com.asher.companion.security.CryptoPrimitives
import java.security.KeyPair

enum class PairingState {
    UNPAIRED,
    AWAITING_CONFIRMATION,
    PAIRED,
    REVOKED,
}

/** In-memory half of a pairing handshake; discarded on process death. */
private data class PendingPairing(
    val invitation: PairingInvitation,
    val request: PairingRequest,
    val keyAlias: String,
    val identityKey: KeyPair,
)

/**
 * Coordinates user-confirmed pairing and derives channels from Keystore keys.
 * A pairing invitation is not trusted merely because it arrived over a local
 * network: the human must compare the displayed code on both devices.
 */
class PairingManager(
    context: Context,
    private val store: PairingStore = PairingStore(context.applicationContext),
    private val nowEpochSeconds: () -> Long = { System.currentTimeMillis() / 1_000L },
) {
    private val appContext = context.applicationContext
    private var pending: PendingPairing? = null
    private var state: PairingState = if (store.read() == null) PairingState.UNPAIRED else PairingState.PAIRED

    @Synchronized
    fun state(): PairingState = state

    @Synchronized
    fun currentPeer(): PairedPeer? = store.read()

    /**
     * Validate the invitation and the code shown by the PC, then create the
     * first handshake request. No persistent state is changed until complete.
     */
    @Synchronized
    fun begin(invitation: PairingInvitation, codeShownByPc: String): PairingRequest {
        if (state == PairingState.PAIRED) throw IllegalStateException("A paired device must be revoked first")
        if (nowEpochSeconds() >= invitation.expiresAtEpochSeconds) {
            throw SecurityException("Pairing invitation has expired")
        }
        val normalizedCode = Protocol.normalizePairingCode(codeShownByPc)
        val nonce = Protocol.decodeBase64(invitation.nonce, "nonce")
        val pcPublic = Protocol.decodeBase64(invitation.pcPublicKey, "pc_public_key")
        val expectedCommitment = CryptoPrimitives.codeCommitment(
            invitation.pairingId,
            nonce,
            pcPublic,
            invitation.expiresAtEpochSeconds,
            normalizedCode,
        )
        val suppliedCommitment = Protocol.decodeBase64(invitation.codeCommitment, "code_commitment")
        if (!Protocol.constantTimeEquals(expectedCommitment, suppliedCommitment)) {
            throw SecurityException("Pairing code does not match the PC invitation")
        }

        // Revoke an abandoned key for the same invitation before replacing it.
        pending?.let { CryptoPrimitives.deleteIdentityKey(it.keyAlias) }
        val keyAlias = aliasFor(invitation.pairingId)
        val identity = CryptoPrimitives.getOrCreateIdentityKey(keyAlias)
        val mobilePublic = identity.public.encoded
        val request = PairingRequest(
            pairingId = invitation.pairingId,
            mobilePublicKey = Protocol.encodeBase64(mobilePublic),
            nonce = invitation.nonce,
            invitationDigest = Protocol.encodeBase64(CryptoPrimitives.invitationDigest(invitation.canonicalWithoutCode())),
            codeProof = Protocol.encodeBase64(
                CryptoPrimitives.codeProof(invitation.pairingId, nonce, normalizedCode),
            ),
        )
        pending = PendingPairing(invitation, request, keyAlias, identity)
        state = PairingState.AWAITING_CONFIRMATION
        return request
    }

    /** Complete the handshake only when the PC proves possession of its key. */
    @Synchronized
    fun complete(response: PairingResponse): PairedPeer {
        val current = pending ?: throw IllegalStateException("No pairing handshake is pending")
        val invitation = current.invitation
        if (response.pairingId != invitation.pairingId || response.nonce != invitation.nonce) {
            throw SecurityException("Pairing response is bound to another invitation")
        }
        if (response.pcPublicKey != invitation.pcPublicKey) {
            throw SecurityException("Desktop key changed during pairing")
        }
        val peerPublic = CryptoPrimitives.decodePublicKey(
            Protocol.decodeBase64(response.pcPublicKey, "pc_public_key"),
        )
        val nonce = Protocol.decodeBase64(invitation.nonce, "nonce")
        val sharedSecret = CryptoPrimitives.deriveSharedSecret(current.identityKey.private, peerPublic)
        val channelKey = CryptoPrimitives.deriveChannelKey(sharedSecret, nonce)
        try {
            val expectedMac = CryptoPrimitives.transcriptMac(
                channelKey,
                current.request.toJson(),
                response.canonicalWithoutMac(),
            )
            val suppliedMac = Protocol.decodeBase64(response.transcriptMac, "transcript_mac")
            if (!Protocol.constantTimeEquals(expectedMac, suppliedMac)) {
                throw SecurityException("Desktop did not prove possession of the pairing key")
            }
        } finally {
            CryptoPrimitives.wipe(sharedSecret, channelKey)
        }

        val paired = PairedPeer(
            pairingId = invitation.pairingId,
            keyAlias = current.keyAlias,
            peerPublicKey = response.pcPublicKey,
            nonce = invitation.nonce,
            pairedAtEpochSeconds = nowEpochSeconds(),
        )
        store.write(paired)
        pending = null
        state = PairingState.PAIRED
        return paired
    }

    /** Derive a fresh session key for each transport connection. */
    @Synchronized
    fun openChannel(sessionId: String): SecureChannel {
        val peer = store.read() ?: throw IllegalStateException("No paired desktop")
        val identity = CryptoPrimitives.getIdentityKey(peer.keyAlias)
            ?: throw SecurityException("Pairing identity key is unavailable; revoke and pair again")
        val peerPublic = CryptoPrimitives.decodePublicKey(
            Protocol.decodeBase64(peer.peerPublicKey, "peer_public_key"),
        )
        val nonce = Protocol.decodeBase64(peer.nonce, "nonce")
        val sharedSecret = CryptoPrimitives.deriveSharedSecret(identity.private, peerPublic)
        val channelKey = CryptoPrimitives.deriveChannelKey(sharedSecret, nonce)
        CryptoPrimitives.wipe(sharedSecret)
        return SecureChannel(channelKey, sessionId).also { CryptoPrimitives.wipe(channelKey) }
    }

    @Synchronized
    fun revoke() {
        val peer = store.read()
        peer?.let { CryptoPrimitives.deleteIdentityKey(it.keyAlias) }
        pending?.let { CryptoPrimitives.deleteIdentityKey(it.keyAlias) }
        pending = null
        store.clear()
        state = PairingState.REVOKED
    }

    private fun aliasFor(pairingId: String): String {
        val digest = CryptoPrimitives.sha256(Protocol.utf8(pairingId))
        return "asher_pair_${digest.take(12).joinToString("") { "%02x".format(it) }}"
    }
}
