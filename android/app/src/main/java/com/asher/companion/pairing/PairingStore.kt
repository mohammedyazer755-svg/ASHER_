package com.asher.companion.pairing

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import com.asher.companion.protocol.Protocol

/**
 * Stores only public peer material and metadata. The Android private key lives
 * in Android Keystore and is never copied into preferences or a protocol frame.
 */
data class PairedPeer(
    val pairingId: String,
    val keyAlias: String,
    val peerPublicKey: String,
    val nonce: String,
    val pairedAtEpochSeconds: Long,
    val version: Int = Protocol.VERSION,
) {
    init {
        Protocol.requireVersion(version)
        if (pairingId.isBlank() || keyAlias.isBlank() || pairedAtEpochSeconds <= 0L) {
            throw IllegalArgumentException("Invalid paired peer metadata")
        }
        Protocol.decodeBase64(peerPublicKey, "peer_public_key")
        Protocol.decodeBase64(nonce, "nonce")
    }
}

class PairingStore(context: Context) {
    private val preferences: SharedPreferences

    init {
        val masterKey = MasterKey.Builder(context.applicationContext)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
        preferences = EncryptedSharedPreferences.create(
            context.applicationContext,
            FILE_NAME,
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
        )
    }

    @Synchronized
    fun read(): PairedPeer? {
        val pairingId = preferences.getString(KEY_PAIRING_ID, null) ?: return null
        val keyAlias = preferences.getString(KEY_KEY_ALIAS, null) ?: return null
        val peerKey = preferences.getString(KEY_PEER_KEY, null) ?: return null
        val nonce = preferences.getString(KEY_NONCE, null) ?: return null
        val pairedAt = preferences.getLong(KEY_PAIRED_AT, 0L)
        val version = preferences.getInt(KEY_VERSION, Protocol.VERSION)
        return try {
            PairedPeer(pairingId, keyAlias, peerKey, nonce, pairedAt, version)
        } catch (_: Exception) {
            // Corrupt state is treated as unpaired; never try to "repair" it
            // by generating a new key without explicit user action.
            null
        }
    }

    @Synchronized
    fun write(peer: PairedPeer) {
        preferences.edit()
            .putString(KEY_PAIRING_ID, peer.pairingId)
            .putString(KEY_KEY_ALIAS, peer.keyAlias)
            .putString(KEY_PEER_KEY, peer.peerPublicKey)
            .putString(KEY_NONCE, peer.nonce)
            .putLong(KEY_PAIRED_AT, peer.pairedAtEpochSeconds)
            .putInt(KEY_VERSION, peer.version)
            .commit()
            .also { committed ->
                if (!committed) throw IllegalStateException("Unable to persist pairing state")
            }
    }

    @Synchronized
    fun clear() {
        if (!preferences.edit().clear().commit()) {
            throw IllegalStateException("Unable to clear pairing state")
        }
    }

    private companion object {
        const val FILE_NAME = "asher_pairing_secure"
        const val KEY_PAIRING_ID = "pairing_id"
        const val KEY_KEY_ALIAS = "key_alias"
        const val KEY_PEER_KEY = "peer_public_key"
        const val KEY_NONCE = "nonce"
        const val KEY_PAIRED_AT = "paired_at"
        const val KEY_VERSION = "version"
    }
}
