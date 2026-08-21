package com.asher.companion.security

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

/** Explicit local capability grants, encrypted at rest and revocable at any time. */
class CapabilityGrantStore(context: Context) {
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
    fun read(): Set<CompanionCapability> {
        val encoded = preferences.getString(KEY_GRANTS, null) ?: return PermissionPolicy.DEFAULT_GRANTS
        return encoded.split(',')
            .asSequence()
            .filter { it.isNotBlank() }
            .mapNotNull { value -> runCatching { CompanionCapability.valueOf(value) }.getOrNull() }
            .toSet()
            .plus(CompanionCapability.PAIR)
    }

    @Synchronized
    fun write(grants: Set<CompanionCapability>) {
        val normalized = grants.plus(CompanionCapability.PAIR)
            .map(CompanionCapability::name)
            .sorted()
            .joinToString(",")
        check(preferences.edit().putString(KEY_GRANTS, normalized).commit()) {
            "Unable to persist capability grants"
        }
    }

    @Synchronized
    fun clear() {
        check(preferences.edit().clear().commit()) { "Unable to clear capability grants" }
    }

    private companion object {
        const val FILE_NAME = "asher_capabilities_secure"
        const val KEY_GRANTS = "grants"
    }
}
