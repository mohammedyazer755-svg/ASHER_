package com.asher.companion

import android.app.Application
import com.asher.companion.pairing.PairingManager
import com.asher.companion.security.BiometricGate
import com.asher.companion.security.CapabilityGrantStore
import com.asher.companion.security.PermissionPolicy

class AsherApplication : Application() {
    lateinit var pairingManager: PairingManager
        private set
    lateinit var biometricGate: BiometricGate
        private set
    lateinit var permissionPolicy: PermissionPolicy
        private set
    lateinit var capabilityStore: CapabilityGrantStore
        private set

    override fun onCreate() {
        super.onCreate()
        capabilityStore = CapabilityGrantStore(this)
        permissionPolicy = PermissionPolicy(capabilityStore.read()) { updated -> capabilityStore.write(updated) }
        pairingManager = PairingManager(this)
        biometricGate = BiometricGate(this)
    }
}
