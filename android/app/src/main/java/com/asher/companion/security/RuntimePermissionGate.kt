package com.asher.companion.security

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.content.ContextCompat

/**
 * Runtime permissions are deliberately separate from ASHER capability grants.
 * A granted Android permission never implies that a remote command is allowed.
 */
object RuntimePermissionGate {
    fun missing(context: Context, permissions: Set<String>): Set<String> = permissions.filterTo(mutableSetOf()) {
        ContextCompat.checkSelfPermission(context, it) != PackageManager.PERMISSION_GRANTED
    }

    fun forCapability(capability: CompanionCapability): Set<String> = when (capability) {
        CompanionCapability.PHONE_CALL -> setOf(Manifest.permission.CALL_PHONE)
        else -> emptySet()
    }

    fun forNearbyTransport(): Set<String> = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        setOf(Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT)
    } else {
        emptySet()
    }

    fun forNotifications(): Set<String> = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
        setOf(Manifest.permission.POST_NOTIFICATIONS)
    } else {
        emptySet()
    }
}
