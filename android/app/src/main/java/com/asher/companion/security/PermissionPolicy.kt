package com.asher.companion.security

import android.Manifest
import android.os.Build

enum class CompanionCapability {
    PAIR,
    READ_STATUS,
    CONVERSATION,
    READ_PRIVATE_MEMORY,
    REMOTE_COMMAND,
    EXTERNAL_MESSAGE,
    PHONE_CALL,
    DEVICE_SETTINGS,
    REVOKE_PAIRING,
}

enum class AuthenticationRequirement {
    NONE,
    LOCAL_CONFIRMATION,
    BIOMETRIC_OR_DEVICE_CREDENTIAL,
}

enum class RequestSource {
    LOCAL_UI,
    REMOTE_PC,
    VOICE,
}

data class AuthorizationContext(
    val paired: Boolean,
    val userConfirmed: Boolean = false,
    val biometricVerified: Boolean = false,
    val source: RequestSource = RequestSource.REMOTE_PC,
)

data class PermissionDecision(
    val allowed: Boolean,
    val requirement: AuthenticationRequirement,
    val reason: String,
) {
    companion object {
        fun deny(reason: String, requirement: AuthenticationRequirement = AuthenticationRequirement.NONE) =
            PermissionDecision(false, requirement, reason)
    }
}

/**
 * Android-side capability boundary. The PC may propose a command, but this
 * policy is evaluated locally before anything sensitive is sent to a tool.
 */
class PermissionPolicy(
    initialGrants: Set<CompanionCapability> = DEFAULT_GRANTS,
    private val onChange: (Set<CompanionCapability>) -> Unit = {},
) {
    private val grants = initialGrants.toMutableSet()

    @Synchronized
    fun grant(capability: CompanionCapability) {
        if (capability != CompanionCapability.PAIR) {
            grants += capability
            onChange(grants.toSet())
        }
    }

    @Synchronized
    fun revoke(capability: CompanionCapability) {
        if (capability != CompanionCapability.PAIR) {
            grants -= capability
            onChange(grants.toSet())
        }
    }

    @Synchronized
    fun revokeAll() {
        grants.clear()
        onChange(emptySet())
    }

    @Synchronized
    fun grantedCapabilities(): Set<CompanionCapability> = grants.toSet()

    @Synchronized
    fun evaluate(capability: CompanionCapability, context: AuthorizationContext): PermissionDecision {
        if (capability == CompanionCapability.PAIR) {
            return if (context.source == RequestSource.LOCAL_UI && context.userConfirmed) {
                PermissionDecision(true, AuthenticationRequirement.LOCAL_CONFIRMATION, "Pairing was confirmed locally")
            } else {
                PermissionDecision.deny("Pairing must be initiated and confirmed on this device")
            }
        }
        if (!context.paired) return PermissionDecision.deny("An encrypted pairing is required")
        if (capability !in grants) return PermissionDecision.deny("Capability has not been granted")

        return when (capability) {
            CompanionCapability.READ_STATUS,
            CompanionCapability.CONVERSATION ->
                PermissionDecision(true, AuthenticationRequirement.NONE, "Low-risk paired capability")

            CompanionCapability.READ_PRIVATE_MEMORY,
            CompanionCapability.REVOKE_PAIRING -> if (context.biometricVerified) {
                PermissionDecision(true, AuthenticationRequirement.BIOMETRIC_OR_DEVICE_CREDENTIAL, "Device authentication verified")
            } else {
                PermissionDecision.deny(
                    "Device authentication is required",
                    AuthenticationRequirement.BIOMETRIC_OR_DEVICE_CREDENTIAL,
                )
            }

            CompanionCapability.REMOTE_COMMAND,
            CompanionCapability.EXTERNAL_MESSAGE,
            CompanionCapability.PHONE_CALL,
            CompanionCapability.DEVICE_SETTINGS -> when {
                context.source == RequestSource.VOICE ->
                    PermissionDecision.deny("Voice-only authorization is not accepted", AuthenticationRequirement.LOCAL_CONFIRMATION)
                !context.userConfirmed ->
                    PermissionDecision.deny("Show and confirm the exact action locally", AuthenticationRequirement.LOCAL_CONFIRMATION)
                !context.biometricVerified ->
                    PermissionDecision.deny(
                        "Device authentication is required",
                        AuthenticationRequirement.BIOMETRIC_OR_DEVICE_CREDENTIAL,
                    )
                else -> PermissionDecision(true, AuthenticationRequirement.BIOMETRIC_OR_DEVICE_CREDENTIAL, "Protected action authorized")
            }

            CompanionCapability.PAIR -> error("handled above")
        }
    }

    /** Android runtime permissions needed by a capability, if any. */
    fun requiredAndroidPermissions(capability: CompanionCapability): Set<String> = when (capability) {
        CompanionCapability.PHONE_CALL -> setOf(Manifest.permission.CALL_PHONE)
        CompanionCapability.DEVICE_SETTINGS -> emptySet()
        else -> emptySet()
    }

    companion object {
        val DEFAULT_GRANTS: Set<CompanionCapability> = setOf(
            CompanionCapability.PAIR,
            CompanionCapability.READ_STATUS,
            CompanionCapability.CONVERSATION,
        )

        fun bluetoothPermissions(): Set<String> = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            setOf(Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT)
        } else {
            emptySet()
        }

        fun notificationPermissions(): Set<String> = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            setOf(Manifest.permission.POST_NOTIFICATIONS)
        } else {
            emptySet()
        }
    }
}
