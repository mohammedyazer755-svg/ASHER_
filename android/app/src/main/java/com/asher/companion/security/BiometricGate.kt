package com.asher.companion.security

import android.content.Context
import android.os.Build
import androidx.biometric.BiometricManager
import androidx.biometric.BiometricPrompt
import androidx.core.content.ContextCompat
import androidx.fragment.app.FragmentActivity
import java.util.UUID
import java.util.concurrent.Executor

enum class BiometricAvailability {
    AVAILABLE,
    NONE_ENROLLED,
    HARDWARE_UNAVAILABLE,
    TEMPORARILY_UNAVAILABLE,
    UNKNOWN,
}

data class BiometricGrant(
    val token: String,
    val expiresAtEpochMillis: Long,
)

data class AuthenticationResult(
    val verified: Boolean,
    val reason: String,
    val grant: BiometricGrant? = null,
)

/**
 * One-shot local authentication gate. Callers must pass the result into
 * PermissionPolicy; merely showing a prompt never authorizes a command.
 */
class BiometricGate(context: Context) {
    private val appContext = context.applicationContext
    private val executor: Executor = ContextCompat.getMainExecutor(appContext)
    private val grants = mutableMapOf<String, Long>()
    private val grantTtlMillis = 60_000L

    fun availability(): BiometricAvailability {
        val manager = BiometricManager.from(appContext)
        val result = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            manager.canAuthenticate(
                BiometricManager.Authenticators.BIOMETRIC_STRONG or
                    BiometricManager.Authenticators.DEVICE_CREDENTIAL,
            )
        } else {
            @Suppress("DEPRECATION")
            manager.canAuthenticate(BiometricManager.Authenticators.BIOMETRIC_STRONG)
        }
        return when (result) {
            BiometricManager.BIOMETRIC_SUCCESS -> BiometricAvailability.AVAILABLE
            BiometricManager.BIOMETRIC_ERROR_NONE_ENROLLED -> BiometricAvailability.NONE_ENROLLED
            BiometricManager.BIOMETRIC_ERROR_HW_UNAVAILABLE -> BiometricAvailability.HARDWARE_UNAVAILABLE
            BiometricManager.BIOMETRIC_ERROR_UNSUPPORTED,
            BiometricManager.BIOMETRIC_ERROR_NO_HARDWARE -> BiometricAvailability.HARDWARE_UNAVAILABLE
            BiometricManager.BIOMETRIC_ERROR_SECURITY_UPDATE_REQUIRED,
            BiometricManager.BIOMETRIC_ERROR_UNABLE_TO_PROCESS,
            BiometricManager.BIOMETRIC_ERROR_TIMEOUT -> BiometricAvailability.TEMPORARILY_UNAVAILABLE
            else -> BiometricAvailability.UNKNOWN
        }
    }

    fun authenticate(
        activity: FragmentActivity,
        title: String,
        subtitle: String = "Confirm this protected ASHER action",
        onResult: (AuthenticationResult) -> Unit,
    ) {
        if (availability() != BiometricAvailability.AVAILABLE) {
            onResult(AuthenticationResult(false, "Device authentication is unavailable"))
            return
        }
        val promptInfo = BiometricPrompt.PromptInfo.Builder()
            .setTitle(title)
            .setSubtitle(subtitle)
            .setConfirmationRequired(true)
            .apply {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    setAllowedAuthenticators(
                        BiometricManager.Authenticators.BIOMETRIC_STRONG or
                            BiometricManager.Authenticators.DEVICE_CREDENTIAL,
                    )
                } else {
                    @Suppress("DEPRECATION")
                    setNegativeButtonText("Cancel")
                }
            }
            .build()

        val prompt = BiometricPrompt(
            activity,
            executor,
            object : BiometricPrompt.AuthenticationCallback() {
                override fun onAuthenticationSucceeded(result: BiometricPrompt.AuthenticationResult) {
                    val grant = issueGrant()
                    onResult(AuthenticationResult(true, "Device authentication verified", grant))
                }

                override fun onAuthenticationError(errorCode: Int, errString: CharSequence) {
                    onResult(AuthenticationResult(false, "Authentication failed: $errString"))
                }

                override fun onAuthenticationFailed() {
                    // Do not grant on a failed attempt. The system may keep the
                    // prompt open and eventually call succeeded or error.
                }
            },
        )
        prompt.authenticate(promptInfo)
    }

    /** Consume exactly one successful prompt result before a protected action. */
    @Synchronized
    fun consume(grant: BiometricGrant?): Boolean {
        if (grant == null) return false
        val expiry = grants.remove(grant.token) ?: return false
        return expiry == grant.expiresAtEpochMillis && expiry > System.currentTimeMillis()
    }

    @Synchronized
    private fun issueGrant(): BiometricGrant {
        val now = System.currentTimeMillis()
        grants.entries.removeIf { it.value <= now }
        val grant = BiometricGrant(UUID.randomUUID().toString(), now + grantTtlMillis)
        grants[grant.token] = grant.expiresAtEpochMillis
        return grant
    }
}
