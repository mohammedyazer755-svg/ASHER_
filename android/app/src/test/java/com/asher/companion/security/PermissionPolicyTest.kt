package com.asher.companion.security

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PermissionPolicyTest {
    @Test
    fun unpairedAndVoiceOnlyRequestsFailClosed() {
        val policy = PermissionPolicy(setOf(CompanionCapability.REMOTE_COMMAND))
        assertFalse(
            policy.evaluate(
                CompanionCapability.REMOTE_COMMAND,
                AuthorizationContext(paired = false, userConfirmed = true, biometricVerified = true),
            ).allowed,
        )
        assertFalse(
            policy.evaluate(
                CompanionCapability.REMOTE_COMMAND,
                AuthorizationContext(
                    paired = true,
                    userConfirmed = true,
                    biometricVerified = true,
                    source = RequestSource.VOICE,
                ),
            ).allowed,
        )
    }

    @Test
    fun protectedActionNeedsBothLocalConfirmationAndBiometric() {
        val policy = PermissionPolicy(setOf(CompanionCapability.REMOTE_COMMAND))
        assertFalse(
            policy.evaluate(
                CompanionCapability.REMOTE_COMMAND,
                AuthorizationContext(paired = true, userConfirmed = true),
            ).allowed,
        )
        assertTrue(
            policy.evaluate(
                CompanionCapability.REMOTE_COMMAND,
                AuthorizationContext(paired = true, userConfirmed = true, biometricVerified = true),
            ).allowed,
        )
    }
}
