package com.asher.companion.protocol

import com.asher.companion.security.AuthorizationContext
import com.asher.companion.security.CompanionCapability
import com.asher.companion.security.PermissionPolicy
import com.asher.companion.security.RequestSource
import org.json.JSONObject

/**
 * Boundary used by a future socket/Bluetooth adapter. It keeps policy checks
 * next to the encrypted channel so a transport cannot accidentally forward a
 * raw command around the authorization layer.
 */
class CompanionSession(
    private val channel: SecureChannel,
    private val permissionPolicy: PermissionPolicy,
    private val paired: () -> Boolean,
) {
    fun sendCommand(
        capability: CompanionCapability,
        command: String,
        arguments: JSONObject = JSONObject(),
        userConfirmed: Boolean,
        biometricVerified: Boolean,
        source: RequestSource = RequestSource.LOCAL_UI,
    ): ByteArray {
        val decision = permissionPolicy.evaluate(
            capability,
            AuthorizationContext(
                paired = paired(),
                userConfirmed = userConfirmed,
                biometricVerified = biometricVerified,
                source = source,
            ),
        )
        if (!decision.allowed) throw SecurityException(decision.reason)
        val payload = JSONObject()
            .put("command", command)
            .put("arguments", arguments)
            .put("capability", capability.name)
        return channel.encrypt(MessageType.COMMAND, payload.toString())
    }

    fun receive(): DecryptedMessage = channel.decrypt(nextFrame())

    /** Transport adapters provide one complete length-prefixed frame here. */
    var nextFrame: () -> ByteArray = {
        throw IllegalStateException("No transport frame is attached")
    }
}
