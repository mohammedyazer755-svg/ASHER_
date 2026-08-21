package com.asher.companion.protocol

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class SecureChannelTest {
    @Test
    fun roundTripAndReplayAreRejected() {
        val key = ByteArray(32) { it.toByte() }
        val iv = ByteArray(Protocol.GCM_IV_BYTES) { (it + 1).toByte() }
        val sender = SecureChannel(key, "test-session") { iv.copyOf() }
        val receiver = SecureChannel(key, "test-session") { iv.copyOf() }

        val encoded = sender.encrypt(MessageType.COMMAND, "{\"name\":\"status\"}")
        val message = receiver.decrypt(encoded)
        assertEquals(MessageType.COMMAND, message.type)
        assertEquals("{\"name\":\"status\"}", message.payload)
        assertThrows(SecurityException::class.java) { receiver.decrypt(encoded) }
    }

    @Test
    fun tamperingDoesNotAdvanceSequence() {
        val key = ByteArray(32) { 0x44 }
        val sender = SecureChannel(key, "session") { ByteArray(12) { 7 } }
        val receiver = SecureChannel(key, "session") { ByteArray(12) { 7 } }
        val encoded = sender.encrypt(MessageType.PING, "{}")
        val tampered = encoded.copyOf().also { it[it.lastIndex] = (it.last() + 1).toByte() }
        assertThrows(Exception::class.java) { receiver.decrypt(tampered) }
        assertEquals(0L, receiver.receivedSequence())
        assertEquals(MessageType.PING, receiver.decrypt(encoded).type)
    }
}
