package com.asher.companion.protocol

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class ProtocolTest {
    @Test
    fun frameJsonRoundTrips() {
        val header = FrameHeader(MessageType.EVENT, "session", 4)
        val frame = EncryptedFrame(
            header,
            Protocol.encodeBase64(ByteArray(12) { 1 }),
            Protocol.encodeBase64(ByteArray(32) { 2 }),
        )
        assertEquals(frame.toJson(), EncryptedFrame.fromJson(frame.toJson()).toJson())
    }

    @Test
    fun malformedVersionAndCodeAreRejected() {
        assertThrows(ProtocolException::class.java) {
            FrameHeader(MessageType.PING, "session", 0, version = 99)
        }
        assertThrows(ProtocolException::class.java) { Protocol.normalizePairingCode("short") }
    }
}
