package com.asher.companion.protocol

import com.asher.companion.security.CryptoPrimitives
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets

data class DecryptedMessage(
    val type: MessageType,
    val sessionId: String,
    val sequence: Long,
    val payload: String,
)

/**
 * Stateful AEAD channel. Every direction has its own instance and sequence
 * counter. Frames must arrive in order; rejecting gaps is intentional because
 * the companion does not implement an acknowledgement/replay window yet.
 */
class SecureChannel(
    channelKey: ByteArray,
    val sessionId: String,
    private val randomBytes: (Int) -> ByteArray = CryptoPrimitives::randomBytes,
) : AutoCloseable {
    private val lock = Any()
    private val key = channelKey.copyOf().also {
        require(it.size == 32) { "Channel key must be 256 bits" }
    }
    private var sendSequence = 0L
    private var receiveSequence = 0L
    private var closed = false

    init {
        if (sessionId.isBlank() || sessionId.length > 128 ||
            !sessionId.matches(Regex("[A-Za-z0-9._~:-]+"))
        ) {
            throw ProtocolException("Invalid channel session id")
        }
    }

    fun encrypt(type: MessageType, payload: String): ByteArray = synchronized(lock) {
        checkOpen()
        val plaintext = Protocol.boundedUtf8(payload, "payload")
        val sequence = sendSequence
        if (sequence == Long.MAX_VALUE) throw ProtocolException("Channel sequence exhausted")
        val header = FrameHeader(type, sessionId, sequence)
        val iv = randomBytes(Protocol.GCM_IV_BYTES)
        if (iv.size != Protocol.GCM_IV_BYTES) throw ProtocolException("Random source returned an invalid IV")
        val ciphertext = CryptoPrimitives.encrypt(key, iv, Protocol.utf8(header.canonical()), plaintext)
        val frame = EncryptedFrame(
            header = header,
            iv = Protocol.encodeBase64(iv),
            ciphertext = Protocol.encodeBase64(ciphertext),
        )
        val encoded = FrameCodec.encode(frame)
        if (encoded.size > Protocol.MAX_FRAME_BYTES) throw ProtocolException("Encrypted frame is too large")
        sendSequence += 1
        encoded
    }

    fun decrypt(encodedFrame: ByteArray): DecryptedMessage = synchronized(lock) {
        checkOpen()
        if (encodedFrame.size > Protocol.MAX_FRAME_BYTES) throw ProtocolException("Encrypted frame is too large")
        val frame = FrameCodec.decode(encodedFrame)
        if (frame.header.sessionId != sessionId) throw SecurityException("Frame belongs to another session")
        if (frame.header.sequence != receiveSequence) {
            throw SecurityException("Unexpected or replayed frame sequence")
        }
        val iv = Protocol.decodeBase64(frame.iv, "iv")
        val ciphertext = Protocol.decodeBase64(frame.ciphertext, "ciphertext")
        val plaintext = CryptoPrimitives.decrypt(
            key,
            iv,
            Protocol.utf8(frame.header.canonical()),
            ciphertext,
        )
        if (plaintext.size > Protocol.MAX_PAYLOAD_BYTES) throw ProtocolException("Payload is too large")
        val payload = plaintext.toString(StandardCharsets.UTF_8)
        receiveSequence += 1
        DecryptedMessage(frame.header.type, sessionId, frame.header.sequence, payload)
    }

    fun sentSequence(): Long = synchronized(lock) { sendSequence }

    fun receivedSequence(): Long = synchronized(lock) { receiveSequence }

    fun isClosed(): Boolean = synchronized(lock) { closed }

    override fun close() = synchronized(lock) {
        if (!closed) {
            closed = true
            CryptoPrimitives.wipe(key)
        }
    }

    private fun checkOpen() {
        if (closed) throw IllegalStateException("Secure channel is closed")
    }
}

/** Length-prefix framing keeps TCP/WebSocket adapters from relying on JSON delimiters. */
object FrameCodec {
    private const val LENGTH_PREFIX_BYTES = 4

    fun encode(frame: EncryptedFrame): ByteArray {
        val json = frame.toJson().toByteArray(StandardCharsets.UTF_8)
        if (json.isEmpty() || json.size > Protocol.MAX_FRAME_BYTES - LENGTH_PREFIX_BYTES) {
            throw ProtocolException("Frame JSON exceeds protocol limits")
        }
        return ByteBuffer.allocate(LENGTH_PREFIX_BYTES + json.size)
            .order(ByteOrder.BIG_ENDIAN)
            .putInt(json.size)
            .put(json)
            .array()
    }

    fun decode(encoded: ByteArray): EncryptedFrame {
        if (encoded.size < LENGTH_PREFIX_BYTES) throw ProtocolException("Truncated frame")
        val length = ByteBuffer.wrap(encoded, 0, LENGTH_PREFIX_BYTES).order(ByteOrder.BIG_ENDIAN).int
        if (length <= 0 || length > Protocol.MAX_FRAME_BYTES - LENGTH_PREFIX_BYTES || encoded.size != length + LENGTH_PREFIX_BYTES) {
            throw ProtocolException("Invalid frame length")
        }
        val json = encoded.copyOfRange(LENGTH_PREFIX_BYTES, encoded.size)
        return EncryptedFrame.fromJson(json.toString(StandardCharsets.UTF_8))
    }
}
