"""Legacy speech helper backed by ASHER's provider-independent TTS layer."""

from __future__ import annotations

from asher.core.redaction import redact_text


_tts_manager = None


def speak(message, *, voice_profile: str | None = None, block: bool = False):
    """Print safe speech text and lazily request TTS.

    This compatibility function intentionally does not append plaintext turns
    to the old ``history.json`` file. The modern controller owns local timeline
    and audit persistence with redaction and retention controls.
    """
    global _tts_manager
    safe_message = redact_text(str(message))
    print(f"Asher: {safe_message}")
    try:
        if _tts_manager is None:
            from asher.voice.tts import build_default_tts

            _tts_manager = build_default_tts()
        handle = _tts_manager.speak_async(
            safe_message,
            profile_name=voice_profile,
            interrupt=False,
        )
        if block:
            handle.result()
    except Exception as error:
        # Provider messages can contain request metadata; log only a type.
        print(f"TTS unavailable: {type(error).__name__}")

