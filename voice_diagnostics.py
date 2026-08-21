"""Optional microphone diagnostics with no hardware access on import."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv


def print_microphones() -> None:
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
        devices = sd.query_devices()
    except Exception as error:
        print(f"Microphone diagnostics unavailable: {type(error).__name__}")
        return
    inputs = [
        (index, item.get("name", "Input device"))
        for index, item in enumerate(devices)
        if isinstance(item, dict) and int(item.get("max_input_channels", 0)) > 0
    ]
    if not inputs:
        print("No input microphones were reported.")
        return
    for index, name in inputs:
        print(f"[{index}] {name}")
    print("Set ASHER_MIC_INDEX locally in .env if a specific device is required.")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Use the authenticated main voice runtime (opens hardware)")
    args = parser.parse_args()
    print_microphones()
    if args.test:
        from asher.agent.controller import CompanionController
        from asher.voice.runtime import VoiceRuntime

        controller = CompanionController()
        runtime = VoiceRuntime(controller, on_event=lambda event: print(f"[{event.kind}] {event.message}"))
        try:
            runtime.run_forever()
        finally:
            runtime.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

