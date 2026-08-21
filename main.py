"""ASHER entry point.

Importing this module is side-effect free. Choose ``--ui`` for the desktop
frontend, ``--voice`` for explicit microphone mode, or ``--text`` for a safe
terminal fallback.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from typing import Iterable

from dotenv import load_dotenv

from asher.agent.controller import CompanionController
from asher.config import AsherConfig
from asher.types import AuthMethod
from asher.voice.wakeword import DEFAULT_WAKE_PHRASES, match_wake_phrase


WAKE_PHRASES = DEFAULT_WAKE_PHRASES
SLEEP_COMMANDS = frozenset({"sleep", "go to sleep", "standby", "go to standby", "that’s all", "that's all", "nothing else"})


def detect_wake_phrase(text: str) -> tuple[bool, str]:
    match = match_wake_phrase(text, WAKE_PHRASES)
    return match.detected, match.command


def process_command(command: str, *, controller: CompanionController | None = None, session=None) -> str:
    """Process one text command through the typed controller."""

    controller = controller or CompanionController()
    session = session or controller.create_owner_session(AuthMethod.LOCAL_UI)
    reply = controller.handle_text(command, session)
    print(f"Asher: {reply.text}")
    for update in reply.updates:
        if update.confirmation_id:
            print(f"Confirmation required: {update.confirmation_id}")
    return reply.text


def _text_mode(controller: CompanionController) -> int:
    session = controller.create_owner_session(AuthMethod.LOCAL_UI)
    print("ASHER text mode (dry-run=%s). Type 'exit' to close." % controller.config.dry_run)
    print("Use 'emergency stop' to cancel every active plan.")
    while True:
        try:
            command = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if command.casefold().rstrip(".,!?;:") in {"exit", "quit", "goodbye", "bye"}:
            break
        if command.casefold().startswith("approve "):
            confirmation_id = command.split(None, 1)[1].strip()
            reply = controller.approve(confirmation_id, session, device_authenticated=False)
        elif command.casefold().startswith("approve-device "):
            confirmation_id = command.split(None, 1)[1].strip()
            reply = controller.approve(confirmation_id, session, device_authenticated=True)
        elif command.casefold().startswith("reject "):
            reply = controller.reject(command.split(None, 1)[1].strip(), session)
        else:
            reply = controller.handle_text(command, session)
        print(f"Asher: {reply.text}")
        for update in reply.updates:
            if update.confirmation_id:
                print(f"Confirmation ID: {update.confirmation_id} (use approve or approve-device)")
    return 0


def _voice_mode(controller: CompanionController) -> int:
    from asher.voice.runtime import VoiceRuntime, load_active_voiceguard_verifier

    runtime = VoiceRuntime(
        controller,
        voiceguard=load_active_voiceguard_verifier(controller),
        on_event=lambda event: print(f"[{event.kind}] {event.message}"),
    )
    try:
        runtime.run_forever()
    finally:
        runtime.stop()
    return 0


def _ui_mode(controller: CompanionController) -> int:
    from asher.ui.app import run
    from asher.ui.companion_adapter import CompanionDesktopController

    # Widgets consume the narrow UI protocol; the adapter keeps all policy,
    # sessions, memory, and tool execution in the real companion controller.
    return run(sys.argv, controller=CompanionDesktopController(controller))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASHER authenticated personal companion")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--ui", action="store_true", help="Launch the responsive PySide6 desktop UI")
    modes.add_argument("--voice", action="store_true", help="Start explicit microphone/wake-word mode")
    modes.add_argument("--text", action="store_true", help="Run the safe terminal text fallback")
    parser.add_argument("--live", action="store_true", help="Disable dry-run mode; policy and confirmations still apply")
    parser.add_argument("--runtime-dir", help="Override the private ASHER runtime directory")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    config = AsherConfig.load(args.runtime_dir)
    if args.live:
        config = replace(config, dry_run=False)
    controller = CompanionController(config)
    if args.voice:
        return _voice_mode(controller)
    if args.text:
        return _text_mode(controller)
    try:
        return _ui_mode(controller)
    except RuntimeError as error:
        print(f"Desktop UI unavailable: {error}", file=sys.stderr)
        return _text_mode(controller)


if __name__ == "__main__":
    raise SystemExit(main())
