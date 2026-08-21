"""Consent-first VoiceGuard enrollment helper.

This command imports explicitly selected WAV files by default.  Microphone
capture is opt-in through ``--record`` and still asks for ``--consent``; raw
audio stays under the private runtime directory and is never printed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from asher.config import AsherConfig
from asher.voiceguard import EnrollmentManager, SpeakerRole


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a consented ASHER VoiceGuard recording session")
    parser.add_argument("user_id", help="Stable local user identifier (not a password)")
    parser.add_argument("--role", choices=("owner", "trusted", "unknown"), default="trusted")
    parser.add_argument("--wav", action="append", type=Path, default=[], help="WAV file to import; repeat for multiple samples")
    parser.add_argument("--record", action="store_true", help="Capture one microphone sample after explicit consent")
    parser.add_argument(
        "--wake-phrase",
        action="store_true",
        help="Mark imported/recorded audio as containing the Hey Asher wake phrase (default: negative sample)",
    )
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--consent", action="store_true", help="Confirm that selected voice recordings may be retained privately")
    parser.add_argument("--runtime-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if not args.consent:
        raise SystemExit("Refusing to record/import audio without --consent.")
    if not args.wav and not args.record:
        raise SystemExit("Provide at least one --wav or use --record.")
    config = AsherConfig.load(args.runtime_dir)
    manager = EnrollmentManager(config.runtime.root / "voiceguard")
    session = manager.begin_enrollment(
        args.user_id,
        role=SpeakerRole(args.role),
        environment="cli",
        consent=True,
    )
    for wav in args.wav:
        session.import_wav(wav, contains_wake_phrase=args.wake_phrase, expected_authorized=args.role in {"owner", "trusted"})
    if args.record:
        session.record_microphone(
            args.duration,
            contains_wake_phrase=args.wake_phrase,
            expected_authorized=args.role in {"owner", "trusted"},
        )
    enrollment = manager.finalize_enrollment(session, minimum_samples=1)
    print(f"VoiceGuard enrollment recorded: {len(enrollment.session_ids)} private session(s).")
    print("No authentication accuracy is claimed until a session-separated model is trained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
