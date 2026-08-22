"""Guided local VoiceGuard speaker-auth and wake-word data collection.

This helper records one *session* containing several short clips. Re-run it in
separate environments/times so VoiceGuard can keep train/validation/test data
session-separated. Raw audio remains inside ASHER's private runtime directory
and is never printed. Wake labels describe the literal guided prompt; this
collector does not claim synthetic, noisy, or replay conditions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from asher.config import AsherConfig
from asher.security.users import UserStore
from asher.storage import Database
from asher.voiceguard import (
    EnrollmentManager,
    SpeakerRole,
    TrainingConfig,
    TrainingReadiness,
    TrainingTask,
    VoiceGuardError,
)


DEFAULT_PROMPTS = (
    "Open Chrome and search for today's notes.",
    "Turn the volume down a little.",
    "What did I work on yesterday?",
    "Open WhatsApp and find my recent chat.",
    "Read the next task on my list.",
    "Close the current window.",
)

WAKE_POSITIVE_PROMPTS = (
    "Hey Asher.",
    "Hey Asher, open Chrome.",
    "Hey Asher, what is next on my list?",
    "Hey Asher, turn the volume down.",
    "Hey Asher, read today's notes.",
    "Hey Asher, go to standby.",
)

WAKE_NEGATIVE_PROMPTS = DEFAULT_PROMPTS


@dataclass(frozen=True)
class CollectionIdentity:
    speaker_id: str
    role: SpeakerRole
    expected_authorized: bool
    display_name: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect one consented multi-clip VoiceGuard session"
    )
    parser.add_argument(
        "--task",
        choices=(TrainingTask.SPEAKER_AUTH.value, TrainingTask.WAKE_WORD.value),
        default=TrainingTask.SPEAKER_AUTH.value,
        help="Collect speaker identity clips or explicitly labelled wake-word clips",
    )
    parser.add_argument(
        "--wake-label",
        choices=("positive", "negative"),
        help="Required for --task wake_word; whether every guided prompt contains 'Hey Asher'",
    )
    parser.add_argument(
        "--speaker",
        choices=("owner", "unknown"),
        default="owner",
        help="Collect the real ASHER owner or a consented negative/unknown speaker",
    )
    parser.add_argument(
        "--speaker-id",
        default="unknown_pool",
        help="Dataset-only label for --speaker unknown; ignored for the owner",
    )
    parser.add_argument("--environment", default="quiet_room", help="Short environment tag")
    parser.add_argument("--samples", type=int, default=6, help="Clips in this recording session (3-20)")
    parser.add_argument("--duration", type=float, default=3.0, help="Seconds per clip (1.5-8.0)")
    parser.add_argument("--device", help="Optional sounddevice input-device index or name")
    parser.add_argument("--runtime-dir")
    parser.add_argument(
        "--consent",
        action="store_true",
        help="Confirm the recorded speaker agreed to private local retention for VoiceGuard",
    )
    return parser


def resolve_identity(config: AsherConfig, *, speaker: str, speaker_id: str) -> CollectionIdentity:
    if speaker == "owner":
        config.runtime.ensure()
        owner = UserStore(Database(config.runtime.database)).ensure_owner(config.owner_name)
        return CollectionIdentity(owner.user_id, SpeakerRole.OWNER, True, owner.display_name)

    label = speaker_id.strip()
    if not label:
        raise ValueError("--speaker-id is required for an unknown speaker")
    return CollectionIdentity(label, SpeakerRole.UNKNOWN, False, "consented unknown speaker")


def validate_collection_settings(samples: int, duration: float, environment: str) -> None:
    if not 3 <= samples <= 20:
        raise ValueError("--samples must be between 3 and 20")
    if not 1.5 <= duration <= 8.0:
        raise ValueError("--duration must be between 1.5 and 8.0 seconds")
    if not environment.strip():
        raise ValueError("--environment must not be empty")


def collection_prompt_plan(
    task: str,
    wake_label: str | None,
) -> tuple[tuple[str, ...], bool]:
    """Return honest prompts and the literal manifest wake-phrase label."""

    try:
        selected = TrainingTask(task)
    except ValueError as exc:
        raise ValueError("--task must be speaker_auth or wake_word") from exc
    if selected is TrainingTask.SPEAKER_AUTH:
        if wake_label is not None:
            raise ValueError("--wake-label is only valid with --task wake_word")
        return DEFAULT_PROMPTS, False
    if wake_label is None:
        raise ValueError("--wake-label is required with --task wake_word")
    if wake_label == "positive":
        return WAKE_POSITIVE_PROMPTS, True
    if wake_label == "negative":
        return WAKE_NEGATIVE_PROMPTS, False
    raise ValueError("--wake-label must be positive or negative")


def collect_session(
    manager: EnrollmentManager,
    identity: CollectionIdentity,
    *,
    environment: str,
    samples: int,
    duration: float,
    task: str = TrainingTask.SPEAKER_AUTH.value,
    wake_label: str | None = None,
    device: int | str | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> tuple[str, int]:
    """Record one leakage-safe session after consent was already established."""

    validate_collection_settings(samples, duration, environment)
    prompts, contains_wake_phrase = collection_prompt_plan(task, wake_label)
    session = manager.begin_enrollment(
        identity.speaker_id,
        role=identity.role,
        environment=environment.strip(),
        consent=True,
    )
    try:
        label_suffix = f" ({wake_label})" if wake_label is not None else ""
        output_fn(
            f"Collecting {samples} private {task}{label_suffix} clips for "
            f"{identity.display_name}."
        )
        output_fn("Each clip starts immediately after you press Enter. Raw audio is not printed.")
        for index in range(samples):
            prompt = prompts[index % len(prompts)]
            input_fn(f"[{index + 1}/{samples}] Say: {prompt!r}  Press Enter when ready...")
            session.record_microphone(
                duration,
                contains_wake_phrase=contains_wake_phrase,
                expected_authorized=identity.expected_authorized,
                device=device,
            )
            output_fn(f"Captured {index + 1}/{samples}.")
        record = manager.finalize_enrollment(session, minimum_samples=samples)
    except Exception:
        # Keep a partially recorded consented session on disk for inspection;
        # do not register it as training data until finalization succeeds.
        raise
    return session.manifest.session_id, len(record.session_ids)


def _parse_device(raw: str | None) -> int | str | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def readiness_summary(readiness: TrainingReadiness) -> str:
    state = "READY" if readiness.ready else "NOT READY"
    return (
        f"VoiceGuard dataset: {state}; sessions={readiness.session_count}, "
        f"samples={readiness.sample_count}, classes={readiness.class_count}."
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if not args.consent:
        raise SystemExit(
            "Refusing to record without --consent. For another person, obtain their agreement first."
        )
    try:
        validate_collection_settings(args.samples, args.duration, args.environment)
        collection_prompt_plan(args.task, args.wake_label)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    config = AsherConfig.load(args.runtime_dir)
    identity = resolve_identity(config, speaker=args.speaker, speaker_id=args.speaker_id)
    manager = EnrollmentManager(config.runtime.root / "voiceguard")
    session_id, total_sessions = collect_session(
        manager,
        identity,
        environment=args.environment,
        samples=args.samples,
        duration=args.duration,
        task=args.task,
        wake_label=args.wake_label,
        device=_parse_device(args.device),
    )
    label_suffix = f", wake_label={args.wake_label}" if args.wake_label else ""
    print(
        f"VoiceGuard session complete: task={args.task}{label_suffix}, "
        f"speaker={args.speaker}, clips={args.samples}, "
        f"registered_sessions={total_sessions}."
    )
    try:
        readiness = manager.assess_training_readiness(
            config=TrainingConfig(task=args.task)
        )
    except VoiceGuardError as exc:
        print(f"Dataset readiness could not be checked safely: {exc}")
    else:
        print(readiness_summary(readiness))
    print(
        f"Run `python train_voiceguard.py --task {args.task} --check` for aggregate requirements. "
        "Session-separated evaluation needs repeated sessions recorded at different times/environments."
    )
    print("Do not upload the private WAV files; keep them local to ASHER.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
