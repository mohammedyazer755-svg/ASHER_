"""Collect and evaluate ASHER's real 100-command voice corpus.

The manifest and WAV files default to ASHER's private runtime directory.  The
CLI emits aggregate metrics only; it never prints audio paths or recognized
utterances.  Microphone and Faster-Whisper dependencies stay lazy until the
explicit ``record`` or ``evaluate`` command is selected.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from asher.brain.deterministic import ContactResolver
from asher.config import AsherConfig
from asher.types import AuthMethod
from asher.voice.evaluation import (
    VoiceEvaluationReport,
    VoiceFixture,
    VoicePrediction,
    evaluate_predictions,
    generate_non_private_fixtures,
    read_fixture_manifest,
    write_fixture_manifest,
)


REQUIRED_CORPUS_SIZE = 100


def default_manifest_path(config: AsherConfig | None = None) -> Path:
    selected = config or AsherConfig.load()
    selected.runtime.ensure()
    return selected.runtime.evaluations / "voice_accuracy" / "fixtures.jsonl"


def _audio_path(manifest_path: Path, fixture: VoiceFixture) -> Path:
    raw = str(fixture.audio_path or "").strip()
    if not raw:
        raise ValueError("fixture has no audio file assigned")
    relative = Path(raw)
    if relative.is_absolute() or relative.suffix.casefold() != ".wav":
        raise ValueError("fixture audio must be a relative WAV path")
    root = manifest_path.resolve().parent
    resolved = (root / relative).resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError("fixture audio path leaves the private evaluation directory")
    return resolved


def initialize_manifest(
    path: str | Path,
    *,
    count: int = REQUIRED_CORPUS_SIZE,
    overwrite: bool = False,
) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError("voice evaluation manifest already exists")
    fixtures = tuple(
        replace(item, audio_path=f"audio/{item.fixture_id}.wav")
        for item in generate_non_private_fixtures(count)
    )
    return write_fixture_manifest(fixtures, destination)


def corpus_inventory(path: str | Path) -> dict[str, Any]:
    manifest = Path(path).expanduser().resolve()
    fixtures = read_fixture_manifest(manifest)
    condition_totals = Counter(item.condition for item in fixtures)
    condition_recorded: Counter[str] = Counter()
    recorded = 0
    invalid = 0
    for fixture in fixtures:
        try:
            exists = _audio_path(manifest, fixture).is_file()
        except ValueError:
            invalid += 1
            exists = False
        if exists:
            recorded += 1
            condition_recorded[fixture.condition] += 1
    return {
        "fixture_count": len(fixtures),
        "recorded_count": recorded,
        "missing_count": len(fixtures) - recorded,
        "invalid_path_count": invalid,
        "required_count": REQUIRED_CORPUS_SIZE,
        "ready": (
            len(fixtures) >= REQUIRED_CORPUS_SIZE
            and recorded == len(fixtures)
            and invalid == 0
        ),
        "conditions": {
            name: {
                "fixture_count": condition_totals[name],
                "recorded_count": condition_recorded[name],
            }
            for name in sorted(condition_totals)
        },
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _record_wav(
    destination: Path,
    *,
    duration_seconds: float,
    sample_rate: int,
    device: int | str | None,
) -> None:
    try:
        import sounddevice
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "real corpus recording requires the optional sounddevice package and a working microphone"
        ) from error
    from asher.voiceguard.audio import PcmAudio, write_wav

    frames = max(1, round(duration_seconds * sample_rate))
    capture = sounddevice.rec(
        frames,
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        device=device,
        blocking=False,
    )
    sounddevice.wait()
    samples = tuple(
        int(value)
        for row in capture.tolist()
        for value in (row if isinstance(row, list) else [row])
    )
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.wav")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_wav(temporary, PcmAudio(samples, sample_rate, 1))
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def record_fixtures(
    path: str | Path,
    *,
    start: int,
    count: int,
    consent: bool,
    duration_seconds: float,
    sample_rate: int = 16_000,
    device: int | str | None = None,
    replace_existing: bool = False,
    recorder: Callable[..., None] = _record_wav,
    wait_for_user: Callable[[str], str] = input,
) -> int:
    if not consent:
        raise PermissionError("explicit consent is required before retaining corpus recordings")
    if duration_seconds <= 0 or duration_seconds > 30:
        raise ValueError("recording duration must be between 0 and 30 seconds")
    manifest = Path(path).expanduser().resolve()
    fixtures = read_fixture_manifest(manifest)
    if start < 1 or count < 1 or start > len(fixtures):
        raise ValueError("recording range is outside the fixture manifest")
    selected = fixtures[start - 1 : start - 1 + count]
    recorded = 0
    for offset, fixture in enumerate(selected, start=start):
        destination = _audio_path(manifest, fixture)
        if destination.exists() and not replace_existing:
            continue
        print(f"Fixture {offset}/{len(fixtures)} · condition: {fixture.condition}")
        print(f"Speak exactly: {fixture.expected_transcript}")
        wait_for_user("Press Enter when ready to record this real sample...")
        recorder(
            destination,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            device=device,
        )
        recorded += 1
    return recorded


_TOOL_INTENTS = {
    "whatsapp.prepare": "whatsapp_search",
    "whatsapp.send": "send_whatsapp",
    "app.open": "open_app",
    "app.close": "close_app",
    "system.volume_up": "volume_up",
    "system.volume_down": "volume_down",
    "system.toggle_mute": "toggle_mute",
    "system.screenshot": "take_screenshot",
}


class SafeTaskEvaluator:
    """Run prescribed transcripts through ASHER's real dry-run controller."""

    def __init__(self, fixtures: Iterable[VoiceFixture]) -> None:
        from asher.agent.controller import CompanionController

        contacts = tuple(
            dict.fromkeys(
                item.expected_contact
                for item in fixtures
                if item.expected_contact
            )
        )
        self._temporary = tempfile.TemporaryDirectory()
        config = replace(AsherConfig.load(self._temporary.name), dry_run=True)
        self.controller = CompanionController(
            config,
            contact_resolver=ContactResolver(contacts),
        )
        # Evaluation must never reach a cloud or local generative provider.
        self.controller.planner.openai = None
        self.controller.planner.ollama = None

    def close(self) -> None:
        self._temporary.cleanup()

    def __enter__(self) -> "SafeTaskEvaluator":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def interpret(self, text: str) -> tuple[str | None, str | None, bool]:
        normalized = str(text).strip()
        if normalized.casefold().rstrip(".,!?;:") == "go to sleep":
            return "sleep", None, True
        plan = self.controller.planner.deterministic.plan(normalized)
        if plan is None or not plan.steps:
            return None, None, False
        calls = tuple(step.call for step in plan.steps)
        selected_call = next(
            (call for call in calls if call.tool_name == "whatsapp.send"),
            calls[0],
        )
        intent = _TOOL_INTENTS.get(selected_call.tool_name)
        contact_value = selected_call.arguments.get("contact")
        contact = str(contact_value) if contact_value else None

        session = self.controller.create_owner_session(AuthMethod.LOCAL_UI)
        reply = self.controller.handle_text(normalized, session)
        if reply.confirmation_id:
            task_success = intent == "send_whatsapp"
            self.controller.reject(reply.confirmation_id, session)
        else:
            statuses = {item.status for item in reply.updates}
            task_success = "complete" in statuses and not statuses.intersection(
                {"failed", "denied", "cancelled"}
            )
        return intent, contact, task_success


def evaluate_manifest(
    path: str | Path,
    *,
    transcribe: Callable[[Path], str],
    interpret: Callable[[str], tuple[str | None, str | None, bool]],
    clock: Callable[[], float] = time.perf_counter,
) -> VoiceEvaluationReport:
    manifest = Path(path).expanduser().resolve()
    fixtures = read_fixture_manifest(manifest)
    predictions: list[VoicePrediction] = []
    for fixture in fixtures:
        try:
            audio = _audio_path(manifest, fixture)
        except ValueError:
            continue
        if not audio.is_file():
            continue
        started = clock()
        transcript = str(transcribe(audio)).strip()
        intent, contact, task_success = interpret(transcript)
        predictions.append(
            VoicePrediction(
                fixture_id=fixture.fixture_id,
                transcript=transcript,
                intent=intent,
                contact=contact,
                latency_ms=max(0.0, (clock() - started) * 1000.0),
                task_success=task_success,
            )
        )
    return evaluate_predictions(fixtures, predictions)


def _device(value: str | None) -> int | str | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError:
        return value.strip()
    if parsed < 0:
        raise argparse.ArgumentTypeError("device index must be non-negative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Private fixture manifest (defaults under ASHER_RUNTIME_DIR)")
    parser.add_argument("--json", action="store_true", help="Print aggregate JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("init", help="Create a prescribed non-private 100-command manifest")
    initialize.add_argument("--count", type=int, default=REQUIRED_CORPUS_SIZE)
    initialize.add_argument("--force", action="store_true")

    record = commands.add_parser("record", help="Record an explicitly selected range with consent")
    record.add_argument("--start", type=int, default=1)
    record.add_argument("--count", type=int, default=1)
    record.add_argument("--duration", type=float, default=4.0)
    record.add_argument("--device")
    record.add_argument("--consent", action="store_true")
    record.add_argument("--replace", action="store_true")

    commands.add_parser("check", help="Report aggregate corpus readiness without loading speech models")
    evaluate = commands.add_parser("evaluate", help="Run Faster-Whisper and the dry-run ASHER task path")
    evaluate.add_argument("--report", type=Path, help="Aggregate JSON report destination")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = (args.manifest or default_manifest_path()).expanduser().resolve()
    try:
        if args.command == "init":
            initialize_manifest(manifest, count=args.count, overwrite=args.force)
            result = corpus_inventory(manifest)
        elif args.command == "record":
            recorded = record_fixtures(
                manifest,
                start=args.start,
                count=args.count,
                consent=args.consent,
                duration_seconds=args.duration,
                device=_device(args.device),
                replace_existing=args.replace,
            )
            result = corpus_inventory(manifest)
            result["recorded_this_run"] = recorded
        elif args.command == "check":
            result = corpus_inventory(manifest)
        else:
            from asher.voice.transcription import FasterWhisperTranscriber, TranscriptionConfig

            config = AsherConfig.load()
            transcriber = FasterWhisperTranscriber(
                TranscriptionConfig(
                    model_size=config.whisper_model,
                    device=config.whisper_device,
                    cpu_compute_type=config.whisper_compute_type,
                )
            )
            fixtures = read_fixture_manifest(manifest)
            vocabulary = tuple(
                dict.fromkeys(
                    item.expected_contact
                    for item in fixtures
                    if item.expected_contact
                )
            )
            with SafeTaskEvaluator(fixtures) as task_evaluator:
                report = evaluate_manifest(
                    manifest,
                    transcribe=lambda audio: transcriber.transcribe(
                        audio,
                        vocabulary=vocabulary,
                    ).normalized_text,
                    interpret=task_evaluator.interpret,
                )
            result = report.to_dict()
            report_path = (
                args.report.expanduser().resolve()
                if args.report
                else manifest.parent / "report.json"
            )
            _atomic_json(report_path, result)
            result["ready"] = (
                report.sample_count >= REQUIRED_CORPUS_SIZE
                and report.missing_prediction_count == 0
            )
    except Exception as error:
        message = f"Voice accuracy workflow failed: {type(error).__name__}: {error}"
        if args.json:
            print(json.dumps({"ready": False, "error": message}, sort_keys=True))
        else:
            print(message)
        return 1

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            "Voice accuracy corpus: "
            f"{result.get('recorded_count', result.get('sample_count', 0))}/"
            f"{result.get('fixture_count', result.get('sample_count', 0))} real recordings; "
            f"ready={'yes' if result.get('ready') else 'no'}"
        )
    return 0 if result.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
