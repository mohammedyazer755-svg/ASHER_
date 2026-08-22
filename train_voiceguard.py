"""Check, train, calibrate, and report the private VoiceGuard baseline."""

from __future__ import annotations

import argparse
import json
from typing import Any

from dotenv import load_dotenv

from asher.config import AsherConfig
from asher.voiceguard import (
    EnrollmentManager,
    TrainingConfig,
    TrainingReadiness,
    VoiceGuardError,
    ml_dependencies_available,
)


NOT_READY_EXIT = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check or train VoiceGuard from finalized consented recording sessions"
    )
    parser.add_argument("--runtime-dir")
    parser.add_argument("--task", choices=("wake_word", "speaker_auth"), default="speaker_auth")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic session-split seed")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Inspect structural readiness without extracting features or importing ML packages",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write one aggregate-only machine-readable result to stdout",
    )
    return parser


def _readiness_payload(readiness: TrainingReadiness) -> dict[str, Any]:
    return {"trained": False, "readiness": readiness.to_dict()}


def _format_metric(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def _print_readiness(readiness: TrainingReadiness) -> None:
    state = "READY" if readiness.ready else "NOT READY"
    print(
        f"VoiceGuard {readiness.task} readiness: {state}; "
        f"sessions={readiness.session_count}, samples={readiness.sample_count}, "
        f"classes={readiness.class_count}."
    )
    for name, coverage in readiness.split_coverage.items():
        print(
            f"  {name}: sessions={coverage.session_count}, samples={coverage.sample_count}, "
            f"authorized={coverage.authorized_sample_count}, "
            f"unauthorized={coverage.unauthorized_sample_count}"
        )
    if readiness.issues:
        print("Blocking requirements:")
        for issue in readiness.issues:
            details = ", ".join(f"{key}={value}" for key, value in sorted(issue.details.items()))
            suffix = f" ({details})" if details else ""
            print(f"  - {issue.message}{suffix}")
    if readiness.unavailable_conditions:
        print(
            "Unrecorded evaluation conditions: "
            + ", ".join(readiness.unavailable_conditions)
            + ". Their metrics will remain unavailable."
        )


def _training_payload(result: Any, readiness: TrainingReadiness) -> dict[str, Any]:
    artifacts = result.artifacts
    return {
        "trained": True,
        "runtime_activation": (
            "speaker_auth_active"
            if result.model.task == "speaker_auth"
            else "wake_word_active"
        ),
        "readiness": readiness.to_dict(),
        "model_version": result.model.model_version,
        "threshold": result.model.threshold,
        "split": {
            "train_sessions": len(result.split.train_sessions),
            "validation_sessions": len(result.split.validation_sessions),
            "test_sessions": len(result.split.test_sessions),
            "train_samples": len(result.split.train),
            "validation_samples": len(result.split.validation),
            "test_samples": len(result.split.test),
        },
        "test_metrics": {
            "measured": result.test_report.measured,
            "accuracy": result.test_report.accuracy,
            "f1": result.test_report.f1,
            "false_accept_rate": result.test_report.false_accept_rate,
            "false_reject_rate": result.test_report.false_reject_rate,
            "authorized_identity_accuracy": result.test_report.authorized_identity_accuracy,
            "authorized_identity_error_count": result.test_report.authorized_identity_error_count,
            "authorized_identity_sample_count": result.test_report.authorized_identity_sample_count,
            "replay_acceptance_rate": result.test_report.replay_acceptance_rate,
            "unavailable_conditions": list(result.test_report.unavailable_conditions),
        },
        "artifacts": (
            None
            if artifacts is None
            else {
                "model": artifacts.model_path.name,
                "validation_report": artifacts.validation_report_path.name,
                "test_report": artifacts.test_report_path.name,
            }
        ),
    }


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    config = AsherConfig.load(args.runtime_dir)
    manager = EnrollmentManager(config.runtime.root / "voiceguard")
    training_config = TrainingConfig(task=args.task, seed=args.seed)

    try:
        readiness = manager.assess_training_readiness(config=training_config)
    except (VoiceGuardError, ValueError) as exc:
        if args.json:
            print(json.dumps({"trained": False, "error": str(exc)}, sort_keys=True))
        else:
            print(f"VoiceGuard readiness check failed safely: {exc}")
        return 1

    if args.check or not readiness.ready:
        if args.json:
            print(json.dumps(_readiness_payload(readiness), sort_keys=True))
        else:
            _print_readiness(readiness)
        return 0 if readiness.ready else NOT_READY_EXIT

    available, reason = ml_dependencies_available()
    if not available:
        if args.json:
            print(json.dumps({**_readiness_payload(readiness), "error": reason}, sort_keys=True))
        else:
            print(reason or "Optional VoiceGuard ML dependencies are unavailable.")
        return 1

    try:
        result = manager.retrain(
            config=training_config,
        )
    except (VoiceGuardError, ValueError) as exc:
        if args.json:
            print(json.dumps({**_readiness_payload(readiness), "error": str(exc)}, sort_keys=True))
        else:
            print(f"VoiceGuard training stopped safely: {exc}")
        return 1

    payload = _training_payload(result, readiness)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
        return 0

    report = result.test_report
    print(
        "VoiceGuard private baseline trained: "
        f"model_version={result.model.model_version}, threshold={result.model.threshold:.4f}."
    )
    print(
        "Held-out test metrics: "
        f"samples={report.sample_count}, accuracy={_format_metric(report.accuracy)}, "
        f"F1={_format_metric(report.f1)}, FAR={_format_metric(report.false_accept_rate)}, "
        f"FRR={_format_metric(report.false_reject_rate)}."
    )
    if result.model.task == "speaker_auth":
        print(
            "Authorized identity accuracy="
            f"{_format_metric(report.authorized_identity_accuracy)} "
            f"({report.authorized_identity_error_count} identity errors across "
            f"{report.authorized_identity_sample_count} authorized trials)."
        )
    if report.replay_acceptance_rate is None:
        print("Replay acceptance remains unavailable because no real replay trials were tested.")
    else:
        print(f"Replay acceptance rate={_format_metric(report.replay_acceptance_rate)}.")
    if result.artifacts is not None:
        if result.model.task == "speaker_auth":
            print(
                "The speaker model was activated after its model and validation/test reports "
                "were saved in ASHER's private VoiceGuard directories."
            )
        else:
            print(
                "The wake-word model was activated as the separate standby wake gate after "
                "its model and reports were saved privately; it did not replace the active "
                "speaker-auth model."
            )
    print("VoiceGuard remains a convenience signal; high-risk actions still require device authentication.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
