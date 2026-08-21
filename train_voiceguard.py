"""Train and calibrate the student-trained VoiceGuard classifier head."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from asher.config import AsherConfig
from asher.voiceguard import EnrollmentManager, TrainingConfig, ml_dependencies_available


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train VoiceGuard from consented session manifests")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--model-path")
    parser.add_argument("--task", choices=("wake_word", "speaker_auth"), default="speaker_auth")
    args = parser.parse_args(argv)
    load_dotenv()
    available, reason = ml_dependencies_available()
    if not available:
        raise SystemExit(reason or "Optional VoiceGuard ML dependencies are unavailable.")
    config = AsherConfig.load(args.runtime_dir)
    manager = EnrollmentManager(config.runtime.root / "voiceguard")
    result = manager.retrain(
        config=TrainingConfig(task=args.task),
        model_path=args.model_path,
    )
    print(
        "VoiceGuard model trained from private sessions: "
        f"measured_test={result.measured_test}, test_samples={len(result.split.test)}"
    )
    print("Replay/noisy metrics remain unavailable unless real recordings for those conditions were supplied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
