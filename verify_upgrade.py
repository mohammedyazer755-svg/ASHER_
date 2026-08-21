"""Safe local verifier for the current ASHER architecture.

This replaces the legacy verifier that embedded personal contact names and
required a particular Ollama model. It reports optional blockers without
claiming they are installed.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from asher.brain.deterministic import ContactResolver, DeterministicPlanner
from asher.voice.wakeword import match_wake_phrase


ROOT = Path(__file__).resolve().parent


def main() -> int:
    failures: list[str] = []
    for path in ROOT.rglob("*.py"):
        if any(part in {".venv", "__pycache__"} for part in path.parts):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            failures.append(f"{path.relative_to(ROOT)}: {type(error).__name__}")
    planner = DeterministicPlanner(ContactResolver(("Demo Contact",), {"demo": "Demo Contact"}))
    if planner.plan("search D E M O").steps[0].call.arguments.get("contact") != "Demo Contact":
        failures.append("spelled contact resolution")
    if match_wake_phrase("washer").detected:
        failures.append("wake boundary safety")
    if not match_wake_phrase("Hey Asher, open chrome").detected:
        failures.append("wake phrase detection")
    if failures:
        for item in failures:
            print(f"[FAILED] {item}")
        return 1

    print("[PASSED] ASHER source tree parses.")
    print("[PASSED] Deterministic planner and wake boundary smoke checks passed.")
    for package in ("PySide6", "faster_whisper", "sounddevice", "openai", "sklearn"):
        state = "available" if importlib.util.find_spec(package) else "optional package missing"
        print(f"[INFO] {package}: {state}")
    print("[INFO] Ollama/model and physical microphone checks are runtime-only and were not fabricated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
