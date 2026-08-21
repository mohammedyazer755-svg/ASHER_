"""Compatibility smoke entry point.

The previous file printed the complete private conversation history when
imported. Use the documented test runner instead; this command performs only
safe import and wake-boundary checks.
"""

from main import detect_wake_phrase


def main() -> int:
    assert detect_wake_phrase("Hey Asher")[0]
    assert not detect_wake_phrase("washer")[0]
    print("ASHER safe smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

