"""Offline text-normalization tests for the Asher v0.8.1 voice upgrade."""

from voice.text_normalizer import normalise_contact_name, normalise_voice_command


TESTS = {
    "t-h-a-r-i-k-a": "Tharika",
    "t h a r i k a": "Tharika",
    "tee aitch ay ar eye kay ay": "Tharika",
    "tarika": "Tharika",
    "my lesson": "Malleshwaran",
    "money kandan": "Manikandan",
}

COMMAND_TESTS = {
    "touch t-h-a-r-i-k-a": "search Tharika",
    "such tarika": "search Tharika",
    "search tee aitch ay ar eye kay ay": "search Tharika",
    "send hi to t h a r i k a": "send hi to Tharika",
    "hey usher": "hey asher",
    "send it.": "send it",
    "don't send it!": "don't send it",
    "lock it?": "lock it",
    "go to sleep.": "go to sleep",
    "goodbye.": "goodbye",
}


def main() -> None:
    failures = []

    for source, expected in TESTS.items():
        actual = normalise_contact_name(source)
        print(f"CONTACT: {source!r} -> {actual!r}")
        if actual != expected:
            failures.append((source, expected, actual))

    for source, expected in COMMAND_TESTS.items():
        actual = normalise_voice_command(source)
        print(f"COMMAND: {source!r} -> {actual!r}")
        if actual != expected:
            failures.append((source, expected, actual))

    if failures:
        print("\n[FAILED] Voice normalization tests:")
        for source, expected, actual in failures:
            print(f"- {source!r}: expected {expected!r}, got {actual!r}")
        raise SystemExit(1)

    print("\n[PASSED] All offline voice-accuracy tests passed.")


if __name__ == "__main__":
    main()
