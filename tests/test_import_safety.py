from __future__ import annotations

import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO


class ImportSafetyTests(unittest.TestCase):
    def test_main_import_has_no_listener_side_effect(self) -> None:
        result = subprocess.run(
            [sys.executable, "-B", "-c", "import main; print(main.detect_wake_phrase('washer'))"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "(False, '')")
        self.assertEqual(result.stderr.strip(), "")

    def test_legacy_helpers_do_not_speak_or_dump_history_on_import(self) -> None:
        result = subprocess.run(
            [sys.executable, "-B", "-c", "import conversations, utils, test; print('safe')"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "safe")

    def test_upgrade_verifier_accepts_safe_wake_boundary(self) -> None:
        import verify_upgrade

        with redirect_stdout(StringIO()):
            self.assertEqual(verify_upgrade.main(), 0)


if __name__ == "__main__":
    unittest.main()
