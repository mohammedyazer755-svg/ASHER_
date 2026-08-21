"""Deterministic, non-private vocabulary regression tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asher.voice.vocabulary import (
    DynamicVocabulary,
    ResolutionStatus,
    collapse_spelled_name,
)
from asher.voice.text_normalizer import collapse_spelled_name as legacy_collapse


class VocabularyTests(unittest.TestCase):
    def test_spelled_name_variants_collapse(self) -> None:
        for value in ("A V E R Y", "A-V-E-R-Y", "ay vee ee ar why"):
            collapsed, was_spelled = collapse_spelled_name(value)
            self.assertTrue(was_spelled)
            self.assertEqual(collapsed, "avery")
        self.assertEqual(legacy_collapse("A V E R Y"), "Avery")
        collapsed, was_spelled = collapse_spelled_name("double you ee en dee why")
        self.assertTrue(was_spelled)
        self.assertEqual(collapsed, "wendy")

    def test_exact_and_fuzzy_resolution(self) -> None:
        vocabulary = DynamicVocabulary(
            contacts=("Avery Stone", "Jordan Reed"),
            applications=("Text Editor",),
            application_aliases={"editor": "Text Editor"},
        )
        self.assertEqual(
            vocabulary.resolve_contact("A V E R Y").candidate,
            "Avery Stone",
        )
        self.assertEqual(
            vocabulary.resolve_contact("Jordon Reed").candidate,
            "Jordan Reed",
        )
        self.assertEqual(
            vocabulary.resolve_application("text editor").candidate,
            "Text Editor",
        )
        self.assertEqual(
            vocabulary.resolve_application("editor").candidate,
            "Text Editor",
        )

    def test_ambiguous_names_never_guess(self) -> None:
        vocabulary = DynamicVocabulary(
            contacts=("Alex Stone", "Alex Stone Jr"),
            minimum_score=0.55,
            ambiguity_margin=0.20,
        )
        result = vocabulary.resolve_contact("Alex")
        self.assertEqual(result.status, ResolutionStatus.AMBIGUOUS)
        self.assertIsNone(result.candidate)
        command = vocabulary.repair_contact_command("search Alex")
        self.assertFalse(command.executable)
        self.assertIn("Which contact", command.clarification or "")

    def test_close_fuzzy_names_use_margin_not_top_score_only(self) -> None:
        vocabulary = DynamicVocabulary(contacts=("Sara", "Sarah"))
        result = vocabulary.resolve_contact("Sarra")
        self.assertEqual(result.status, ResolutionStatus.AMBIGUOUS)
        self.assertIsNone(result.candidate)

    def test_send_repair_preserves_message_payload(self) -> None:
        vocabulary = DynamicVocabulary(contacts=("Avery Stone",))
        command = vocabulary.repair_contact_command(
            "send Hello WORLD! to A V E R Y"
        )
        self.assertTrue(command.executable)
        self.assertEqual(command.resolved_text, "send Hello WORLD! to Avery Stone")

    def test_alias_and_malformed_files_are_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contacts_path = Path(directory) / "contacts.json"
            contacts_path.write_text(
                json.dumps(
                    {
                        "contacts": ["Taylor North"],
                        "aliases": {"tay": "Taylor North"},
                    }
                ),
                encoding="utf-8",
            )
            vocabulary = DynamicVocabulary(contacts_path=contacts_path)
            self.assertEqual(vocabulary.resolve_contact("tay").candidate, "Taylor North")
            contacts_path.write_text("[]", encoding="utf-8")
            vocabulary.refresh(force=True)
            # A broken/incorrectly shaped optional file must not crash or erase
            # the explicit base vocabulary.
            self.assertEqual(vocabulary.contacts, ())

    def test_application_catalog_reloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            apps_path = Path(directory) / "apps.json"
            apps_path.write_text(json.dumps({"Editor": ["editor.exe"]}), encoding="utf-8")
            vocabulary = DynamicVocabulary(applications_path=apps_path)
            self.assertEqual(vocabulary.resolve_application("editor").candidate, "Editor")
            apps_path.write_text(json.dumps({"Calculator": ["calc.exe"]}), encoding="utf-8")
            vocabulary.refresh(force=True)
            self.assertIsNone(vocabulary.resolve_application("editor").candidate)
            self.assertEqual(
                vocabulary.resolve_application("calculator").candidate,
                "Calculator",
            )

    def test_windows_application_discovery_is_fixed_command_and_parseable(self) -> None:
        from asher.voice.vocabulary import discover_windows_applications

        class Result:
            returncode = 0
            stdout = '["Editor", "Calculator"]'

        with patch("asher.voice.vocabulary.subprocess.run", return_value=Result()) as run:
            self.assertEqual(discover_windows_applications(), ("Editor", "Calculator"))
            command = run.call_args.args[0]
            self.assertIn("Get-StartApps", command[-1])

    def test_explicit_installed_catalog_refresh_updates_vocabulary(self) -> None:
        vocabulary = DynamicVocabulary()
        with patch(
            "asher.voice.vocabulary.discover_windows_applications",
            return_value=("Synthetic Editor",),
        ):
            discovered = vocabulary.refresh_installed_applications()
        self.assertEqual(discovered, ("Synthetic Editor",))
        self.assertEqual(
            vocabulary.resolve_application("synthetic editor").candidate,
            "Synthetic Editor",
        )


if __name__ == "__main__":
    unittest.main()
