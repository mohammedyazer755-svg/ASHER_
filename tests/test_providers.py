from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asher.brain.deterministic import DeterministicPlanner
from asher.brain.plans import PlanValidationError, validate_plan_payload
from asher.brain.providers import HybridPlanner, OllamaProvider, OpenAIResponsesProvider, ProviderError
from asher.config import AsherConfig
from asher.storage import Database
from asher.tools.catalog import build_registry


class _Response:
    def __init__(self, text: str) -> None:
        self.output_text = text


class _OpenAIResponses:
    def __init__(self, text: str) -> None:
        self.text = text
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _Response(self.text)


class _OpenAIClient:
    def __init__(self, text: str) -> None:
        self.responses = _OpenAIResponses(text)


class _HTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class ProviderTests(unittest.TestCase):
    def test_openai_responses_uses_strict_schema_without_session_payload(self) -> None:
        payload = {"goal": "say hello", "response": "Ready", "steps": []}
        client = _OpenAIClient(json.dumps(payload))
        with tempfile.TemporaryDirectory() as directory:
            config = AsherConfig.load(directory)
            registry = build_registry(config, Database(Path(directory) / "db.sqlite"))
            provider = OpenAIResponsesProvider(client=client)
            result = provider.plan(
                "hello with api_key=should-not-leak",
                context={"session": object(), "memory": "minimal"},
                tool_schemas=[],
                registry=registry,
            )
        self.assertEqual(result.response, "Ready")
        self.assertFalse(client.responses.kwargs["store"])
        self.assertNotIn("session", client.responses.kwargs["input"])
        self.assertNotIn("should-not-leak", client.responses.kwargs["input"])


    def test_qwen35_9b_is_the_default_local_model_and_timeout_allows_cold_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {}, clear=True):
                config = AsherConfig.load(directory)
        self.assertEqual(config.ollama_model, "qwen3.5:9b")
        provider = OllamaProvider()
        self.assertEqual(provider.model, "qwen3.5:9b")
        self.assertGreaterEqual(provider.timeout, 60.0)

    def test_ollama_planner_disables_long_thinking_for_strict_schema_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = AsherConfig.load(directory)
            registry = build_registry(config, Database(Path(directory) / "db.sqlite"))
            provider = OllamaProvider(model="qwen3.5:9b")
            captured = {}

            def fake_urlopen(request, timeout):
                captured["body"] = json.loads(request.data.decode("utf-8"))
                captured["timeout"] = timeout
                return _HTTPResponse(
                    {"message": {"content": json.dumps({"goal": "answer", "response": "Ready", "steps": []})}}
                )

            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                result = provider.plan("hello", context={}, tool_schemas=[], registry=registry)

        self.assertEqual(result.response, "Ready")
        self.assertFalse(captured["body"]["think"])
        self.assertEqual(captured["body"]["options"]["num_ctx"], 8192)
        self.assertGreaterEqual(captured["timeout"], 60.0)


    def test_ollama_conversation_collapses_harmless_no_tool_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = AsherConfig.load(directory)
            registry = build_registry(config, Database(Path(directory) / "db.sqlite"))
            provider = OllamaProvider(model="qwen3.5:9b")
            payload = {
                "goal": "explain AI agents",
                "response": "An AI agent is software that can reason about a goal and take actions toward it.",
                "steps": [
                    {
                        "id": "answer",
                        "tool": "none",
                        "arguments": {},
                        "description": "No tool is required",
                        "depends_on": [],
                    }
                ],
            }
            with patch(
                "urllib.request.urlopen",
                return_value=_HTTPResponse({"message": {"content": json.dumps(payload)}}),
            ):
                result = provider.plan(
                    "Explain what an AI agent is in one sentence.",
                    context={},
                    tool_schemas=[],
                    registry=registry,
                )

        self.assertEqual(result.steps, ())
        self.assertIn("AI agent", result.response)

    def test_ollama_unknown_real_tool_still_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = AsherConfig.load(directory)
            registry = build_registry(config, Database(Path(directory) / "db.sqlite"))
            provider = OllamaProvider(model="qwen3.5:9b")
            payload = {
                "goal": "do something",
                "response": "Trying.",
                "steps": [
                    {
                        "id": "bad",
                        "tool": "definitely.not.a.real.tool",
                        "arguments": {},
                        "description": "Invalid tool",
                        "depends_on": [],
                    }
                ],
            }
            with patch(
                "urllib.request.urlopen",
                return_value=_HTTPResponse({"message": {"content": json.dumps(payload)}}),
            ):
                with self.assertRaises(ProviderError):
                    provider.plan("Do something.", context={}, tool_schemas=[], registry=registry)

    def test_ollama_invalid_structured_output_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = AsherConfig.load(directory)
            registry = build_registry(config, Database(Path(directory) / "db.sqlite"))
            provider = OllamaProvider(timeout=0.1)
            with patch(
                "urllib.request.urlopen",
                return_value=_HTTPResponse({"message": {"content": "{bad json"}}),
            ):
                with self.assertRaises(ProviderError):
                    provider.plan("do something", context={}, tool_schemas=[], registry=registry)

    def test_hybrid_planner_returns_safe_fallback_when_providers_fail(self) -> None:
        class Missing:
            name = "missing"
            offline = True

            def plan(self, **_kwargs):
                raise ProviderError("unavailable")

        planner = HybridPlanner(
            deterministic=type("Never", (), {"plan": lambda *_args, **_kwargs: None})(),
            registry=type("Registry", (), {"schemas_for": lambda *_args: []})(),
            ollama=Missing(),
        )
        result = planner.plan("an unsupported multi-step request")
        self.assertTrue(result.offline)
        self.assertEqual(result.provider, "fallback")
        self.assertNotIn("password", result.response.casefold())

    def test_hybrid_planner_skips_remote_provider_when_disabled(self) -> None:
        class TrackingProvider:
            name = "remote"
            offline = False

            def __init__(self) -> None:
                self.called = False

            def plan(self, *_args, **_kwargs):
                self.called = True
                raise AssertionError("remote provider must not be called")

        remote = TrackingProvider()
        planner = HybridPlanner(
            deterministic=type("Never", (), {"plan": lambda *_args, **_kwargs: None})(),
            registry=type("Registry", (), {"schemas_for": lambda *_args: []})(),
            openai=remote,
            online_enabled=False,
        )
        result = planner.plan("an unsupported request")
        self.assertFalse(remote.called)
        self.assertTrue(result.offline)

    def test_provider_step_dependencies_use_explicit_earlier_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = AsherConfig.load(directory)
            registry = build_registry(config, Database(Path(directory) / "db.sqlite"))
            payload = {
                "goal": "open then close",
                "response": "",
                "steps": [
                    {
                        "id": "open_browser",
                        "tool": "app.open",
                        "arguments": {"app_name": "chrome"},
                        "description": "Open Chrome",
                        "depends_on": [],
                    },
                    {
                        "id": "close_browser",
                        "tool": "app.close",
                        "arguments": {"app_name": "chrome"},
                        "description": "Close Chrome",
                        "depends_on": ["open_browser"],
                    },
                ],
            }
            plan = validate_plan_payload(
                payload,
                registry,
                provider="fixture",
                offline=True,
            )
            self.assertEqual(plan.steps[1].depends_on, ("open_browser",))
            payload["steps"][0]["depends_on"] = ["close_browser"]
            with self.assertRaises(PlanValidationError):
                validate_plan_payload(
                    payload,
                    registry,
                    provider="fixture",
                    offline=True,
                )


if __name__ == "__main__":
    unittest.main()
