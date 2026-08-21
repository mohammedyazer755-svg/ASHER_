"""Local Qwen action planner for Asher.

Whisper turns speech into text.  This module turns that text into a small,
validated list of actions.  It never executes Windows actions itself.
"""

import json
from typing import Literal

from ollama import chat
from pydantic import BaseModel, Field


MODEL_NAME = "qwen3:4b"


ActionName = Literal[
    "open_app",
    "close_app",
    "google_search",
    "youtube_search",
    "whatsapp_search",
    "send_whatsapp",
    "volume_up",
    "volume_down",
    "toggle_mute",
    "take_screenshot",
    "request_lock",
    "speak",
    "unknown",
]


class PlannedAction(BaseModel):
    action: ActionName
    target: str = ""
    message: str = ""


class AssistantPlan(BaseModel):
    actions: list[PlannedAction] = Field(default_factory=list, max_length=8)
    response: str = ""


SYSTEM_PROMPT = """
You are the action planner inside Asher, a Windows voice assistant.
Convert the user's command into the smallest safe ordered list of actions.
Return only data matching the supplied JSON schema. Never invent tool names.

Available actions:
- open_app: target is an installed application name.
- close_app: target is an application name.
- google_search: target is the search query.
- youtube_search: target is the video search query.
- whatsapp_search: target is a WhatsApp contact name.
- send_whatsapp: target is the contact and message is the message text.
- volume_up, volume_down, toggle_mute, take_screenshot.
- request_lock: request confirmation before locking Windows.
- speak: message is a short clarification or answer.
- unknown: only when none of the available actions can help.

Rules:
1. Output English only, even if speech recognition produced unusual text.
2. Preserve names and message text exactly when possible.
3. Resolve him, her, them, that contact and the same person using last_contact.
4. Resolve it, that app and the application using last_app.
5. If WhatsApp is the current app and the user says search followed by a name,
   use whatsapp_search, not google_search.
6. For a compound command, return the actions in execution order.
7. Sending WhatsApp messages and locking Windows always require confirmation;
   the executor handles that confirmation.
8. If essential information is missing, use speak to ask one short question.
9. Do not claim an action succeeded. The executor reports the result.
""".strip()


class PlannerError(RuntimeError):
    """Raised when Ollama cannot produce a valid action plan."""


def plan_command(command: str, context: dict | None = None) -> AssistantPlan:
    context = context or {}

    user_message = (
        f"Current context: {json.dumps(context, ensure_ascii=False)}\n"
        f"User command: {command}"
    )

    try:
        response = chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            format=AssistantPlan.model_json_schema(),
            options={
                "temperature": 0,
                "num_predict": 450,
            },
            think=False,
            stream=False,
            keep_alive="10m",
        )

        return AssistantPlan.model_validate_json(response.message.content)

    except Exception as error:
        raise PlannerError(str(error)) from error


if __name__ == "__main__":
    print("Asher AI planner test. Type exit to close.")

    while True:
        text = input("You: ").strip()

        if text.lower() in {"exit", "quit", "bye"}:
            break

        try:
            print(plan_command(text).model_dump_json(indent=2))
        except PlannerError as error:
            print(f"Planner error: {error}")
