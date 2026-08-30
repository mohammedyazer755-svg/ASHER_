"""Fast intent patterns for common commands and safe compound goals."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from asher.brain.plans import ProposedPlan
from asher.types import PlanStep, ToolCall


@dataclass(frozen=True)
class ContactResolution:
    canonical: str | None
    alternatives: tuple[str, ...] = ()

    @property
    def ambiguous(self) -> bool:
        return self.canonical is None and bool(self.alternatives)


class ContactResolver:
    """Fuzzy/phonetic contact resolver with an ambiguity margin."""

    def __init__(self, names: tuple[str, ...] = (), aliases: dict[str, str] | None = None) -> None:
        self.names = tuple(name.strip() for name in names if name.strip())
        self.aliases = {key.casefold().strip(): value.strip() for key, value in (aliases or {}).items()}

    def resolve(self, raw: str, *, threshold: float = 0.68, margin: float = 0.10) -> ContactResolution:
        import difflib
        from voice.text_normalizer import normalise_contact_name

        candidate = normalise_contact_name(raw)
        lowered = candidate.casefold().strip()
        if lowered in self.aliases:
            candidate = self.aliases[lowered]
        if not self.names:
            return ContactResolution(candidate or None)
        exact = [name for name in self.names if name.casefold() == lowered or name.casefold() == candidate.casefold()]
        if len(exact) == 1:
            return ContactResolution(exact[0])
        compact = "".join(character for character in lowered if character.isalnum())
        scores = sorted(
            (
                difflib.SequenceMatcher(None, compact, "".join(character for character in name.casefold() if character.isalnum())).ratio(),
                name,
            )
            for name in self.names
        )
        scores.reverse()
        if not scores or scores[0][0] < threshold:
            return ContactResolution(candidate.title() if candidate else None)
        if len(scores) > 1 and scores[0][0] - scores[1][0] < margin:
            return ContactResolution(None, tuple(name for _, name in scores[:3]))
        return ContactResolution(scores[0][1])


def _step(tool: str, arguments: dict, description: str) -> PlanStep:
    return PlanStep(ToolCall(tool, arguments), description)


class DeterministicPlanner:
    def __init__(self, resolver: ContactResolver | None = None) -> None:
        self.resolver = resolver or ContactResolver()

    def plan(
        self,
        command: str,
        *,
        last_app: str = "",
        last_contact: str = "",
        last_search_query: str = "",
    ) -> ProposedPlan | None:
        original = command.strip()
        lowered = original.casefold().strip().rstrip(".,!?;:")
        if not lowered:
            return ProposedPlan(goal="", response="Please tell me what you would like me to do.")
        if lowered in {"cancel", "stop", "never mind", "nevermind"}:
            return ProposedPlan(goal=original, response="Cancelled.")
        if lowered in {"emergency stop", "stop everything", "asher emergency stop"}:
            return ProposedPlan(goal=original, response="Emergency stop requested.")

        search_follow_up = re.match(
            r"^(?:(?:search|look\s+up)\s+(?:that|it|the\s+same(?:\s+(?:thing|topic|query))?)|look\s+(?:that|it)\s+up)\s+(?:on|in)\s+(youtube|google)$|"
            r"^(?:search|look\s+up)\s+(?:on\s+)?(youtube|google)\s+(?:for\s+)?(?:that|it|the\s+same(?:\s+(?:thing|topic|query))?)$|"
            r"^(youtube|google)\s+(?:that|it|the\s+same(?:\s+(?:thing|topic|query))?)$",
            lowered,
        )
        if search_follow_up:
            engine = (search_follow_up.group(1) or search_follow_up.group(2) or search_follow_up.group(3) or "google").casefold()
            if not last_search_query.strip():
                return ProposedPlan(
                    original,
                    (),
                    "Which search topic would you like me to use?",
                )
            return ProposedPlan(
                original,
                (
                    _step(
                        "browser.search",
                        {"query": last_search_query.strip(), "engine": engine},
                        f"Search {engine.title()} for the previous topic",
                    ),
                ),
            )

        compound = self._compound(
            lowered,
            original,
            last_app=last_app,
            last_contact=last_contact,
            last_search_query=last_search_query,
        )
        if compound is not None:
            return compound

        patterns: tuple[tuple[re.Pattern[str], Callable[[re.Match[str], str], ProposedPlan]], ...] = (
            (re.compile(r"^(?:remember that|remember:|remember)\s+(.+?)\s+(?:is|=|:)\s+(.+)$"), lambda m, _: ProposedPlan(original, (_step("memory.put", {"memory_type": "user_preference", "key": m.group(1).strip(), "value": m.group(2).strip(), "sensitivity": "normal"}, f"Save memory for {m.group(1).strip()}"),))),
            (re.compile(r"^(?:what is my|what's my|do you remember|recall|search memory for|find in memory)\s+(.+)$"), lambda m, _: ProposedPlan(original, (_step("memory.search", {"query": m.group(1).strip()}, "Search relevant private memory"),))),
            (re.compile(r"^(?:delete memory|forget memory|remove memory)\s+(?:id\s+)?([a-f0-9\-]+)$"), lambda m, _: ProposedPlan(original, (_step("memory.delete", {"memory_id": m.group(1).strip()}, f"Delete memory {m.group(1).strip()}"),))),
            (re.compile(r"^(?:open|launch|start)\s+(.+)$"), lambda m, _: ProposedPlan(original, (_step("app.open", {"app_name": m.group(1).strip()}, f"Open {m.group(1).strip()}"),))),
            (re.compile(r"^(?:close|quit|exit)\s+(.+)$"), lambda m, _: ProposedPlan(original, (_step("app.close", {"app_name": m.group(1).strip()}, f"Close {m.group(1).strip()}"),))),
            (re.compile(r"^(?:search google for|google search for|search for|look up|google)\s+(.+)$"), lambda m, _: ProposedPlan(original, (_step("browser.search", {"query": m.group(1).strip(), "engine": "google"}, "Search Google"),))),
            (re.compile(r"^(?:search youtube for|search on youtube for|youtube search for|youtube)\s+(.+)$"), lambda m, _: ProposedPlan(original, (_step("browser.search", {"query": m.group(1).strip(), "engine": "youtube"}, "Search YouTube"),))),
            (re.compile(r"^(?:increase|raise|turn up) (?:the )?volume$"), lambda _, __: ProposedPlan(original, (_step("system.volume_up", {}, "Increase volume"),))),
            (re.compile(r"^(?:decrease|lower|turn down) (?:the )?volume$"), lambda _, __: ProposedPlan(original, (_step("system.volume_down", {}, "Decrease volume"),))),
            (re.compile(r"^(?:toggle )?mute(?: volume)?$|^unmute(?: volume)?$"), lambda _, __: ProposedPlan(original, (_step("system.toggle_mute", {}, "Toggle mute"),))),
            (re.compile(r"^(?:take (?:a )?)?screenshot$|^capture (?:my )?screen$"), lambda _, __: ProposedPlan(original, (_step("system.screenshot", {}, "Capture screenshot"),))),
        )
        for pattern, builder in patterns:
            match = pattern.match(lowered)
            if match:
                return builder(match, original)

        message = self._message_goal(original, last_contact)
        if message is not None:
            return message

        whatsapp_search = re.match(r"^(?:search|find|touch|such)(?: whatsapp)?\s+(.+)$", lowered)
        if whatsapp_search:
            raw_contact = original[len(original) - len(whatsapp_search.group(1)):].strip()
            # A bare search is a contact lookup only when the local vocabulary
            # can resolve it.  Unknown text remains a browser query rather
            # than being guessed as a person.
            if self.resolver.names:
                resolution = self.resolver.resolve(raw_contact)
                if resolution.ambiguous or (
                    resolution.canonical
                    and any(item.casefold() == resolution.canonical.casefold() for item in self.resolver.names)
                ):
                    return self._contact_plan(original, raw_contact, "whatsapp.prepare", "Prepare WhatsApp contact")
            return ProposedPlan(
                original,
                (_step("browser.search", {"query": raw_contact, "engine": "google"}, "Search Google"),),
            )

        if lowered in {"hi", "hello", "hey", "how are you", "how are you doing"}:
            return ProposedPlan(original, (), "Hi. I’m here and ready to help.")
        return None

    def _contact_plan(self, goal: str, raw: str, tool: str, description: str) -> ProposedPlan:
        resolution = self.resolver.resolve(raw)
        if resolution.ambiguous:
            return ProposedPlan(goal, (), f"Which contact did you mean: {' or '.join(resolution.alternatives)}?")
        if not resolution.canonical or not any(
            item.casefold() == resolution.canonical.casefold() for item in self.resolver.names
        ):
            return ProposedPlan(goal, (), "I could not identify that contact confidently. Please say the full name.")
        return ProposedPlan(goal, (_step(tool, {"contact": resolution.canonical}, description),))

    def _message_goal(self, original: str, last_contact: str) -> ProposedPlan | None:
        lowered = original.casefold()
        match = re.match(r"^send(?: a message| message)?\s+(.+?)\s+to\s+(.+)$", original, re.IGNORECASE)
        if match:
            message = match.group(1).strip()
            contact = match.group(2).strip()
        else:
            indirect = re.match(r"^(?:ask|tell)\s+(.+?)\s+(whether|if)\s+(.+)$", original, re.IGNORECASE)
            if not indirect:
                return None
            contact = indirect.group(1).strip()
            proposition = indirect.group(3).strip()
            # Turn a third-person question into a natural direct message when
            # the user supplied a pronoun ("ask Sai whether he is ready" ->
            # "Hey Sai, are you ready?").  Do not rewrite arbitrary payloads.
            proposition = re.sub(r"^(?:he|she)\s+is\b", "are you", proposition, flags=re.IGNORECASE)
            proposition = re.sub(r"^they\s+are\b", "are you", proposition, flags=re.IGNORECASE)
            proposition = re.sub(r"^(?:he|she|they)\s+(can|will|would|could|should)\b", r"\1 you", proposition, flags=re.IGNORECASE)
            message = f"Hey {contact}, {proposition}?"
            if not message.endswith("?"):
                message += "?"
        if contact.casefold() in {"him", "her", "them", "that contact", "the same person"}:
            contact = last_contact
        resolution = self.resolver.resolve(contact)
        if resolution.ambiguous:
            return ProposedPlan(original, (), f"Which contact did you mean: {' or '.join(resolution.alternatives)}?")
        if (
            not resolution.canonical
            or not any(item.casefold() == resolution.canonical.casefold() for item in self.resolver.names)
            or not message
        ):
            return ProposedPlan(original, (), "Please tell me both the recipient and the message.")
        return ProposedPlan(
            original,
            (
                _step("whatsapp.prepare", {"contact": resolution.canonical}, f"Prepare the chat for {resolution.canonical}"),
                _step("whatsapp.send", {"contact": resolution.canonical, "message": message}, f"Preview and send the message to {resolution.canonical}"),
            ),
        )

    def _compound(
        self,
        lowered: str,
        original: str,
        *,
        last_app: str,
        last_contact: str,
        last_search_query: str,
    ) -> ProposedPlan | None:
        # Split only on command connectors, retaining payload punctuation.
        if not any(token in lowered for token in (" and ", ",", " then ", " after that ")):
            return None
        pieces = [piece.strip(" ,") for piece in re.split(r"\s+then\s+|\s+after that\s+|\s+and\s+|,", original, flags=re.IGNORECASE) if piece.strip(" ,")]
        if len(pieces) < 2:
            return None
        steps: list[PlanStep] = []
        responses: list[str] = []
        for piece in pieces:
            plan = self.plan(
                piece,
                last_app=last_app,
                last_contact=last_contact,
                last_search_query=last_search_query,
            )
            if plan is None or not plan.steps:
                return None
            steps.extend(plan.steps)
            if plan.response:
                responses.append(plan.response)
        return ProposedPlan(original, tuple(steps), " ".join(responses))
