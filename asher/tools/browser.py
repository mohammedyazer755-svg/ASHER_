"""Safe browser navigation and search tools with observable window evidence."""

from __future__ import annotations

import time
import webbrowser
from typing import Any
from urllib.parse import quote_plus, urlparse

from asher.security.policy import ToolPolicy
from asher.tools.registry import ToolContext, ToolDefinition, successful_result
from asher.tools.windows import visible_windows
from asher.types import Evidence, RiskLevel, ToolResult


class BrowserTools:
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return (
            ToolDefinition(
                name="browser.search",
                description="Search Google or YouTube in the default browser.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 500},
                        "engine": {"type": "string", "enum": ["google", "youtube"]},
                    },
                    "required": ["query", "engine"],
                    "additionalProperties": False,
                },
                policy=ToolPolicy("public_web", RiskLevel.HARMLESS_LOCAL),
                timeout_seconds=15,
                handler=self.search,
                preview=lambda args: (args["engine"], f"Search {args['engine']} for {args['query']}", dict(args)),
            ),
            ToolDefinition(
                name="browser.navigate",
                description="Open an explicit HTTP or HTTPS URL in the default browser.",
                input_schema={
                    "type": "object",
                    "properties": {"url": {"type": "string", "minLength": 8, "maxLength": 2048}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
                policy=ToolPolicy("public_web", RiskLevel.HARMLESS_LOCAL),
                timeout_seconds=15,
                handler=self.navigate,
                preview=lambda args: (args["url"], f"Open {args['url']} in the default browser", {"url": args["url"]}),
            ),
        )

    def search(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        query = arguments["query"].strip()
        engine = arguments["engine"]
        encoded = quote_plus(query)
        url = (
            f"https://www.youtube.com/results?search_query={encoded}"
            if engine == "youtube"
            else f"https://www.google.com/search?q={encoded}"
        )
        return self._open(
            url,
            context,
            f"Searching {engine.title()} for {query}.",
            query=query,
            engine=engine,
        )

    def navigate(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        url = arguments["url"].strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return _failure(context, "invalid_url", "Only complete HTTP or HTTPS URLs are allowed.")
        if parsed.username or parsed.password:
            return _failure(context, "credentials_in_url", "URLs containing credentials are not allowed.")
        return self._open(url, context, f"Opened {parsed.netloc}.")

    def _open(
        self,
        url: str,
        context: ToolContext,
        message: str,
        *,
        query: str | None = None,
        engine: str | None = None,
    ) -> ToolResult:
        evidence_data: dict[str, Any] = {"url": url}
        if query:
            evidence_data["query"] = query
        if engine:
            evidence_data["engine"] = engine
        if context.dry_run:
            return successful_result(
                context.metadata["call_id"],
                context.metadata["tool_name"],
                f"Dry run verified browser navigation to {url}.",
                (Evidence("dry_run", "No browser navigation occurred", evidence_data),),
                dry_run=True,
            )
        before = {handle for handle, _ in visible_windows()}
        context.cancellation.raise_if_cancelled()
        if not webbrowser.open(url, new=2):
            return _failure(context, "dispatch_failed", "Windows did not accept the browser request.")
        deadline = time.monotonic() + 8
        observed = None
        while time.monotonic() < deadline:
            context.cancellation.raise_if_cancelled()
            browser_windows = [
                (handle, title)
                for handle, title in visible_windows()
                if any(name in title.casefold() for name in ("chrome", "edge", "firefox", "opera", "brave"))
            ]
            observed = next(((handle, title) for handle, title in browser_windows if handle not in before), None)
            if observed is None and browser_windows:
                observed = browser_windows[0]
            if observed:
                break
            time.sleep(0.2)
        if not observed:
            return _failure(context, "unverified_navigation", "The URL was dispatched, but no browser window was observed.")
        evidence_data["window_title"] = observed[1]
        return successful_result(
            context.metadata["call_id"],
            context.metadata["tool_name"],
            message,
            (Evidence("browser_window_observed", "A browser window was visible after navigation", evidence_data),),
        )


def _failure(context: ToolContext, code: str, message: str) -> ToolResult:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return ToolResult(
        call_id=context.metadata["call_id"], tool_name=context.metadata["tool_name"],
        success=False, status="failed", message=message, error_code=code,
        started_at=now, completed_at=now,
    )

