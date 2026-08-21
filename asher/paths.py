"""Private runtime paths kept separate from source code."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    database: Path
    audit_log: Path
    recordings: Path
    models: Path
    evaluations: Path
    screenshots: Path

    @classmethod
    def discover(cls, override: str | Path | None = None) -> "RuntimePaths":
        configured = override or os.getenv("ASHER_RUNTIME_DIR")
        if configured:
            root = Path(configured).expanduser().resolve()
        else:
            local_app_data = os.getenv("LOCALAPPDATA")
            base = Path(local_app_data) if local_app_data else Path.home() / ".local" / "share"
            root = (base / "Asher").resolve()

        return cls(
            root=root,
            database=root / "asher.db",
            audit_log=root / "audit.jsonl",
            recordings=root / "recordings",
            models=root / "models",
            evaluations=root / "evaluations",
            screenshots=root / "screenshots",
        )

    def ensure(self) -> "RuntimePaths":
        for path in (
            self.root,
            self.recordings,
            self.models,
            self.evaluations,
            self.screenshots,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

