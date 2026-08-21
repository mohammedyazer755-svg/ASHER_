"""Application factory for the optional PySide6 desktop frontend."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from typing import Any


def require_pyside6() -> Any:
    if importlib.util.find_spec("PySide6") is None:
        raise RuntimeError(
            "PySide6 is required for the ASHER desktop UI. "
            "Install PySide6 in the project environment and retry."
        )
    from PySide6.QtWidgets import QApplication

    return QApplication


def create_application(
    argv: Sequence[str] | None = None,
    *,
    controller: Any | None = None,
) -> tuple[Any, Any]:
    QApplication = require_pyside6()
    from asher.ui.window import AsherMainWindow

    app = QApplication.instance()
    if app is None:
        app = QApplication(list(argv) if argv is not None else sys.argv[:1])
    window = AsherMainWindow(controller=controller)
    return app, window


create_window = create_application


def run(argv: Sequence[str] | None = None, *, controller: Any | None = None) -> int:
    app, window = create_application(argv, controller=controller)
    window.show()
    return int(app.exec())


launch = run

__all__ = ["create_application", "create_window", "launch", "require_pyside6", "run"]
