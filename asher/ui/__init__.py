"""Lazy access to the optional ASHER PySide6 frontend."""

from __future__ import annotations

import importlib.util


def is_available() -> bool:
    """Return whether the optional PySide6 dependency is installed."""

    return importlib.util.find_spec("PySide6") is not None


is_pyside6_available = is_available


def create_application(*args, **kwargs):
    from asher.ui.app import create_application as factory

    return factory(*args, **kwargs)


create_window = create_application


def run(*args, **kwargs):
    from asher.ui.app import run as runner

    return runner(*args, **kwargs)


def launch(*args, **kwargs):
    return run(*args, **kwargs)


run_ui = run


def __getattr__(name: str):
    if name == "CompanionDesktopController":
        from asher.ui.companion_adapter import CompanionDesktopController

        return CompanionDesktopController
    if name in {"AsherMainWindow", "AsherWindow"}:
        from asher.ui.window import AsherMainWindow, AsherWindow

        return {"AsherMainWindow": AsherMainWindow, "AsherWindow": AsherWindow}[name]
    if name in {
        "AsherUIController",
        "AuditRecord",
        "ConversationTurn",
        "DesktopController",
        "DesktopControllerProtocol",
        "DesktopSettings",
        "DesktopStatus",
        "DiagnosticReport",
        "LiveStep",
        "MemoryRecord",
        "PendingAction",
        "PermissionRecord",
        "UserRecord",
        "UIControllerProtocol",
    }:
        from asher.ui.controller import (
            AsherUIController,
            AuditRecord,
            ConversationTurn,
            DesktopController,
            DesktopControllerProtocol,
            DesktopSettings,
            DesktopStatus,
            DiagnosticReport,
            LiveStep,
            MemoryRecord,
            PendingAction,
            PermissionRecord,
            UserRecord,
            UIControllerProtocol,
        )

        return {
            "AsherUIController": AsherUIController,
            "AuditRecord": AuditRecord,
            "ConversationTurn": ConversationTurn,
            "DesktopController": DesktopController,
            "DesktopControllerProtocol": DesktopControllerProtocol,
            "DesktopSettings": DesktopSettings,
            "DesktopStatus": DesktopStatus,
            "DiagnosticReport": DiagnosticReport,
            "LiveStep": LiveStep,
            "MemoryRecord": MemoryRecord,
            "PendingAction": PendingAction,
            "PermissionRecord": PermissionRecord,
            "UserRecord": UserRecord,
            "UIControllerProtocol": UIControllerProtocol,
        }[name]
    raise AttributeError(name)


__all__ = [
    "AsherMainWindow",
    "AsherWindow",
    "CompanionDesktopController",
    "AsherUIController",
    "AuditRecord",
    "ConversationTurn",
    "DesktopController",
    "DesktopControllerProtocol",
    "DesktopSettings",
    "DesktopStatus",
    "DiagnosticReport",
    "LiveStep",
    "MemoryRecord",
    "PendingAction",
    "PermissionRecord",
    "UserRecord",
    "UIControllerProtocol",
    "create_application",
    "create_window",
    "is_available",
    "is_pyside6_available",
    "launch",
    "run",
    "run_ui",
]
