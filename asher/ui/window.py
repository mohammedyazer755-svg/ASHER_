"""Responsive PySide6 desktop shell for ASHER.

The widgets intentionally contain presentation and event wiring only. All
state changes go through :class:`DesktopControllerProtocol`, and controller
calls are dispatched to ``QThreadPool`` workers so recognition/tool adapters
cannot freeze the GUI.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

from asher.core.state import AssistantState
from asher.types import RiskLevel
from asher.ui.controller import (
    AuditRecord,
    ConversationTurn,
    DesktopController,
    DesktopControllerProtocol,
    DesktopSettings,
    DiagnosticReport,
    LiveStep,
    MemoryRecord,
    PendingAction,
    PermissionRecord,
    UserRecord,
)
from asher.ui.orb_widget import visual_for_state
from asher.ui.web_orb_widget import CompanionOrbHost
from asher.ui.workers import FunctionWorker, QT_WORKERS_AVAILABLE


try:
    from PySide6.QtCore import (
        QAbstractAnimation,
        QEasingCurve,
        QObject,
        QPointF,
        QPropertyAnimation,
        QRectF,
        QSize,
        QThreadPool,
        QTimer,
        Qt,
        Signal,
    )
    from PySide6.QtGui import (
        QBrush,
        QColor,
        QConicalGradient,
        QFont,
        QFontMetrics,
        QIcon,
        QKeySequence,
        QLinearGradient,
        QPainter,
        QPainterPath,
        QPalette,
        QPen,
        QRadialGradient,
        QShortcut,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSpinBox,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextBrowser,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
        QInputDialog,
        QHeaderView,
    )

    QT_AVAILABLE = QT_WORKERS_AVAILABLE
except ImportError as _qt_error:  # pragma: no cover - dependency-free path
    _QT_IMPORT_ERROR = _qt_error
    QT_AVAILABLE = False


def _require_qt() -> None:
    if not QT_AVAILABLE:
        raise RuntimeError(
            "PySide6 is required for the ASHER desktop UI. "
            "Install PySide6 in the project environment and retry."
        ) from _QT_IMPORT_ERROR


_COMPANION_ACTIVE_STATES = frozenset({
    AssistantState.WAKE_DETECTED,
    AssistantState.AUTHENTICATING,
    AssistantState.AUTHENTICATED,
    AssistantState.LOCKED,
    AssistantState.LISTENING,
    AssistantState.TRANSCRIBING,
    AssistantState.THINKING,
    AssistantState.AWAITING_CONFIRMATION,
    AssistantState.EXECUTING,
    AssistantState.OBSERVING,
    AssistantState.SPEAKING,
    AssistantState.SUCCESS,
    AssistantState.ERROR,
})


def should_use_companion_mode(state: AssistantState, microphone_active: bool) -> bool:
    """Return True only for an active voice interaction.

    Text-only planning never forces the immersive scene. The companion scene is
    a presentation mode layered on top of the same controller and safety policy.
    """

    return bool(microphone_active and state in _COMPANION_ACTIVE_STATES)


if not QT_AVAILABLE:

    class AsherMainWindow:  # type: ignore[no-redef]
        """Import-safe placeholder when optional Qt is not installed."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_qt()


else:

    from asher.ui.home_companion_widget import (
        FemaleCompanion,
        HomeCompanionHost,
        MaleCompanion,
    )

    APP_STYLE = """
    /* UI-5B — ASHER reference-inspired premium glass workspace.
       Palette: obsidian / smoke-violet glass / electric blue / violet / magenta.
       Companion-mode selectors remain visually isolated below. */

    QWidget {
        color: #F3F0FF;
        font-family: "Segoe UI", Arial;
        font-size: 10pt;
        background: transparent;
    }

    QMainWindow, QDialog, QWidget#workspace {
        background: qradialgradient(
            cx: 0.63, cy: 0.34, radius: 1.05,
            fx: 0.63, fy: 0.34,
            stop: 0 rgba(38, 31, 70, 255),
            stop: 0.30 rgba(16, 18, 36, 255),
            stop: 0.72 rgba(8, 10, 21, 255),
            stop: 1 rgba(4, 6, 13, 255)
        );
    }

    QFrame#sidebar {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(28, 29, 48, 222),
            stop: 0.55 rgba(17, 20, 35, 228),
            stop: 1 rgba(12, 15, 28, 235)
        );
        border: 1px solid rgba(204, 195, 255, 54);
        border-radius: 22px;
    }

    QFrame#header {
        background: rgba(25, 25, 45, 190);
        border: 1px solid rgba(202, 193, 255, 48);
        border-radius: 18px;
    }

    QFrame#card, QGroupBox {
        background: rgba(24, 25, 43, 182);
        border: 1px solid rgba(191, 183, 237, 40);
        border-radius: 17px;
    }

    QFrame#card:hover {
        background: rgba(29, 31, 52, 194);
        border-color: rgba(132, 118, 255, 78);
    }

    QGroupBox {
        margin-top: 12px;
        padding: 14px;
        font-weight: 600;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 7px;
        color: #C7C0DE;
        background: transparent;
    }

    QLabel#brand {
        color: #F7F4FF;
        font-size: 18pt;
        font-weight: 750;
        letter-spacing: 2px;
    }

    QLabel#pageTitle {
        color: #F8F6FF;
        font-size: 19pt;
        font-weight: 700;
    }

    QLabel#subtitle, QLabel#muted {
        color: #9F9AB5;
    }

    QLabel#state {
        color: #8DE8D0;
        font-size: 11pt;
        font-weight: 750;
    }

    QLabel#orbState {
        color: #F8F6FF;
        font-size: 16pt;
        font-weight: 750;
        letter-spacing: 2px;
    }

    QLabel#orbMessage {
        color: #A9A4BC;
        font-size: 10pt;
    }

    QLabel#eyebrow {
        color: #9B8CFF;
        font-size: 8pt;
        font-weight: 750;
        letter-spacing: 3px;
    }

    QLabel#heroTitle {
        color: #F9F7FF;
        font-size: 30pt;
        font-weight: 800;
        letter-spacing: 2px;
    }

    QLabel#heroAccent {
        color: #6FC8FF;
        font-size: 30pt;
        font-weight: 800;
        letter-spacing: 1px;
    }

    QLabel#heroSubtitle {
        color: #F2EEFF;
        font-size: 16pt;
        font-weight: 650;
    }

    QLabel#heroDescription {
        color: #AAA5BA;
        font-size: 10pt;
        line-height: 1.45;
    }

    QLabel#heroState {
        color: #C6B8FF;
        font-size: 10pt;
        font-weight: 750;
        letter-spacing: 2px;
    }

    QLabel#heroMessage {
        color: #B9B4C7;
        font-size: 9pt;
    }

    QLabel#miniChip {
        background: rgba(37, 39, 63, 164);
        color: #C9C4D8;
        border: 1px solid rgba(180, 171, 233, 34);
        border-radius: 10px;
        padding: 6px 9px;
        font-size: 8pt;
    }

    QLabel#replyPreview {
        background: rgba(14, 16, 29, 150);
        color: #C7C1D6;
        border-left: 2px solid #7C5CFF;
        border-radius: 8px;
        padding: 9px 10px;
        font-size: 9pt;
    }

    QFrame#hero {
        background: qradialgradient(
            cx: 0.60, cy: 0.46, radius: 0.86,
            fx: 0.60, fy: 0.46,
            stop: 0 rgba(67, 72, 154, 70),
            stop: 0.27 rgba(90, 55, 156, 44),
            stop: 0.58 rgba(27, 27, 52, 210),
            stop: 1 rgba(13, 15, 28, 236)
        );
        border: 1px solid rgba(216, 207, 255, 64);
        border-radius: 24px;
    }

    QFrame#orbPresentation {
        background: qradialgradient(
            cx: 0.50, cy: 0.48, radius: 0.78,
            fx: 0.50, fy: 0.48,
            stop: 0 rgba(79, 157, 255, 38),
            stop: 0.34 rgba(124, 92, 255, 22),
            stop: 0.66 rgba(217, 70, 239, 8),
            stop: 1 rgba(0, 0, 0, 0)
        );
        border: 0;
        border-radius: 22px;
    }

    QFrame#companionSelectorShell {
        background: transparent;
        border: 0;
    }
    QFrame#companionSelectorPill {
        background: rgba(14, 18, 32, 190);
        border: 1px solid rgba(120, 140, 210, 55);
        border-radius: 14px;
    }
    QLabel#companionSelectorLabel {
        color: #8E9AC0;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.8px;
        padding-left: 6px;
        padding-right: 4px;
    }
    QPushButton#companionSelectorBtn {
        background: transparent;
        color: #94A3B8;
        border: 0;
        border-radius: 11px;
        padding: 3px 11px;
        font-size: 11px;
        font-weight: 600;
    }
    QPushButton#companionSelectorBtn:hover {
        background: rgba(100, 150, 255, 35);
        color: #FFFFFF;
    }
    QPushButton#companionSelectorBtn:checked {
        background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #00B0FF, stop: 1 #7852FF);
        color: #FFFFFF;
        font-weight: 700;
    }

    QFrame#voicePanel {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(39, 36, 61, 202),
            stop: 0.58 rgba(26, 27, 47, 210),
            stop: 1 rgba(18, 20, 35, 220)
        );
        border: 1px solid rgba(205, 194, 255, 62);
        border-radius: 19px;
    }

    QLabel#panelTitle {
        color: #F1EDFA;
        font-size: 10pt;
        font-weight: 700;
    }

    QLabel#panelKicker {
        color: #D65CFF;
        font-size: 8pt;
        font-weight: 750;
        letter-spacing: 2px;
    }

    QLabel#panelState {
        color: #B7A5FF;
        font-size: 14pt;
        font-weight: 750;
    }

    QLabel#panelMessage {
        color: #9F9AB1;
        font-size: 9pt;
    }

    QFrame#statTile {
        background: rgba(16, 18, 31, 155);
        border: 1px solid rgba(172, 161, 220, 32);
        border-radius: 12px;
    }

    QLabel#statTitle {
        color: #827C96;
        font-size: 7.5pt;
        font-weight: 700;
        letter-spacing: 1px;
    }

    QLabel#statValue {
        color: #F3F0FA;
        font-size: 10.5pt;
        font-weight: 650;
    }

    QPushButton {
        background: rgba(38, 40, 65, 175);
        color: #E9E5F4;
        border: 1px solid rgba(184, 174, 232, 48);
        border-radius: 11px;
        padding: 8px 13px;
    }

    QPushButton:hover {
        background: rgba(52, 50, 82, 200);
        border-color: rgba(157, 137, 255, 95);
        color: #FFFFFF;
    }

    QPushButton:pressed {
        background: rgba(31, 31, 55, 220);
        border-color: rgba(124, 92, 255, 120);
    }

    QPushButton:disabled {
        color: rgba(194, 189, 210, 80);
        background: rgba(27, 28, 45, 90);
        border-color: rgba(149, 142, 180, 28);
    }

    QPushButton#primary {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 #4F9DFF,
            stop: 0.48 #765BFF,
            stop: 1 #D946EF
        );
        color: #FFFFFF;
        border: 1px solid rgba(221, 210, 255, 125);
        border-radius: 12px;
        font-weight: 750;
        padding: 9px 15px;
    }

    QPushButton#primary:hover {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 #67B5FF,
            stop: 0.48 #8B6DFF,
            stop: 1 #E65AF5
        );
        border-color: rgba(237, 229, 255, 180);
    }

    QPushButton#secondaryAction {
        background: rgba(37, 38, 61, 185);
        color: #ECE8F6;
        border: 1px solid rgba(189, 180, 231, 65);
        font-weight: 650;
        border-radius: 12px;
        padding: 9px 15px;
    }

    QPushButton#dangerButton {
        background: rgba(122, 30, 48, 205);
        color: #FFE8EC;
        border: 1px solid rgba(235, 92, 118, 145);
        font-weight: 750;
    }

    QPushButton#dangerButton:hover {
        background: rgba(159, 38, 62, 225);
        border-color: rgba(255, 112, 138, 185);
    }

    QPushButton#nav {
        text-align: left;
        border: 1px solid transparent;
        background: transparent;
        border-radius: 12px;
        padding: 10px 12px;
        color: #AAA5BB;
        font-weight: 550;
    }

    QPushButton#nav:hover {
        background: rgba(92, 74, 156, 48);
        border-color: rgba(139, 113, 238, 40);
        color: #EAE5F6;
    }

    QPushButton#nav:checked {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 rgba(122, 78, 219, 115),
            stop: 0.58 rgba(97, 75, 196, 74),
            stop: 1 rgba(68, 104, 180, 42)
        );
        border: 1px solid rgba(170, 128, 255, 92);
        color: #FFFFFF;
    }

    QPushButton#featureCard {
        text-align: left;
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(42, 40, 67, 190),
            stop: 0.58 rgba(28, 29, 48, 190),
            stop: 1 rgba(19, 21, 36, 205)
        );
        border: 1px solid rgba(188, 177, 230, 45);
        border-radius: 15px;
        color: #EDE9F5;
        padding: 14px;
        min-height: 76px;
        font-weight: 650;
    }

    QPushButton#featureCard:hover {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(72, 54, 116, 205),
            stop: 0.52 rgba(39, 39, 67, 210),
            stop: 1 rgba(24, 29, 48, 215)
        );
        border-color: rgba(151, 115, 255, 105);
    }

    QFrame#composer {
        background: rgba(22, 23, 40, 190);
        border: 1px solid rgba(190, 181, 236, 50);
        border-radius: 16px;
    }

    QLabel#statusChip {
        background: rgba(35, 36, 58, 165);
        color: #C5BED5;
        border: 1px solid rgba(176, 167, 220, 36);
        border-radius: 10px;
        padding: 6px 10px;
    }

    QLabel#warning {
        background: rgba(83, 56, 18, 150);
        color: #FFE2A9;
        border: 1px solid rgba(212, 157, 61, 105);
        border-radius: 10px;
        padding: 8px 10px;
    }

    QLabel#danger {
        color: #FFB9C4;
    }

    QLineEdit,
    QPlainTextEdit,
    QTextBrowser,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QListWidget,
    QTreeWidget {
        background: rgba(10, 12, 23, 188);
        color: #ECE8F5;
        border: 1px solid rgba(161, 151, 205, 48);
        border-radius: 10px;
        padding: 7px;
        selection-background-color: rgba(111, 78, 202, 180);
    }

    QLineEdit:focus,
    QPlainTextEdit:focus,
    QTextBrowser:focus,
    QComboBox:focus,
    QSpinBox:focus,
    QDoubleSpinBox:focus {
        border: 1px solid rgba(129, 101, 255, 135);
        background: rgba(14, 16, 29, 210);
    }

    QLineEdit#heroInput {
        background: rgba(10, 12, 24, 188);
        border: 1px solid rgba(172, 162, 219, 55);
        border-radius: 12px;
        padding: 10px 12px;
        font-size: 10pt;
    }

    QHeaderView::section {
        background: rgba(35, 34, 56, 210);
        color: #C9C2D8;
        padding: 7px;
        border: 0;
        border-bottom: 1px solid rgba(168, 156, 214, 34);
    }

    QTableWidget, QListWidget, QTreeWidget {
        gridline-color: rgba(96, 90, 126, 40);
    }

    QProgressBar {
        background: rgba(11, 13, 24, 190);
        border: 1px solid rgba(142, 131, 185, 48);
        border-radius: 7px;
        text-align: center;
    }

    QProgressBar::chunk {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 #4F9DFF,
            stop: 0.55 #7C5CFF,
            stop: 1 #D946EF
        );
        border-radius: 6px;
    }

    QScrollBar:vertical {
        background: transparent;
        width: 10px;
        margin: 2px;
    }

    QScrollBar::handle:vertical {
        background: rgba(119, 105, 155, 95);
        border-radius: 5px;
        min-height: 28px;
    }

    QScrollBar::handle:vertical:hover {
        background: rgba(151, 127, 211, 130);
    }

    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {
        height: 0;
    }

    /* Companion mode is intentionally preserved as a separate cinematic scene. */
    QWidget#companionMode {
        background: #02050b;
        border: 0;
    }

    QLabel#companionBrand {
        color: #e9fbff;
        font-size: 10pt;
        font-weight: 650;
        letter-spacing: 4px;
    }

    QLabel#companionState {
        color: #f4f7fb;
        font-size: 16pt;
        font-weight: 650;
    }

    QLabel#companionMessage {
        color: #9cb0c1;
        font-size: 10pt;
    }

    QLabel#companionTelemetry {
        background: rgba(121, 223, 255, 16);
        color: #aeeeff;
        border: 1px solid rgba(121, 223, 255, 38);
        border-radius: 9px;
        font-size: 8pt;
        letter-spacing: 1px;
        padding: 5px 10px;
    }

    QPushButton#companionStop {
        background: rgba(66, 18, 28, 150);
        color: #ffcbd3;
        border: 1px solid rgba(199, 70, 90, 135);
        border-radius: 9px;
        padding: 6px 13px;
        font-weight: 700;
    }

    QPushButton#companionStop:hover {
        background: rgba(111, 27, 42, 190);
    }

    QPushButton#companionGesture {
        background: rgba(28, 72, 91, 92);
        color: #cff7ff;
        border: 1px solid rgba(121, 223, 255, 58);
        border-radius: 9px;
        padding: 6px 11px;
        font-size: 8pt;
        letter-spacing: 1px;
    }

    QPushButton#companionGesture:checked {
        background: rgba(42, 126, 151, 130);
        border-color: rgba(180, 244, 255, 125);
    }

    QPushButton#companionGesture:disabled {
        color: rgba(171, 194, 203, 90);
        border-color: rgba(121, 223, 255, 20);
    }

    QFrame#companionConfirm {
        background: rgba(7, 27, 44, 232);
        border: 1px solid rgba(255, 180, 74, 120);
        border-radius: 12px;
    }
    """

    UI5D_STYLE = """
    /* UI-5D — final Workspace polish.
       Reference direction: one coherent obsidian-glass product surface with
       restrained violet / magenta / electric-blue illumination. */

    QWidget#workspace {
        background: qradialgradient(
            cx: 0.08, cy: 0.20, radius: 0.70,
            fx: 0.08, fy: 0.20,
            stop: 0 rgba(98, 49, 181, 62),
            stop: 0.34 rgba(19, 18, 40, 40),
            stop: 0.72 rgba(7, 9, 18, 18),
            stop: 1 rgba(4, 6, 13, 0)
        );
    }

    QFrame#workspaceShell {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(25, 24, 43, 232),
            stop: 0.46 rgba(13, 16, 31, 238),
            stop: 1 rgba(9, 12, 24, 244)
        );
        border: 1px solid rgba(221, 211, 255, 82);
        border-radius: 28px;
    }

    QFrame#sidebar {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(31, 29, 49, 208),
            stop: 0.58 rgba(18, 20, 36, 220),
            stop: 1 rgba(14, 17, 31, 226)
        );
        border: 1px solid rgba(212, 202, 255, 40);
        border-radius: 20px;
    }

    QLabel#brandMark {
        min-width: 34px;
        max-width: 34px;
        min-height: 34px;
        max-height: 34px;
        color: #F9F6FF;
        background: qradialgradient(
            cx: 0.50, cy: 0.50, radius: 0.72,
            stop: 0 rgba(79, 157, 255, 220),
            stop: 0.45 rgba(124, 92, 255, 215),
            stop: 0.78 rgba(217, 70, 239, 170),
            stop: 1 rgba(217, 70, 239, 18)
        );
        border: 1px solid rgba(231, 222, 255, 155);
        border-radius: 17px;
        font-size: 12pt;
        font-weight: 800;
    }

    QLabel#brand {
        color: #FCFAFF;
        font-size: 17pt;
        font-weight: 800;
        letter-spacing: 3px;
    }

    QLabel#brandSub {
        color: #858099;
        font-size: 8pt;
        letter-spacing: 1px;
    }

    QFrame#sidebarCore {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(80, 53, 139, 88),
            stop: 0.55 rgba(31, 34, 57, 160),
            stop: 1 rgba(18, 24, 43, 175)
        );
        border: 1px solid rgba(160, 129, 255, 55);
        border-radius: 14px;
    }

    QLabel#sidebarCoreTitle {
        color: #BCA8FF;
        font-size: 7.5pt;
        font-weight: 800;
        letter-spacing: 2px;
    }

    QLabel#sidebarCoreText {
        color: #BCB7CB;
        font-size: 8pt;
    }

    QPushButton#nav {
        min-height: 38px;
        padding: 7px 11px;
        border-radius: 11px;
        color: #A7A1B8;
        border: 1px solid transparent;
        background: transparent;
        font-weight: 550;
    }

    QPushButton#nav:hover {
        color: #F1EDFA;
        background: rgba(117, 85, 190, 36);
        border-color: rgba(165, 128, 255, 38);
    }

    QPushButton#nav:checked {
        color: #FFFFFF;
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 rgba(119, 77, 214, 120),
            stop: 0.54 rgba(83, 74, 179, 74),
            stop: 1 rgba(59, 105, 184, 38)
        );
        border: 1px solid rgba(181, 137, 255, 100);
    }

    QFrame#header {
        background: rgba(27, 27, 47, 155);
        border: 1px solid rgba(214, 204, 255, 36);
        border-radius: 17px;
    }

    QLabel#statusChip {
        background: rgba(34, 34, 56, 138);
        color: #BAB4CB;
        border: 1px solid rgba(180, 168, 220, 34);
        border-radius: 10px;
        padding: 6px 10px;
        font-size: 8.5pt;
    }

    QFrame#hero {
        background: qradialgradient(
            cx: 0.61, cy: 0.42, radius: 0.92,
            fx: 0.61, fy: 0.42,
            stop: 0 rgba(86, 84, 187, 72),
            stop: 0.20 rgba(116, 57, 173, 50),
            stop: 0.46 rgba(46, 34, 88, 57),
            stop: 0.76 rgba(20, 21, 42, 215),
            stop: 1 rgba(13, 15, 29, 236)
        );
        border: 1px solid rgba(225, 215, 255, 72);
        border-radius: 24px;
    }

    QLabel#eyebrow {
        color: #B598FF;
        font-size: 7.5pt;
        font-weight: 800;
        letter-spacing: 3px;
    }

    QLabel#heroTitle {
        color: #FDFCFF;
        font-size: 32pt;
        font-weight: 850;
        letter-spacing: 2px;
    }

    QLabel#heroAccent {
        color: #68C8FF;
        font-size: 32pt;
        font-weight: 850;
        letter-spacing: 1px;
    }

    QLabel#heroSubtitle {
        color: #F6F2FF;
        font-size: 15pt;
        font-weight: 700;
    }

    QLabel#heroDescription {
        color: #A6A0B6;
        font-size: 9.5pt;
    }

    QLabel#heroState {
        color: #C1ABFF;
        font-size: 9pt;
        font-weight: 800;
        letter-spacing: 2px;
    }

    QLabel#heroMessage {
        color: #A7A2B5;
        font-size: 9pt;
    }

    QFrame#orbPresentation {
        background: qradialgradient(
            cx: 0.50, cy: 0.49, radius: 0.78,
            fx: 0.50, fy: 0.49,
            stop: 0 rgba(92, 147, 255, 44),
            stop: 0.32 rgba(112, 83, 227, 34),
            stop: 0.57 rgba(216, 74, 239, 13),
            stop: 0.78 rgba(18, 20, 39, 12),
            stop: 1 rgba(0, 0, 0, 0)
        );
        border: 0;
        border-radius: 22px;
    }

    QFrame#voicePanel {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(45, 40, 68, 210),
            stop: 0.48 rgba(31, 30, 53, 218),
            stop: 1 rgba(20, 22, 39, 226)
        );
        border: 1px solid rgba(218, 206, 255, 72);
        border-radius: 18px;
    }

    QLabel#panelKicker {
        color: #DA5CFF;
        font-size: 7.5pt;
        font-weight: 800;
        letter-spacing: 2px;
    }

    QLabel#panelTitle {
        color: #F6F1FF;
        font-size: 10pt;
        font-weight: 700;
    }

    QLabel#panelState {
        color: #B69CFF;
        font-size: 14pt;
        font-weight: 800;
    }

    QLabel#panelMessage {
        color: #9892A9;
        font-size: 8.5pt;
    }

    QFrame#voiceMeterShell {
        background: rgba(10, 12, 24, 128);
        border: 1px solid rgba(179, 158, 229, 35);
        border-radius: 12px;
    }

    QFrame#statTile {
        background: rgba(13, 15, 29, 150);
        border: 1px solid rgba(176, 160, 220, 31);
        border-radius: 11px;
    }

    QLabel#statTitle {
        color: #79738B;
        font-size: 7pt;
        font-weight: 800;
        letter-spacing: 1px;
    }

    QLabel#statValue {
        color: #F4F0FA;
        font-size: 9.5pt;
        font-weight: 700;
    }

    QLabel#miniChip {
        background: rgba(31, 32, 53, 145);
        color: #BFB9CD;
        border: 1px solid rgba(170, 157, 214, 31);
        border-radius: 9px;
        padding: 6px 9px;
        font-size: 7.5pt;
    }

    QFrame#capabilityShelf {
        background: rgba(22, 23, 40, 174);
        border: 1px solid rgba(199, 187, 237, 40);
        border-radius: 18px;
    }

    QLabel#sectionKicker {
        color: #B99CFF;
        font-size: 8pt;
        font-weight: 800;
        letter-spacing: 2px;
    }

    QLabel#sectionMeta {
        color: #777286;
        font-size: 7.5pt;
        letter-spacing: 1px;
    }

    QPushButton#featureCard {
        min-height: 82px;
        text-align: left;
        padding: 13px 14px;
        color: #EDE9F5;
        font-weight: 650;
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(43, 40, 68, 185),
            stop: 0.54 rgba(27, 29, 48, 190),
            stop: 1 rgba(18, 21, 37, 205)
        );
        border: 1px solid rgba(194, 179, 232, 42);
        border-radius: 14px;
    }

    QPushButton#featureCard:hover {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(81, 54, 129, 205),
            stop: 0.50 rgba(43, 39, 73, 211),
            stop: 1 rgba(23, 30, 52, 216)
        );
        border-color: rgba(170, 122, 255, 105);
    }

    QFrame#composer {
        background: rgba(19, 20, 35, 190);
        border: 1px solid rgba(205, 190, 239, 45);
        border-radius: 15px;
    }

    QLineEdit#heroInput {
        background: rgba(7, 9, 18, 135);
        color: #EDE9F5;
        border: 1px solid rgba(150, 140, 190, 38);
        border-radius: 11px;
        padding: 10px 12px;
    }

    QLineEdit#heroInput:focus {
        background: rgba(10, 12, 24, 180);
        border-color: rgba(132, 101, 255, 125);
    }

    QPushButton#primary {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 #5AA7FF,
            stop: 0.50 #7B5BFF,
            stop: 1 #D94AEF
        );
        color: #FFFFFF;
        border: 1px solid rgba(231, 219, 255, 120);
        border-radius: 11px;
        font-weight: 800;
    }

    QPushButton#secondaryAction {
        background: rgba(34, 34, 56, 150);
        color: #E4DFEE;
        border: 1px solid rgba(185, 174, 224, 50);
        border-radius: 11px;
        font-weight: 650;
    }

    QPushButton#dangerButton {
        background: rgba(116, 27, 46, 208);
        color: #FFE8EC;
        border: 1px solid rgba(229, 91, 116, 145);
        border-radius: 11px;
        font-weight: 800;
    }
    """

    UI5E_STYLE = """
    /* UI-5E — final visual refinement.
       No controller/runtime changes. This only tightens hierarchy, spacing,
       shell illumination, cards and truthful live-voice presentation. */

    QFrame#workspaceShell {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(30, 27, 48, 238),
            stop: 0.38 rgba(15, 17, 33, 244),
            stop: 0.78 rgba(10, 13, 26, 248),
            stop: 1 rgba(8, 10, 21, 250)
        );
        border: 1px solid rgba(226, 216, 255, 112);
        border-radius: 30px;
    }

    QFrame#sidebar {
        border-color: rgba(216, 204, 255, 55);
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(33, 30, 52, 214),
            stop: 0.55 rgba(18, 20, 37, 224),
            stop: 1 rgba(13, 16, 30, 232)
        );
    }

    QFrame#header {
        min-height: 50px;
        background: rgba(25, 25, 44, 132);
        border-color: rgba(217, 206, 255, 42);
    }

    QLabel#state {
        color: #8FF1D2;
        font-size: 10.5pt;
        font-weight: 800;
        letter-spacing: 1px;
    }

    QFrame#hero {
        border-color: rgba(229, 218, 255, 88);
        background: qradialgradient(
            cx: 0.61, cy: 0.44, radius: 0.96,
            fx: 0.61, fy: 0.44,
            stop: 0 rgba(97, 109, 220, 82),
            stop: 0.17 rgba(120, 67, 190, 62),
            stop: 0.40 rgba(56, 36, 102, 64),
            stop: 0.72 rgba(21, 22, 44, 218),
            stop: 1 rgba(12, 15, 29, 240)
        );
    }

    QLabel#heroTitle {
        font-size: 30pt;
        letter-spacing: 2px;
    }

    QLabel#heroAccent {
        font-size: 30pt;
        color: #6FCBFF;
    }

    QLabel#heroSubtitle {
        font-size: 13.5pt;
        font-weight: 700;
    }

    QLabel#heroDescription {
        color: #9E98AE;
        font-size: 9pt;
    }

    QLabel#heroState {
        margin-top: 3px;
    }

    QFrame#orbPresentation {
        background: qradialgradient(
            cx: 0.50, cy: 0.48, radius: 0.82,
            fx: 0.50, fy: 0.48,
            stop: 0 rgba(107, 174, 255, 62),
            stop: 0.27 rgba(114, 93, 255, 43),
            stop: 0.53 rgba(213, 73, 240, 17),
            stop: 0.78 rgba(20, 21, 42, 8),
            stop: 1 rgba(0, 0, 0, 0)
        );
    }

    QFrame#voicePanel {
        border-color: rgba(227, 214, 255, 86);
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(49, 42, 74, 216),
            stop: 0.46 rgba(31, 30, 54, 224),
            stop: 1 rgba(20, 22, 40, 230)
        );
    }

    QFrame#voiceMeterShell {
        min-height: 62px;
        background: rgba(8, 10, 20, 145);
        border-color: rgba(187, 160, 235, 48);
    }

    QFrame#statTile {
        min-height: 51px;
        background: rgba(12, 14, 27, 165);
        border-color: rgba(184, 165, 226, 38);
    }

    QLabel#statTitle {
        color: #8F87A3;
        font-size: 7pt;
    }

    QLabel#statValue {
        color: #F7F3FC;
        font-size: 9.5pt;
    }

    QFrame#capabilityShelf {
        border-color: rgba(213, 199, 246, 52);
        background: rgba(20, 21, 38, 184);
    }

    QPushButton#featureCard {
        min-height: 88px;
        padding: 14px 15px;
        border-color: rgba(203, 186, 238, 48);
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(49, 43, 76, 196),
            stop: 0.50 rgba(28, 30, 51, 199),
            stop: 1 rgba(18, 22, 39, 212)
        );
    }

    QPushButton#featureCard:hover {
        border-color: rgba(184, 137, 255, 125);
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(91, 58, 143, 214),
            stop: 0.48 rgba(47, 42, 81, 214),
            stop: 1 rgba(23, 31, 55, 220)
        );
    }

    QFrame#composer {
        border-color: rgba(216, 199, 247, 55);
        background: rgba(16, 18, 32, 202);
    }

    QPushButton#primary {
        min-height: 38px;
    }

    QPushButton#secondaryAction {
        min-height: 38px;
    }

    /* UI-5F — cohesive responsive product surface. */
    QWidget {
        font-family: "Segoe UI Variable Text", "Segoe UI", Arial;
    }

    QMainWindow, QDialog {
        background: #040611;
    }

    QWidget#workspace {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 #080A18,
            stop: 0.46 #050713,
            stop: 1 #03050D
        );
    }

    QScrollArea#homeScroll,
    QScrollArea#homeScroll > QWidget > QWidget,
    QWidget#homeCanvas {
        background: transparent;
        border: 0;
    }

    QFrame#workspaceShell {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(31, 28, 50, 235),
            stop: 0.20 rgba(17, 19, 37, 239),
            stop: 0.67 rgba(10, 13, 27, 244),
            stop: 1 rgba(7, 10, 20, 247)
        );
        border: 1px solid rgba(235, 228, 255, 132);
        border-radius: 31px;
    }

    QFrame#sidebar {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(37, 33, 58, 207),
            stop: 0.48 rgba(19, 22, 41, 218),
            stop: 1 rgba(11, 15, 29, 228)
        );
        border: 1px solid rgba(215, 205, 255, 46);
        border-radius: 21px;
    }

    QLabel#brand {
        color: #FBF9FF;
        font-size: 17pt;
        font-weight: 800;
        letter-spacing: 3px;
    }

    QLabel#brandSub {
        color: #827C92;
        font-size: 7pt;
        font-weight: 650;
        letter-spacing: 1px;
    }

    QPushButton#nav {
        min-height: 40px;
        text-align: left;
        padding: 6px 9px 6px 42px;
        border-radius: 12px;
        color: #AAA4B7;
        font-size: 9pt;
        font-weight: 560;
    }

    QPushButton#nav:hover {
        color: #F5F1FF;
        background: rgba(111, 83, 184, 42);
        border-color: rgba(170, 130, 255, 46);
    }

    QPushButton#nav:checked {
        color: #FFFFFF;
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 rgba(126, 76, 222, 130),
            stop: 0.52 rgba(89, 74, 187, 78),
            stop: 1 rgba(65, 121, 199, 42)
        );
        border: 1px solid rgba(186, 143, 255, 108);
    }

    QFrame#sidebarCore {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(82, 53, 141, 74),
            stop: 0.56 rgba(30, 34, 58, 138),
            stop: 1 rgba(16, 23, 41, 160)
        );
        border: 1px solid rgba(159, 127, 255, 48);
        border-radius: 14px;
    }

    QFrame#header {
        min-height: 52px;
        max-height: 58px;
        background: rgba(25, 25, 44, 118);
        border: 1px solid rgba(217, 206, 255, 36);
        border-radius: 17px;
    }

    QFrame#stateDot {
        background: #86EBCB;
        border: 1px solid rgba(210, 255, 242, 160);
        border-radius: 4px;
    }

    QLabel#state {
        color: #8FF1D2;
        font-size: 9.5pt;
        font-weight: 800;
        letter-spacing: 1px;
    }

    QLabel#muted {
        color: #898397;
        font-size: 8.5pt;
    }

    QFrame#trustCluster {
        background: rgba(9, 11, 23, 112);
        border: 1px solid rgba(182, 169, 224, 28);
        border-radius: 12px;
    }

    QLabel#statusChip {
        background: transparent;
        color: #B9B3C7;
        border: 0;
        border-radius: 8px;
        padding: 5px 7px 5px 21px;
        font-size: 7.5pt;
        font-weight: 650;
    }

    QPushButton#reauthButton {
        min-height: 34px;
        background: rgba(82, 61, 132, 86);
        color: #E9E2F7;
        border: 1px solid rgba(179, 143, 255, 75);
        border-radius: 10px;
        padding: 6px 10px;
        font-size: 8pt;
    }

    QPushButton#dangerButton {
        min-height: 36px;
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(126, 29, 49, 225),
            stop: 1 rgba(87, 21, 40, 235)
        );
        color: #FFE9EE;
        border: 1px solid rgba(246, 102, 129, 155);
        border-radius: 11px;
        padding: 7px 12px;
        font-size: 8pt;
        font-weight: 800;
        letter-spacing: 1px;
    }

    QFrame#hero {
        background: qradialgradient(
            cx: 0.60, cy: 0.46, radius: 0.95,
            fx: 0.60, fy: 0.46,
            stop: 0 rgba(91, 113, 224, 88),
            stop: 0.16 rgba(116, 73, 199, 67),
            stop: 0.37 rgba(63, 43, 116, 69),
            stop: 0.69 rgba(22, 23, 45, 220),
            stop: 1 rgba(11, 14, 28, 241)
        );
        border: 1px solid rgba(229, 219, 255, 70);
        border-radius: 25px;
    }

    QLabel#eyebrow {
        color: #B99DFF;
        font-size: 7pt;
        font-weight: 800;
        letter-spacing: 2px;
    }

    QLabel#heroSubtitle {
        color: #F7F3FF;
        font-size: 14.5pt;
        font-weight: 720;
    }

    QLabel#heroDescription {
        color: #ACA6B9;
        font-size: 9.2pt;
    }

    QLabel#heroState {
        color: #C5B2FF;
        font-size: 9pt;
        font-weight: 800;
        letter-spacing: 2px;
        margin-top: 4px;
    }

    QLabel#heroMessage {
        color: #A9A3B5;
        font-size: 8.5pt;
    }

    QLabel#miniChip {
        background: rgba(20, 23, 42, 145);
        color: #BDB7CB;
        border: 1px solid rgba(176, 159, 220, 32);
        border-radius: 9px;
        padding: 5px 8px;
        font-size: 7pt;
    }

    QFrame#orbPresentation {
        background: qradialgradient(
            cx: 0.50, cy: 0.47, radius: 0.76,
            fx: 0.50, fy: 0.47,
            stop: 0 rgba(112, 188, 255, 78),
            stop: 0.20 rgba(99, 116, 255, 54),
            stop: 0.45 rgba(152, 79, 229, 29),
            stop: 0.72 rgba(213, 72, 239, 8),
            stop: 1 rgba(0, 0, 0, 0)
        );
        border: 0;
        border-radius: 24px;
    }

    QFrame#voicePanel {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 rgba(54, 45, 81, 222),
            stop: 0.42 rgba(34, 32, 59, 228),
            stop: 1 rgba(18, 22, 42, 238)
        );
        border: 1px solid rgba(229, 216, 255, 94);
        border-radius: 19px;
    }

    QLabel#panelKicker {
        color: #DD69F5;
        font-size: 7pt;
        font-weight: 800;
        letter-spacing: 2px;
    }

    QLabel#panelBadge {
        color: #857E93;
        background: rgba(12, 14, 27, 108);
        border: 1px solid rgba(179, 161, 220, 28);
        border-radius: 7px;
        padding: 3px 6px;
        font-size: 6.5pt;
        font-weight: 700;
        letter-spacing: 1px;
    }

    QLabel#panelTitle {
        color: #F7F3FF;
        font-size: 10.5pt;
        font-weight: 700;
    }

    QLabel#panelState {
        color: #BEA7FF;
        font-size: 14pt;
        font-weight: 800;
    }

    QLabel#panelMessage {
        color: #9D97A9;
        font-size: 8pt;
    }

    QLabel#meterLabel {
        color: #787286;
        font-size: 6.5pt;
        font-weight: 800;
        letter-spacing: 1px;
    }

    QFrame#voiceMeterShell {
        min-height: 52px;
        background: rgba(6, 8, 18, 158);
        border: 1px solid rgba(186, 164, 232, 42);
        border-radius: 12px;
    }

    QFrame#statTile {
        min-height: 44px;
        background: rgba(10, 13, 26, 170);
        border: 1px solid rgba(183, 165, 224, 34);
        border-radius: 11px;
    }

    QLabel#statTitle {
        color: #827B91;
        font-size: 6.5pt;
        font-weight: 800;
        letter-spacing: 1px;
    }

    QLabel#statValue {
        color: #F7F3FC;
        font-size: 9pt;
        font-weight: 680;
    }

    QLabel#quickActionTitle {
        color: #F4F0FB;
        font-size: 7.5pt;
        font-weight: 800;
        letter-spacing: 1px;
    }

    QLabel#quickActionHelp {
        color: #898395;
        font-size: 7pt;
    }

    QPushButton#micAction {
        background: transparent;
        border: 0;
        padding: 0;
    }

    QFrame#capabilityShelf {
        background: rgba(16, 19, 35, 158);
        border: 1px solid rgba(206, 193, 244, 38);
        border-radius: 18px;
    }

    QLabel#sectionKicker {
        color: #BCA2FF;
        font-size: 7.5pt;
        font-weight: 800;
        letter-spacing: 2px;
    }

    QLabel#sectionMeta {
        color: #716B7E;
        font-size: 6.5pt;
        font-weight: 650;
        letter-spacing: 1px;
    }

    QPushButton#capabilityCard {
        background: transparent;
        border: 0;
        padding: 0;
    }

    QFrame#composer {
        min-height: 58px;
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 rgba(19, 21, 39, 214),
            stop: 0.66 rgba(12, 15, 29, 224),
            stop: 1 rgba(22, 18, 42, 216)
        );
        border: 1px solid rgba(215, 200, 249, 52);
        border-radius: 17px;
    }

    QLineEdit#heroInput,
    QLineEdit#heroInput:focus {
        min-height: 34px;
        background: transparent;
        color: #F2EEF8;
        border: 0;
        border-radius: 0;
        padding: 5px 4px;
        font-size: 9.5pt;
    }

    QLabel#composerHint {
        color: #696477;
        font-size: 6.5pt;
        font-weight: 700;
        letter-spacing: 1px;
        padding-right: 4px;
    }

    QPushButton#sendButton {
        min-width: 68px;
        min-height: 40px;
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 #58AFFF,
            stop: 0.50 #7B5BFF,
            stop: 1 #D94AEF
        );
        color: #FFFFFF;
        border: 1px solid rgba(238, 226, 255, 130);
        border-radius: 12px;
        padding: 7px 13px;
        font-weight: 800;
    }

    QPushButton#sendButton:hover {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 #72C4FF,
            stop: 0.50 #9271FF,
            stop: 1 #E762F4
        );
        border-color: rgba(248, 239, 255, 190);
    }

    QPushButton#primary {
        min-height: 41px;
        border-radius: 12px;
        padding: 8px 15px;
    }

    QPushButton#secondaryAction {
        min-height: 41px;
        border-radius: 12px;
        padding: 8px 14px;
    }

    QToolTip {
        color: #F6F1FF;
        background: #17172A;
        border: 1px solid rgba(184, 159, 238, 90);
        padding: 5px 7px;
    }
    """

    NAV_ITEMS = (
        ("Home", "home"),
        ("Conversation", "conversation"),
        ("Confirmation", "confirmation"),
        ("Memory", "memory"),
        ("Users & VoiceGuard", "voiceguard"),
        ("Permissions", "permissions"),
        ("Activity log", "activity"),
        ("Settings", "settings"),
        ("Diagnostics", "diagnostics"),
    )


    def _card() -> QFrame:
        frame = QFrame()
        frame.setObjectName("card")
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        return frame


    def _label(text: str, object_name: str = "") -> QLabel:
        label = QLabel(text)
        if object_name:
            label.setObjectName(object_name)
        label.setWordWrap(True)
        return label


    def _draw_line_icon(
        painter: QPainter,
        icon_key: str,
        rect: QRectF,
        color: QColor,
        width: float = 1.55,
    ) -> None:
        """Draw small font-independent interface icons with restrained linework."""

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        def point(x_value: float, y_value: float) -> QPointF:
            return QPointF(
                rect.left() + rect.width() * x_value,
                rect.top() + rect.height() * y_value,
            )

        def box(
            left: float,
            top: float,
            box_width: float,
            box_height: float,
        ) -> QRectF:
            return QRectF(
                rect.left() + rect.width() * left,
                rect.top() + rect.height() * top,
                rect.width() * box_width,
                rect.height() * box_height,
            )

        path = QPainterPath()
        if icon_key == "home":
            path.moveTo(point(0.12, 0.48))
            path.lineTo(point(0.50, 0.16))
            path.lineTo(point(0.88, 0.48))
            path.moveTo(point(0.22, 0.43))
            path.lineTo(point(0.22, 0.84))
            path.lineTo(point(0.78, 0.84))
            path.lineTo(point(0.78, 0.43))
            path.moveTo(point(0.43, 0.84))
            path.lineTo(point(0.43, 0.62))
            path.lineTo(point(0.57, 0.62))
            path.lineTo(point(0.57, 0.84))
            painter.drawPath(path)
        elif icon_key == "conversation":
            painter.drawRoundedRect(box(0.10, 0.16, 0.80, 0.58), 3.0, 3.0)
            path.moveTo(point(0.30, 0.74))
            path.lineTo(point(0.22, 0.90))
            path.lineTo(point(0.50, 0.74))
            painter.drawPath(path)
            painter.drawLine(point(0.28, 0.39), point(0.72, 0.39))
            painter.drawLine(point(0.28, 0.53), point(0.59, 0.53))
        elif icon_key == "confirmation":
            path.moveTo(point(0.50, 0.10))
            path.lineTo(point(0.82, 0.24))
            path.lineTo(point(0.77, 0.64))
            path.cubicTo(point(0.73, 0.78), point(0.60, 0.87), point(0.50, 0.92))
            path.cubicTo(point(0.40, 0.87), point(0.27, 0.78), point(0.23, 0.64))
            path.lineTo(point(0.18, 0.24))
            path.closeSubpath()
            painter.drawPath(path)
            path = QPainterPath(point(0.34, 0.50))
            path.lineTo(point(0.46, 0.62))
            path.lineTo(point(0.68, 0.37))
            painter.drawPath(path)
        elif icon_key == "memory":
            painter.drawEllipse(box(0.17, 0.12, 0.66, 0.25))
            painter.drawArc(box(0.17, 0.30, 0.66, 0.25), 180 * 16, 180 * 16)
            painter.drawArc(box(0.17, 0.50, 0.66, 0.25), 180 * 16, 180 * 16)
            painter.drawArc(box(0.17, 0.67, 0.66, 0.20), 180 * 16, 180 * 16)
            painter.drawLine(point(0.17, 0.24), point(0.17, 0.77))
            painter.drawLine(point(0.83, 0.24), point(0.83, 0.77))
        elif icon_key == "voiceguard":
            painter.drawEllipse(box(0.36, 0.16, 0.28, 0.28))
            painter.drawArc(box(0.25, 0.40, 0.50, 0.44), 20 * 16, 140 * 16)
            painter.drawArc(box(0.10, 0.18, 0.28, 0.56), 270 * 16, 180 * 16)
            painter.drawArc(box(0.62, 0.18, 0.28, 0.56), 90 * 16, 180 * 16)
        elif icon_key == "permissions":
            painter.drawRoundedRect(box(0.17, 0.42, 0.66, 0.46), 3.0, 3.0)
            painter.drawArc(box(0.29, 0.10, 0.42, 0.56), 0, 180 * 16)
            painter.drawEllipse(box(0.44, 0.58, 0.12, 0.12))
            painter.drawLine(point(0.50, 0.70), point(0.50, 0.78))
        elif icon_key == "activity":
            path.moveTo(point(0.09, 0.58))
            path.lineTo(point(0.27, 0.58))
            path.lineTo(point(0.38, 0.30))
            path.lineTo(point(0.54, 0.76))
            path.lineTo(point(0.68, 0.43))
            path.lineTo(point(0.91, 0.43))
            painter.drawPath(path)
        elif icon_key == "settings":
            center = point(0.50, 0.50)
            painter.drawEllipse(center, rect.width() * 0.17, rect.height() * 0.17)
            painter.drawEllipse(center, rect.width() * 0.34, rect.height() * 0.34)
            for index in range(8):
                angle = index * math.tau / 8.0
                painter.drawLine(
                    point(0.50 + math.cos(angle) * 0.34, 0.50 + math.sin(angle) * 0.34),
                    point(0.50 + math.cos(angle) * 0.45, 0.50 + math.sin(angle) * 0.45),
                )
        elif icon_key == "diagnostics":
            painter.drawEllipse(box(0.10, 0.10, 0.80, 0.80))
            path.moveTo(point(0.23, 0.53))
            path.lineTo(point(0.38, 0.53))
            path.lineTo(point(0.46, 0.34))
            path.lineTo(point(0.57, 0.68))
            path.lineTo(point(0.66, 0.47))
            path.lineTo(point(0.79, 0.47))
            painter.drawPath(path)
        elif icon_key == "microphone":
            painter.drawRoundedRect(box(0.35, 0.10, 0.30, 0.51), 5.0, 5.0)
            painter.drawArc(box(0.22, 0.31, 0.56, 0.48), 180 * 16, 180 * 16)
            painter.drawLine(point(0.50, 0.72), point(0.50, 0.88))
            painter.drawLine(point(0.35, 0.88), point(0.65, 0.88))
        elif icon_key == "spark":
            path.moveTo(point(0.50, 0.08))
            path.cubicTo(point(0.48, 0.35), point(0.39, 0.47), point(0.10, 0.50))
            path.cubicTo(point(0.39, 0.53), point(0.48, 0.65), point(0.50, 0.92))
            path.cubicTo(point(0.52, 0.65), point(0.61, 0.53), point(0.90, 0.50))
            path.cubicTo(point(0.61, 0.47), point(0.52, 0.35), point(0.50, 0.08))
            painter.drawPath(path)
        else:
            painter.drawEllipse(box(0.16, 0.16, 0.68, 0.68))
        painter.restore()


    class AmbientWorkspace(QWidget):
        """Presentation-only ambient light and wire curves behind the product shell."""

        def paintEvent(self, event: Any) -> None:  # noqa: N802
            super().paintEvent(event)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            width = float(self.width())
            height = float(self.height())

            for center, radius, color in (
                (QPointF(width * 0.02, height * 0.18), height * 0.42, QColor(111, 55, 232, 68)),
                (QPointF(width * 0.95, height * 0.77), height * 0.36, QColor(194, 50, 224, 40)),
                (QPointF(width * 0.58, height * 0.04), height * 0.26, QColor(66, 126, 255, 34)),
            ):
                glow = QRadialGradient(center, radius)
                glow.setColorAt(0.0, color)
                edge = QColor(color)
                edge.setAlpha(0)
                glow.setColorAt(1.0, edge)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(glow))
                painter.drawEllipse(center, radius, radius)

            painter.setBrush(Qt.BrushStyle.NoBrush)
            for side in (-1.0, 1.0):
                for index in range(5):
                    tone = QColor(128, 90, 255, 14 + index * 3)
                    painter.setPen(QPen(tone, 0.8))
                    anchor_x = width * (-0.06 if side < 0 else 1.06)
                    spread = width * (0.24 + index * 0.018)
                    path = QPainterPath(QPointF(anchor_x, height * (0.30 + index * 0.045)))
                    path.cubicTo(
                        QPointF(anchor_x - side * spread * 0.20, height * 0.40),
                        QPointF(anchor_x - side * spread * 0.88, height * 0.52),
                        QPointF(anchor_x - side * spread, height * (0.72 - index * 0.025)),
                    )
                    painter.drawPath(path)


    class BrandMark(QWidget):
        """ASHER's compact luminous ring mark, independent of platform glyph fonts."""

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setFixedSize(42, 42)
            self.setAccessibleName("ASHER brand mark")

        def paintEvent(self, _event: Any) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            center = QPointF(self.width() / 2.0, self.height() / 2.0)

            glow = QRadialGradient(center, 20.0)
            glow.setColorAt(0.35, QColor(113, 103, 255, 54))
            glow.setColorAt(0.70, QColor(200, 72, 241, 32))
            glow.setColorAt(1.0, QColor(200, 72, 241, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(center, 20.0, 20.0)

            inner = QRadialGradient(QPointF(17.0, 15.0), 16.0)
            inner.setColorAt(0.0, QColor(48, 63, 104, 245))
            inner.setColorAt(0.58, QColor(17, 20, 38, 250))
            inner.setColorAt(1.0, QColor(8, 10, 21, 255))
            painter.setBrush(QBrush(inner))
            painter.setPen(QPen(QColor(239, 233, 255, 155), 1.0))
            painter.drawEllipse(center, 14.5, 14.5)

            ring = QConicalGradient(center, -35.0)
            ring.setColorAt(0.00, QColor("#58B8FF"))
            ring.setColorAt(0.34, QColor("#7C5CFF"))
            ring.setColorAt(0.68, QColor("#D94AEF"))
            ring.setColorAt(1.00, QColor("#58B8FF"))
            ring_pen = QPen(QBrush(ring), 3.2)
            ring_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(ring_pen)
            painter.drawEllipse(center, 12.0, 12.0)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(232, 249, 255, 225))
            painter.drawEllipse(QPointF(17.4, 15.8), 1.7, 1.7)


    class GradientTitle(QWidget):
        """Large ASHER identity with a controlled reference-inspired color sweep."""

        def __init__(self, text: str, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._text = text
            self.setMinimumHeight(66)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.setAccessibleName(text)

        def sizeHint(self) -> QSize:  # noqa: N802
            return QSize(370, 68)

        def paintEvent(self, _event: Any) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            font = QFont("Segoe UI Variable Display")
            font.setWeight(QFont.Weight.Black)
            point_size = 38.0
            font.setPointSizeF(point_size)
            metrics = QFontMetrics(font)
            available = max(1, self.width() - 4)
            advance = max(1, metrics.horizontalAdvance(self._text))
            if advance > available:
                point_size *= available / advance
                font.setPointSizeF(max(26.0, point_size))
                metrics = QFontMetrics(font)

            gradient = QLinearGradient(0.0, 0.0, float(self.width()), 0.0)
            gradient.setColorAt(0.00, QColor("#E45AF1"))
            gradient.setColorAt(0.34, QColor("#A06CFF"))
            gradient.setColorAt(0.67, QColor("#708FFF"))
            gradient.setColorAt(1.00, QColor("#62C9FF"))
            painter.setFont(font)
            painter.setPen(QPen(QBrush(gradient), 1.0))
            baseline = (self.height() + metrics.ascent() - metrics.descent()) / 2.0
            painter.drawText(QPointF(1.0, baseline), self._text)


    class StatusChipLabel(QLabel):
        """Truthful status pill with a painted tone indicator instead of an emoji."""

        def __init__(
            self,
            text: str,
            tone: str,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(text, parent)
            self._tone = QColor(tone)
            self.setObjectName("statusChip")
            self.setWordWrap(False)

        def set_tone(self, tone: str) -> None:
            self._tone = QColor(tone)
            self.update()

        def paintEvent(self, event: Any) -> None:  # noqa: N802
            super().paintEvent(event)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            center = QPointF(11.0, self.height() / 2.0)
            glow = QColor(self._tone)
            glow.setAlpha(40)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(center, 5.0, 5.0)
            painter.setBrush(self._tone)
            painter.drawEllipse(center, 2.25, 2.25)


    class VectorNavButton(QPushButton):
        """Navigation control with crisp vector line icons at every DPI."""

        def __init__(
            self,
            label: str,
            icon_key: str,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(label, parent)
            self._icon_key = icon_key
            self.setObjectName("nav")
            self.setCheckable(True)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setAccessibleName(f"{label} navigation")

        def paintEvent(self, event: Any) -> None:  # noqa: N802
            super().paintEvent(event)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            if self.isChecked():
                accent = QLinearGradient(0.0, 7.0, 0.0, float(self.height() - 7))
                accent.setColorAt(0.0, QColor("#D65CFF"))
                accent.setColorAt(0.52, QColor("#8B67FF"))
                accent.setColorAt(1.0, QColor("#58B8FF"))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(accent))
                painter.drawRoundedRect(QRectF(4.0, 9.0, 2.5, self.height() - 18.0), 1.25, 1.25)
            icon_color = QColor(
                "#FFFFFF"
                if self.isChecked()
                else "#CFC7E4"
                if self.underMouse()
                else "#8E899F"
            )
            _draw_line_icon(
                painter,
                self._icon_key,
                QRectF(15.0, self.height() / 2.0 - 8.5, 17.0, 17.0),
                icon_color,
                1.4,
            )


    class CapabilityCard(QPushButton):
        """Clickable real capability destination with painted glass depth."""

        def __init__(
            self,
            title: str,
            description: str,
            icon_key: str,
            accent: str,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(title, parent)
            self._title = title
            self._description = description
            self._icon_key = icon_key
            self._accent = QColor(accent)
            self.setObjectName("capabilityCard")
            self.setMinimumHeight(108)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setAccessibleName(title)
            self.setAccessibleDescription(description)

        def sizeHint(self) -> QSize:  # noqa: N802
            return QSize(235, 108)

        def enterEvent(self, event: Any) -> None:  # noqa: N802
            super().enterEvent(event)
            self.update()

        def leaveEvent(self, event: Any) -> None:  # noqa: N802
            super().leaveEvent(event)
            self.update()

        def paintEvent(self, _event: Any) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            shift = 1.0 if self.isDown() else 0.0
            rect = QRectF(self.rect()).adjusted(0.75, 0.75 + shift, -0.75, -0.75 + shift)
            hovered = self.underMouse()

            surface = QLinearGradient(rect.topLeft(), rect.bottomRight())
            tint = QColor(self._accent)
            tint.setAlpha(58 if hovered else 32)
            surface.setColorAt(0.0, QColor(48, 42, 73, 235))
            surface.setColorAt(0.33, tint)
            surface.setColorAt(0.62, QColor(24, 27, 47, 235))
            surface.setColorAt(1.0, QColor(14, 18, 34, 244))
            painter.setBrush(QBrush(surface))
            border = QColor(self._accent)
            border.setAlpha(142 if hovered or self.hasFocus() else 57)
            painter.setPen(QPen(border, 1.05))
            painter.drawRoundedRect(rect, 15.0, 15.0)

            highlight = QLinearGradient(rect.left() + 14.0, 0.0, rect.right() - 14.0, 0.0)
            start = QColor(self._accent)
            start.setAlpha(100 if hovered else 55)
            end = QColor(self._accent)
            end.setAlpha(0)
            highlight.setColorAt(0.0, start)
            highlight.setColorAt(1.0, end)
            painter.setPen(QPen(QBrush(highlight), 1.0))
            painter.drawLine(
                QPointF(rect.left() + 15.0, rect.top() + 1.0),
                QPointF(rect.right() - 15.0, rect.top() + 1.0),
            )

            icon_rect = QRectF(rect.left() + 15.0, rect.top() + 15.0, 38.0, 38.0)
            icon_glow = QRadialGradient(icon_rect.center(), 27.0)
            glow_color = QColor(self._accent)
            glow_color.setAlpha(110 if hovered else 78)
            icon_glow.setColorAt(0.0, glow_color)
            icon_edge = QColor(self._accent)
            icon_edge.setAlpha(16)
            icon_glow.setColorAt(1.0, icon_edge)
            painter.setPen(
                QPen(
                    QColor(
                        self._accent.red(),
                        self._accent.green(),
                        self._accent.blue(),
                        64,
                    ),
                    1.0,
                )
            )
            painter.setBrush(QBrush(icon_glow))
            painter.drawRoundedRect(icon_rect, 11.0, 11.0)
            _draw_line_icon(
                painter,
                self._icon_key,
                icon_rect.adjusted(9.0, 9.0, -9.0, -9.0),
                QColor("#F2EAFF"),
                1.55,
            )

            title_font = QFont("Segoe UI Variable Text", 10)
            title_font.setWeight(QFont.Weight.DemiBold)
            painter.setFont(title_font)
            painter.setPen(QColor("#F7F3FF"))
            title_rect = QRectF(
                icon_rect.right() + 12.0,
                rect.top() + 14.0,
                rect.width() - 93.0,
                40.0,
            )
            painter.drawText(
                title_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._title,
            )

            description_font = QFont("Segoe UI Variable Text", 8)
            painter.setFont(description_font)
            painter.setPen(QColor("#9C96AC"))
            description_rect = QRectF(
                rect.left() + 15.0,
                rect.top() + 66.0,
                rect.width() - 42.0,
                31.0,
            )
            painter.drawText(
                description_rect,
                Qt.AlignmentFlag.AlignLeft
                | Qt.AlignmentFlag.AlignTop
                | Qt.TextFlag.TextWordWrap,
                self._description,
            )

            arrow_color = QColor(self._accent)
            arrow_color.setAlpha(230 if hovered else 150)
            arrow_pen = QPen(arrow_color, 1.5)
            arrow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(arrow_pen)
            arrow_x = rect.right() - 18.0
            arrow_y = rect.bottom() - 17.0
            painter.drawLine(QPointF(arrow_x - 7.0, arrow_y), QPointF(arrow_x, arrow_y))
            painter.drawLine(QPointF(arrow_x - 3.5, arrow_y - 3.5), QPointF(arrow_x, arrow_y))
            painter.drawLine(QPointF(arrow_x - 3.5, arrow_y + 3.5), QPointF(arrow_x, arrow_y))


    class MicActionButton(QPushButton):
        """Reference-inspired microphone action without fabricated activity."""

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__("Start listening", parent)
            self.setObjectName("micAction")
            self.setFixedSize(60, 60)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setAccessibleName("Start listening")

        def setText(self, text: str) -> None:  # noqa: N802
            super().setText(text)
            self.setAccessibleName(text)
            self.update()

        def enterEvent(self, event: Any) -> None:  # noqa: N802
            super().enterEvent(event)
            self.update()

        def leaveEvent(self, event: Any) -> None:  # noqa: N802
            super().leaveEvent(event)
            self.update()

        def paintEvent(self, _event: Any) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            center = QPointF(self.width() / 2.0, self.height() / 2.0)
            active = self.text().casefold().startswith("stop")

            glow = QRadialGradient(center, 29.0)
            glow.setColorAt(
                0.35,
                QColor(167, 95, 255, 90 if self.underMouse() else 58),
            )
            glow.setColorAt(0.72, QColor(97, 111, 255, 46))
            glow.setColorAt(1.0, QColor(97, 111, 255, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(center, 29.0, 29.0)

            face = QLinearGradient(9.0, 8.0, 51.0, 52.0)
            face.setColorAt(0.0, QColor("#5AB7FF" if not active else "#F06C9B"))
            face.setColorAt(0.50, QColor("#7B5BFF" if not active else "#BE4EAF"))
            face.setColorAt(1.0, QColor("#D94AEF" if not active else "#8C3C99"))
            painter.setBrush(QBrush(face))
            painter.setPen(QPen(QColor(238, 228, 255, 175), 1.2))
            painter.drawEllipse(center, 22.5, 22.5)
            _draw_line_icon(
                painter,
                "microphone",
                QRectF(center.x() - 10.0, center.y() - 10.0, 20.0, 20.0),
                QColor("#FFFFFF"),
                1.6,
            )


    class ComposerGlyph(QWidget):
        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setFixedSize(30, 30)
            self.setAccessibleName("ASHER command input")

        def paintEvent(self, _event: Any) -> None:  # noqa: N802
            painter = QPainter(self)
            glow = QRadialGradient(QPointF(15.0, 15.0), 15.0)
            glow.setColorAt(0.0, QColor(133, 98, 255, 80))
            glow.setColorAt(1.0, QColor(133, 98, 255, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(QPointF(15.0, 15.0), 15.0, 15.0)
            _draw_line_icon(
                painter,
                "spark",
                QRectF(8.0, 8.0, 14.0, 14.0),
                QColor("#BDA8FF"),
                1.3,
            )


    class OrbStage(QFrame):
        """Atmospheric stage around the approved Home orb renderer."""

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setObjectName("orbPresentation")

        def paintEvent(self, event: Any) -> None:  # noqa: N802
            super().paintEvent(event)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            width = float(self.width())
            height = float(self.height())
            center_x = width / 2.0
            base_y = height * 0.86

            # Subtle stage glass reflection at base
            bloom = QRadialGradient(QPointF(center_x, base_y), width * 0.45)
            bloom.setColorAt(0.0, QColor(0, 229, 255, 22))
            bloom.setColorAt(0.40, QColor(120, 82, 255, 14))
            bloom.setColorAt(0.80, QColor(218, 73, 239, 4))
            bloom.setColorAt(1.0, QColor(0, 0, 0, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(bloom))
            painter.drawEllipse(
                QRectF(
                    center_x - width * 0.45,
                    base_y - height * 0.12,
                    width * 0.90,
                    height * 0.24,
                )
            )


    class _Page(QWidget):
        def __init__(self, title: str, subtitle: str = "") -> None:
            super().__init__()
            self.body = QVBoxLayout(self)
            self.body.setContentsMargins(26, 22, 26, 24)
            self.body.setSpacing(15)
            self.body.addWidget(_label(title, "pageTitle"))
            if subtitle:
                self.body.addWidget(_label(subtitle, "subtitle"))

        def add_card(self, title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
            frame = _card()
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(17, 15, 17, 15)
            layout.setSpacing(10)
            if title:
                layout.addWidget(_label(title, "subtitle"))
            self.body.addWidget(frame)
            return frame, layout



    class VoiceMeter(QWidget):
        """Compact level visualization driven only by the real microphone scalar."""

        _PATTERN = (
            0.32, 0.52, 0.78, 0.44, 0.68, 0.91, 0.58,
            0.83, 0.39, 0.71, 1.00, 0.64, 0.88, 0.47,
            0.74, 0.56, 0.92, 0.61, 0.81, 0.42, 0.67,
            0.86, 0.51, 0.76, 0.37, 0.70, 0.48, 0.63,
        )

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._level = 0.0
            self._active = False
            self.setMinimumHeight(52)
            self.setMaximumHeight(52)
            self.setAccessibleName("Real microphone activity meter")

        def set_level(self, level: Any) -> None:
            try:
                value = float(level)
            except (TypeError, ValueError, OverflowError):
                value = 0.0
            self._level = max(0.0, min(1.0, value))
            self.setAccessibleDescription(
                f"Microphone {'active' if self._active else 'standby'}, "
                f"normalized activity {self._level:.2f}"
            )
            self.update()

        def set_active(self, active: bool) -> None:
            self._active = bool(active)
            self.update()

        def paintEvent(self, event: Any) -> None:  # noqa: N802
            super().paintEvent(event)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            rect = self.rect().adjusted(9, 8, -9, -8)
            gradient = QLinearGradient(rect.left(), 0, rect.right(), 0)
            gradient.setColorAt(0.0, QColor("#D94AEF"))
            gradient.setColorAt(0.48, QColor("#8B67FF"))
            gradient.setColorAt(1.0, QColor("#58B8FF"))
            brush = QBrush(gradient)

            if not self._active:
                # A quiet line means standby. It is deliberately not presented
                # as microphone activity.
                pen = QPen(brush, 2.0)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                y = rect.center().y()
                painter.drawLine(rect.left(), y, rect.right(), y)
                return

            count = len(self._PATTERN)
            gap = 3.0
            width = max(1.7, (rect.width() - gap * (count - 1)) / count)
            activity = max(0.0, min(1.0, self._level))

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(brush)
            for index, shape in enumerate(self._PATTERN):
                amount = 0.07 + activity * shape * 0.93
                bar_height = max(4.0, rect.height() * min(1.0, amount))
                x = rect.left() + index * (width + gap)
                y = rect.center().y() - bar_height / 2.0
                painter.drawRoundedRect(
                    x,
                    y,
                    width,
                    bar_height,
                    width / 2.0,
                    width / 2.0,
                )


    class _StateSignalBridge(QObject):
        state_event = Signal(object)


    class HomePage(QWidget):
        """Reference-inspired glass Home screen backed only by truthful ASHER state."""

        def __init__(self, window: "AsherMainWindow") -> None:
            super().__init__()
            self.window = window

            self._layout_mode = ""
            root_layout = QVBoxLayout(self)
            root_layout.setContentsMargins(0, 0, 0, 0)
            root_layout.setSpacing(0)

            self.scroll = QScrollArea(self)
            self.scroll.setObjectName("homeScroll")
            self.scroll.setFrameShape(QFrame.Shape.NoFrame)
            self.scroll.setWidgetResizable(True)
            self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            root_layout.addWidget(self.scroll)

            self.canvas = QWidget()
            self.canvas.setObjectName("homeCanvas")
            self.canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Ignored,
            )
            self.scroll.setWidget(self.canvas)

            self.body = QVBoxLayout(self.canvas)
            self.body.setContentsMargins(8, 4, 8, 8)
            self.body.setSpacing(11)

            hero = QFrame()
            self.hero = hero
            hero.setObjectName("hero")
            hero.setMinimumHeight(438)
            hero_layout = QGridLayout(hero)
            self.hero_layout = hero_layout
            hero_layout.setContentsMargins(28, 24, 22, 24)
            hero_layout.setHorizontalSpacing(18)
            hero_layout.setVerticalSpacing(18)

            # Left: product identity, truthful state and primary actions.
            intro = QVBoxLayout()
            intro.setSpacing(8)
            self.eyebrow = _label(
                "PRIVATE  |  LOCAL-FIRST  |  USER-CONTROLLED",
                "eyebrow",
            )
            self.eyebrow.setWordWrap(False)
            intro.addWidget(self.eyebrow)

            intro.addWidget(GradientTitle("ASHER AI"))

            intro.addWidget(_label("Private intelligence. Local control.", "heroSubtitle"))
            intro.addWidget(
                _label(
                    "Voice, memory and guarded tools in one trusted workspace. "
                    "Consequential actions remain behind permissions and confirmation.",
                    "heroDescription",
                )
            )

            self.state = _label("STANDBY", "heroState")
            intro.addWidget(self.state)
            self.status = _label("Ready for Hey Asher", "heroMessage")
            intro.addWidget(self.status)

            action_row = QHBoxLayout()
            action_row.setSpacing(8)
            self.listen_button = QPushButton("Start listening")
            self.listen_button.setObjectName("primary")
            self.listen_button.clicked.connect(window.toggle_listening)
            action_row.addWidget(self.listen_button)

            conversation_button = QPushButton("Open conversation")
            conversation_button.setObjectName("secondaryAction")
            conversation_button.clicked.connect(lambda: window.select_page("Conversation"))
            action_row.addWidget(conversation_button)
            action_row.addStretch()
            intro.addLayout(action_row)

            chip_row = QHBoxLayout()
            chip_row.setSpacing(6)
            self.mic = _label("MIC | standby", "miniChip")
            self.voice = _label("VOICE | loading", "miniChip")
            self.provider = _label("MODE | local", "miniChip")
            for chip in (self.mic, self.voice, self.provider):
                chip.setWordWrap(False)
            chip_row.addWidget(self.mic)
            chip_row.addWidget(self.voice)
            chip_row.addWidget(self.provider)
            chip_row.addStretch()
            intro.addLayout(chip_row)

            self.last_reply = _label("", "replyPreview")
            self._last_reply_text = ""
            self.last_reply.setWordWrap(False)
            self.last_reply.setMaximumHeight(40)
            self.last_reply.setAccessibleName("Latest ASHER reply")
            self.last_reply.setVisible(False)
            intro.addStretch()

            intro_host = QWidget()
            self.intro_host = intro_host
            intro_host.setLayout(intro)
            intro_host.setMinimumWidth(286)
            intro_host.setMaximumWidth(390)

            # Centre: approved companion presentation inside the hero stage.
            orb_frame = OrbStage()
            self.orb_frame = orb_frame
            orb_layout = QVBoxLayout(orb_frame)
            orb_layout.setContentsMargins(0, 6, 0, 0)
            orb_layout.setSpacing(4)

            # Compact, elegant segmented companion selector
            selector_shell = QFrame()
            selector_shell.setObjectName("companionSelectorShell")
            selector_shell_layout = QHBoxLayout(selector_shell)
            selector_shell_layout.setContentsMargins(0, 0, 0, 0)
            selector_shell_layout.setSpacing(0)
            selector_shell_layout.addStretch()

            selector_pill = QFrame()
            selector_pill.setObjectName("companionSelectorPill")
            pill_layout = QHBoxLayout(selector_pill)
            pill_layout.setContentsMargins(6, 3, 6, 3)
            pill_layout.setSpacing(4)

            sel_lbl = _label("COMPANION", "companionSelectorLabel")
            sel_lbl.setWordWrap(False)
            pill_layout.addWidget(sel_lbl)

            self.companion_male_btn = QPushButton("♂ Male")
            self.companion_male_btn.setObjectName("companionSelectorBtn")
            self.companion_male_btn.setCheckable(True)
            self.companion_male_btn.setChecked(True)
            self.companion_male_btn.clicked.connect(lambda: self.window.set_companion_appearance("male"))
            pill_layout.addWidget(self.companion_male_btn)

            self.companion_female_btn = QPushButton("♀ Female")
            self.companion_female_btn.setObjectName("companionSelectorBtn")
            self.companion_female_btn.setCheckable(True)
            self.companion_female_btn.setChecked(False)
            self.companion_female_btn.clicked.connect(lambda: self.window.set_companion_appearance("female"))
            pill_layout.addWidget(self.companion_female_btn)

            selector_shell_layout.addWidget(selector_pill)
            selector_shell_layout.addStretch()
            orb_layout.addWidget(selector_shell, 0)

            self.companion_host = HomeCompanionHost(orb_frame)
            self.orb = self.companion_host
            self.orb.set_cinematic_mode(True)
            self.orb.set_transparent_canvas(True)
            self.orb.setMinimumSize(250, 250)
            self.orb.setMaximumSize(560, 560)
            self.orb.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            orb_layout.addWidget(self.companion_host, 1)

            # Right: live status only — no fake analytics.
            voice_panel = QFrame()
            self.voice_panel = voice_panel
            voice_panel.setObjectName("voicePanel")
            panel = QVBoxLayout(voice_panel)
            self.voice_layout = panel
            panel.setContentsMargins(17, 16, 17, 16)
            panel.setSpacing(8)

            voice_header = QHBoxLayout()
            live_voice = _label("LIVE VOICE", "panelKicker")
            live_voice.setWordWrap(False)
            voice_header.addWidget(live_voice)
            voice_header.addStretch()
            input_badge = _label("REAL INPUT", "panelBadge")
            input_badge.setWordWrap(False)
            voice_header.addWidget(input_badge)
            panel.addLayout(voice_header)
            self.voice_title = _label("Runtime status", "panelTitle")
            panel.addWidget(self.voice_title)

            self.command_state = _label("STANDBY", "panelState")
            panel.addWidget(self.command_state)
            self.command_message = _label("Ready for Hey Asher", "panelMessage")
            panel.addWidget(self.command_message)

            self.meter_label = _label("MIC LEVEL", "meterLabel")
            self.meter_label.setWordWrap(False)
            panel.addWidget(self.meter_label)
            meter_shell = QFrame()
            meter_shell.setObjectName("voiceMeterShell")
            meter_layout = QVBoxLayout(meter_shell)
            meter_layout.setContentsMargins(8, 4, 8, 4)
            self.waveform = VoiceMeter(meter_shell)
            meter_layout.addWidget(self.waveform)
            panel.addWidget(meter_shell)

            def stat_tile(title: str, value: str) -> tuple[QFrame, QLabel]:
                tile = QFrame()
                tile.setObjectName("statTile")
                layout = QVBoxLayout(tile)
                layout.setContentsMargins(11, 7, 11, 7)
                layout.setSpacing(3)
                layout.addWidget(_label(title, "statTitle"))
                value_label = _label(value, "statValue")
                layout.addWidget(value_label)
                return tile, value_label

            runtime_grid = QGridLayout()
            runtime_grid.setHorizontalSpacing(8)
            runtime_grid.setVerticalSpacing(8)

            brain_tile, self.brain_value = stat_tile("MODE", "Local only")
            session_tile, self.session_value = stat_tile("SESSION", "ACTIVE")
            mic_tile, self.mic_value = stat_tile("MIC", "STANDBY")
            voice_tile, self.voice_value = stat_tile("VOICE", "loading")

            runtime_grid.addWidget(brain_tile, 0, 0)
            runtime_grid.addWidget(session_tile, 0, 1)
            runtime_grid.addWidget(mic_tile, 1, 0)
            runtime_grid.addWidget(voice_tile, 1, 1)
            panel.addLayout(runtime_grid)
            panel.addStretch()

            quick_action = QHBoxLayout()
            quick_action.setSpacing(11)
            quick_copy = QVBoxLayout()
            quick_copy.setSpacing(2)
            self.quick_action_label = _label("START LISTENING", "quickActionTitle")
            self.quick_action_label.setWordWrap(False)
            quick_copy.addWidget(self.quick_action_label)
            self.quick_help = _label('Wake phrase: "Hey Asher"', "quickActionHelp")
            self.quick_help.setWordWrap(False)
            quick_copy.addWidget(self.quick_help)
            quick_action.addLayout(quick_copy, 1)
            quick_listen = MicActionButton()
            quick_listen.clicked.connect(window.toggle_listening)
            self.quick_listen_button = quick_listen
            quick_action.addWidget(quick_listen)
            panel.addLayout(quick_action)
            panel.addStretch()

            voice_panel.setMinimumWidth(264)
            voice_panel.setMaximumWidth(326)

            self._apply_hero_layout("wide")

            self.body.addWidget(hero, 7)

            # Real capabilities become a coherent glass shelf, not fake analytics.
            capability_shelf = QFrame()
            self.capability_shelf = capability_shelf
            capability_shelf.setObjectName("capabilityShelf")
            capability_layout = QVBoxLayout(capability_shelf)
            capability_layout.setContentsMargins(15, 13, 15, 15)
            capability_layout.setSpacing(10)

            capability_header = QHBoxLayout()
            capability_title = _label("CORE CAPABILITIES", "sectionKicker")
            capability_title.setWordWrap(False)
            capability_header.addWidget(capability_title)
            capability_header.addStretch()
            capability_meta = _label("LOCAL  |  GUARDED  |  OBSERVABLE", "sectionMeta")
            capability_meta.setWordWrap(False)
            capability_header.addWidget(capability_meta)
            capability_layout.addLayout(capability_header)

            capability_grid = QGridLayout()
            self.capability_grid = capability_grid
            capability_grid.setHorizontalSpacing(10)
            capability_grid.setVerticalSpacing(10)
            capabilities = (
                (
                    "Memory + Context",
                    "Local, editable and deletable context.",
                    "memory",
                    "#9A78FF",
                    "Memory",
                ),
                (
                    "VoiceGuard",
                    "Speaker identity and enrollment.",
                    "voiceguard",
                    "#5BBEFF",
                    "Users & VoiceGuard",
                ),
                (
                    "Permissions",
                    "Controlled tools with explicit risk gates.",
                    "permissions",
                    "#D467EF",
                    "Permissions",
                ),
                (
                    "Activity",
                    "An observable local audit trail.",
                    "activity",
                    "#778FFF",
                    "Activity log",
                ),
            )
            self.capability_buttons: list[CapabilityCard] = []
            for title, description, icon_key, accent, page_name in capabilities:
                button = CapabilityCard(title, description, icon_key, accent)
                button.clicked.connect(
                    lambda _checked=False, target=page_name: window.select_page(target)
                )
                self.capability_buttons.append(button)

            capability_layout.addLayout(capability_grid)
            self._apply_capability_layout("wide")
            self.body.addWidget(capability_shelf, 2)
            self.body.addWidget(self.last_reply)

            composer = QFrame()
            self.composer = composer
            composer.setObjectName("composer")
            composer_layout = QHBoxLayout(composer)
            composer_layout.setContentsMargins(13, 8, 9, 8)
            composer_layout.setSpacing(8)

            composer_layout.addWidget(ComposerGlyph())

            self.input = QLineEdit()
            self.input.setObjectName("heroInput")
            self.input.setPlaceholderText("Ask Asher or type a command…")
            self.input.returnPressed.connect(window.submit_text)
            composer_layout.addWidget(self.input, 1)

            enter_hint = _label("ENTER TO SEND", "composerHint")
            enter_hint.setWordWrap(False)
            composer_layout.addWidget(enter_hint)

            send = QPushButton("Send")
            send.setObjectName("sendButton")
            send.clicked.connect(window.submit_text)
            composer_layout.addWidget(send)
            self.body.addWidget(composer)

            # Backward-compatible attributes used by older code/tests.
            self.offline = self.provider
            self.api = self.provider

            QTimer.singleShot(0, self._sync_layout_mode)

        def resizeEvent(self, event: Any) -> None:  # noqa: N802
            super().resizeEvent(event)
            self._sync_layout_mode()
            self._update_reply_preview()

        def _sync_layout_mode(self) -> None:
            viewport = getattr(self, "scroll", None)
            available_width = (
                viewport.viewport().width()
                if viewport is not None and viewport.viewport().width() > 0
                else self.width()
            )
            if available_width < 1030:
                mode = "compact"
            else:
                mode = "wide_dense" if self.height() < 720 else "wide"
            if mode == self._layout_mode:
                return
            self._layout_mode = mode
            self._apply_hero_layout(mode)
            self._apply_capability_layout(mode)
            self.voice_layout.invalidate()
            self.hero_layout.invalidate()
            self.capability_grid.invalidate()
            self.body.invalidate()
            self.voice_panel.updateGeometry()
            self.hero.updateGeometry()
            self.capability_shelf.updateGeometry()
            self.canvas.updateGeometry()

        def _apply_hero_layout(self, mode: str) -> None:
            if not hasattr(self, "hero_layout"):
                return
            for widget in (self.intro_host, self.orb_frame, self.voice_panel):
                self.hero_layout.removeWidget(widget)

            condensed_panel = mode != "wide"
            self.voice_title.setVisible(not condensed_panel)
            self.quick_help.setVisible(not condensed_panel)
            self.meter_label.setVisible(not condensed_panel)
            self.waveform.setMinimumHeight(40 if condensed_panel else 52)
            self.waveform.setMaximumHeight(40 if condensed_panel else 52)
            self.voice_layout.setSpacing(4 if condensed_panel else 8)
            self.voice_layout.setContentsMargins(
                14 if condensed_panel else 17,
                13 if condensed_panel else 16,
                14 if condensed_panel else 17,
                13 if condensed_panel else 16,
            )

            if mode == "compact":
                self.hero.setMinimumHeight(770)
                self.hero.setMaximumHeight(16777215)
                self.intro_host.setMinimumWidth(270)
                self.intro_host.setMaximumWidth(380)
                self.orb_frame.setMinimumWidth(285)
                self.voice_panel.setMinimumWidth(0)
                self.voice_panel.setMaximumWidth(16777215)
                self.voice_panel.setMinimumHeight(282)
                self.hero_layout.addWidget(self.intro_host, 0, 0)
                self.hero_layout.addWidget(self.orb_frame, 0, 1)
                self.hero_layout.addWidget(self.voice_panel, 1, 0, 1, 2)
                self.hero_layout.setColumnStretch(0, 42)
                self.hero_layout.setColumnStretch(1, 58)
                self.hero_layout.setColumnStretch(2, 0)
                self.hero_layout.setRowStretch(0, 56)
                self.hero_layout.setRowStretch(1, 44)
                self.hero_layout.setRowMinimumHeight(0, 360)
                self.hero_layout.setRowMinimumHeight(1, 300)
                return

            dense = mode == "wide_dense"
            self.hero.setMinimumHeight(378 if dense else 438)
            self.hero.setMaximumHeight(378 if dense else 16777215)
            self.intro_host.setMinimumWidth(286)
            self.intro_host.setMaximumWidth(390)
            self.orb_frame.setMinimumWidth(230)
            self.voice_panel.setMinimumWidth(264)
            self.voice_panel.setMaximumWidth(326)
            self.voice_panel.setMinimumHeight(0)
            self.hero_layout.addWidget(self.intro_host, 0, 0)
            self.hero_layout.addWidget(self.orb_frame, 0, 1)
            self.hero_layout.addWidget(self.voice_panel, 0, 2)
            self.hero_layout.setColumnStretch(0, 34)
            self.hero_layout.setColumnStretch(1, 42)
            self.hero_layout.setColumnStretch(2, 28)
            self.hero_layout.setRowStretch(0, 1)
            self.hero_layout.setRowStretch(1, 0)
            self.hero_layout.setRowMinimumHeight(0, 0)
            self.hero_layout.setRowMinimumHeight(1, 0)

        def _apply_capability_layout(self, mode: str) -> None:
            if not hasattr(self, "capability_grid"):
                return
            for button in self.capability_buttons:
                self.capability_grid.removeWidget(button)

            columns = 2 if mode == "compact" else 4
            for index, button in enumerate(self.capability_buttons):
                self.capability_grid.addWidget(
                    button,
                    index // columns,
                    index % columns,
                )
            for column in range(4):
                self.capability_grid.setColumnStretch(
                    column,
                    1 if column < columns else 0,
                )

        def set_state_event(self, event: Any) -> None:
            state = getattr(event, "state", None)
            if isinstance(state, AssistantState):
                self.orb.set_state(state)
                state_text = state.value.upper().replace("_", " ")
                self.state.setText(state_text)
                self.command_state.setText(state_text)
                message = str(getattr(event, "message", "") or "").strip()
                if message:
                    self.status.setText(message)
                    self.command_message.setText(message)

        def refresh_status(self, status: Any) -> None:
            self.orb.set_state(status.state)
            microphone_level = (
                getattr(status, "microphone_level", 0.0)
                if status.microphone_active
                else 0.0
            )
            self.orb.set_audio_level(microphone_level)
            self.waveform.set_active(bool(status.microphone_active))
            self.waveform.set_level(microphone_level)

            state_text = status.state.value.upper().replace("_", " ")
            message = status.message or visual_for_state(status.state).label
            self.state.setText(state_text)
            self.status.setText(message)
            self.command_state.setText(state_text)
            self.command_message.setText(message)

            listen_text = "Stop listening" if status.microphone_active else "Start listening"
            self.listen_button.setText(listen_text)
            self.quick_listen_button.setText(listen_text)
            self.quick_action_label.setText(listen_text.upper())

            mic_text = "active" if status.microphone_active else "standby"
            self.mic.setText(f"MIC | {mic_text}")
            self.mic_value.setText(mic_text.upper())

            if status.offline:
                self.provider.setText("MODE | local only")
                self.brain_value.setText("Local only")
            elif status.api_configured:
                self.provider.setText("MODE | local + API")
                self.brain_value.setText("Local + API")
            else:
                self.provider.setText("MODE | local")
                self.brain_value.setText("Local only")

            self.session_value.setText(
                "ACTIVE" if status.owner_session_active else "EXPIRED"
            )
            self.eyebrow.setText(
                (
                    "SESSION ACTIVE"
                    if status.owner_session_active
                    else "SESSION EXPIRED"
                )
                + "  |  LOCAL-FIRST  |  USER-CONTROLLED"
            )

        def update_companion_selector(self, appearance: str) -> None:
            is_female = (str(appearance).lower() == "female")
            if hasattr(self, "companion_male_btn"):
                self.companion_male_btn.setChecked(not is_female)
            if hasattr(self, "companion_female_btn"):
                self.companion_female_btn.setChecked(is_female)

        def refresh_settings(self, settings: DesktopSettings) -> None:
            profile = str(settings.voice_profile or "default").replace("_", " ")
            self.voice.setText(f"VOICE | {profile}")
            self.voice_value.setText(profile)
            comp = getattr(settings, "companion_appearance", "male")
            if hasattr(self, "companion_host"):
                self.companion_host.set_character(comp)
            self.update_companion_selector(comp)

        def show_reply(self, message: str) -> None:
            text = str(message or "").strip()
            self._last_reply_text = text
            self._update_reply_preview()
            self.last_reply.setToolTip(text)
            self.last_reply.setAccessibleDescription(text)
            self.last_reply.setVisible(bool(text))
            if text:
                QTimer.singleShot(0, self._update_reply_preview)

        def _update_reply_preview(self) -> None:
            if not hasattr(self, "last_reply") or not self._last_reply_text:
                return
            available = max(180, self.last_reply.width() - 28)
            self.last_reply.setText(
                self.last_reply.fontMetrics().elidedText(
                    self._last_reply_text,
                    Qt.TextElideMode.ElideRight,
                    available,
                )
            )


    class CompanionModePage(QWidget):
        """Minimal immersive scene shown only during an active voice interaction.

        Workspace owns history, diagnostics and detailed runtime data. Companion
        mode intentionally keeps only the living ASHER visual, truthful state
        text, a tiny provider indicator and the always-available local Stop path.
        """

        def __init__(self, window: "AsherMainWindow") -> None:
            super().__init__()
            self.window = window
            self.setObjectName("companionMode")
            # QWidget style backgrounds are platform-dependent unless the
            # widget explicitly paints its Window role.  Match the cinematic
            # canvas exactly so the square orb widget can never be visible.
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            companion_palette = self.palette()
            companion_palette.setColor(QPalette.ColorRole.Window, QColor("#02050B"))
            self.setPalette(companion_palette)
            self.setAutoFillBackground(True)
            root = QVBoxLayout(self)
            self.root_layout = root
            # Full-frame Companion canvas. Controls keep their own padding.
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)

            top = QHBoxLayout()
            top.setContentsMargins(24, 16, 24, 8)
            self.brand = _label("ASHER", "companionBrand")
            top.addWidget(self.brand)
            top.addStretch()
            self.presence = _label("LOCAL", "companionTelemetry")
            top.addWidget(self.presence)
            gesture = QPushButton("GESTURES")
            gesture.setObjectName("companionGesture")
            gesture.setCheckable(True)
            gesture.setToolTip(
                "Enable local camera hand tracking for visual rotation and energy unfolding"
            )
            gesture.toggled.connect(self._toggle_gestures)
            self.gesture_button = gesture
            top.addWidget(gesture)
            stop = QPushButton("STOP")
            stop.setObjectName("companionStop")
            stop.setToolTip("Emergency stop — cancel the active ASHER plan and voice output")
            stop.setMinimumWidth(74)
            stop.clicked.connect(window.emergency_stop)
            self.stop_button = stop
            top.addWidget(stop)
            self.top_layout = top
            root.addLayout(top)

            self.orb = CompanionOrbHost(self)
            self.orb.set_overlay_text("LISTENING", "")
            self.orb.rendererChanged.connect(self._renderer_changed)
            self.orb.gestureStateChanged.connect(self._gesture_state_changed)
            self._renderer_changed(
                self.orb.uses_webgl,
                self.orb.renderer_error or "Local WebGL renderer is starting",
            )
            root.addWidget(self.orb, 1)

            # Hidden compatibility labels keep real state/message text available
            # to accessibility/smoke tests without cluttering the immersive view.
            self.state = _label("STANDBY", "companionState")
            self.message = _label("Say “Hey Asher”", "companionMessage")
            self.state.setVisible(False)
            self.message.setVisible(False)
            root.addWidget(self.state)
            root.addWidget(self.message)

            self.confirm = QFrame()
            self.confirm.setObjectName("companionConfirm")
            self.confirm.setMaximumWidth(760)
            confirm_layout = QVBoxLayout(self.confirm)
            confirm_layout.setContentsMargins(18, 12, 18, 12)
            confirm_layout.setSpacing(8)
            self.confirm_summary = _label("", "companionMessage")
            self.confirm_summary.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            confirm_layout.addWidget(self.confirm_summary)
            self.confirm_preview_label = _label("EXACT PREVIEW", "companionTelemetry")
            self.confirm_preview_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            confirm_layout.addWidget(self.confirm_preview_label)
            self.confirm_preview = QPlainTextEdit()
            self.confirm_preview.setObjectName("companionPreview")
            self.confirm_preview.setReadOnly(True)
            self.confirm_preview.setAccessibleName("Exact pending action preview")
            self.confirm_preview.setMinimumHeight(96)
            self.confirm_preview.setMaximumHeight(150)
            confirm_layout.addWidget(self.confirm_preview)
            actions = QHBoxLayout()
            actions.addStretch()
            cancel = QPushButton("Cancel")
            cancel.clicked.connect(window.reject_pending)
            approve = QPushButton("Approve")
            approve.setObjectName("primary")
            approve.clicked.connect(window.approve_pending)
            self.cancel_button = cancel
            actions.addWidget(cancel)
            actions.addWidget(approve)
            actions.addStretch()
            confirm_layout.addLayout(actions)
            self.confirm.setVisible(False)
            root.addWidget(self.confirm, 0, Qt.AlignmentFlag.AlignHCenter)
            QTimer.singleShot(0, self._fit_orb_to_viewport)

        def _renderer_changed(self, webgl_active: bool, detail: str) -> None:
            self.gesture_button.setEnabled(bool(webgl_active))
            self.gesture_button.setToolTip(str(detail or "Local WebGL renderer"))
            if not webgl_active and self.gesture_button.isChecked():
                self.gesture_button.setChecked(False)

        def _toggle_gestures(self, enabled: bool) -> None:
            applied = self.orb.set_gesture_enabled(bool(enabled))
            if applied != bool(enabled):
                self.gesture_button.setChecked(applied)
            self.gesture_button.setText("GESTURES ON" if applied else "GESTURES")

        def _gesture_state_changed(self, enabled: bool, detail: str) -> None:
            if self.gesture_button.isChecked() != bool(enabled):
                self.gesture_button.setChecked(bool(enabled))
            self.gesture_button.setText("GESTURES ON" if enabled else "GESTURES")
            if detail:
                self.gesture_button.setToolTip(str(detail))

        def _fit_orb_to_viewport(self) -> int:
            """Keep WebGL fluid and report the usable viewport short edge."""

            top_height = max(
                self.brand.sizeHint().height(),
                self.stop_button.sizeHint().height(),
            )
            confirmation_height = (
                self.confirm.sizeHint().height() if not self.confirm.isHidden() else 0
            )
            available_width = max(1, self.width())
            available_height = max(
                1,
                self.height() - top_height - confirmation_height - 24,
            )
            self.orb.setMinimumSize(1, 1)
            self.orb.setMaximumSize(16777215, 16777215)
            self.orb.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            self.orb.updateGeometry()
            return min(available_width, available_height)

        def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt callback
            super().resizeEvent(event)
            self._fit_orb_to_viewport()

        def set_state_event(self, event: Any) -> None:
            state = getattr(event, "state", None)
            if isinstance(state, AssistantState):
                self.orb.set_state(state)
                self.state.setText(state.value.upper().replace("_", " "))
            message = str(getattr(event, "message", "") or "").strip()
            if message:
                self.message.setText(message)
            self.orb.set_overlay_text(self.state.text(), self.message.text())

        def refresh_status(self, status: Any) -> None:
            self.orb.set_state(status.state)
            microphone_level = (
                getattr(status, "microphone_level", 0.0)
                if status.microphone_active
                else 0.0
            )
            self.orb.set_audio_level(microphone_level)
            state_text = status.state.value.upper().replace("_", " ")
            message = status.message or visual_for_state(status.state).label
            self.state.setText(state_text)
            self.message.setText(message)
            self.orb.set_overlay_text(state_text, message)
            self.presence.setText("LOCAL" if status.offline else "LOCAL + API")
            if status.emergency_stopped:
                self.presence.setText("STOPPED")

        def refresh_settings(self, settings: DesktopSettings) -> None:
            profile = str(settings.voice_profile or "default").replace("_", " ")
            self.presence.setToolTip(f"Voice: {profile}")

        def refresh_pending(self, pending: PendingAction | None) -> None:
            self.confirm.setVisible(pending is not None)
            if pending is None:
                self.confirm_summary.setText("")
                self.confirm_preview.clear()
                self._fit_orb_to_viewport()
                return
            self.confirm_summary.setText(
                f"{pending.action} · {pending.target} · {pending.effect} · "
                f"{pending.risk.name.replace('_', ' ')}"
            )
            # Render the complete preview as plain text so message/body fields
            # remain visible and selectable without interpreting user-provided
            # content as rich text or HTML.
            self.confirm_preview.setPlainText(
                json.dumps(dict(pending.preview), indent=2, ensure_ascii=False, sort_keys=True)
            )
            self._fit_orb_to_viewport()


    class ConversationPage(_Page):
        def __init__(self, window: "AsherMainWindow") -> None:
            super().__init__("Conversation", "A chronological view of the current session and its observable task steps.")
            self.window = window
            card, layout = self.add_card("Timeline")
            self.timeline = QListWidget()
            self.timeline.setAlternatingRowColors(True)
            layout.addWidget(self.timeline, 1)
            steps_card, steps_layout = self.add_card("Live task steps")
            self.steps = QTreeWidget()
            self.steps.setHeaderLabels(["Step", "Status", "Evidence"])
            self.steps.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            self.steps.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            self.steps.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
            steps_layout.addWidget(self.steps)

        def refresh(self, turns: tuple[ConversationTurn, ...], steps: tuple[LiveStep, ...]) -> None:
            self.timeline.clear()
            for turn in turns:
                timestamp = turn.timestamp.strftime("%H:%M:%S")
                self.timeline.addItem(f"{timestamp}  {turn.sender}: {turn.message}")
            self.steps.clear()
            for step in steps:
                QTreeWidgetItem(self.steps, [step.description, step.status, step.detail])
            self.timeline.scrollToBottom()


    class ConfirmationPage(_Page):
        def __init__(self, window: "AsherMainWindow") -> None:
            super().__init__("Pending confirmation", "Review the exact target and effect before approving a consequential action.")
            self.window = window
            card, layout = self.add_card()
            form = QFormLayout()
            self.action = _label("—")
            self.target = _label("—")
            self.effect = _label("—")
            self.risk = _label("—")
            self.expires = _label("—")
            form.addRow("Action", self.action)
            form.addRow("Target", self.target)
            form.addRow("Effect", self.effect)
            form.addRow("Risk", self.risk)
            form.addRow("Expires", self.expires)
            layout.addLayout(form)
            layout.addWidget(_label("Preview", "subtitle"))
            self.preview = QPlainTextEdit()
            self.preview.setReadOnly(True)
            self.preview.setMinimumHeight(150)
            layout.addWidget(self.preview)
            row = QHBoxLayout()
            self.approve = QPushButton("Approve in local UI")
            self.approve.setObjectName("primary")
            self.approve.clicked.connect(window.approve_pending)
            self.reject = QPushButton("Reject")
            self.reject.clicked.connect(window.reject_pending)
            row.addWidget(self.approve)
            row.addWidget(self.reject)
            row.addStretch()
            layout.addLayout(row)
            self.empty = _label("There is no action waiting for approval.", "muted")
            layout.addWidget(self.empty)

        def refresh(self, pending: PendingAction | None) -> None:
            active = pending is not None
            self.approve.setEnabled(active)
            self.reject.setEnabled(active)
            self.empty.setVisible(not active)
            if not active:
                for label in (self.action, self.target, self.effect, self.risk, self.expires):
                    label.setText("—")
                self.preview.clear()
                return
            self.action.setText(pending.action)
            self.target.setText(pending.target)
            self.effect.setText(pending.effect)
            self.risk.setText(pending.risk.name.replace("_", " "))
            self.expires.setText(pending.expires_at.strftime("%H:%M:%S UTC"))
            self.preview.setPlainText(json.dumps(dict(pending.preview), indent=2, ensure_ascii=False))


    class MemoryPage(_Page):
        def __init__(self, window: "AsherMainWindow") -> None:
            super().__init__("Memory manager", "Inspect, edit, or remove local memories. Sensitive values stay behind the controller boundary.")
            self.window = window
            card, layout = self.add_card()
            self.table = QTableWidget(0, 5)
            self.table.setHorizontalHeaderLabels(["Key", "Value", "Type", "Sensitivity", "Updated"])
            self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.table.verticalHeader().setVisible(False)
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            layout.addWidget(self.table)
            self.reveal_sensitive = QCheckBox("Reveal sensitive values in this local view")
            self.reveal_sensitive.stateChanged.connect(lambda _state: self.refresh(self.window.controller.list_memories()))
            layout.addWidget(self.reveal_sensitive)
            row = QHBoxLayout()
            add = QPushButton("Add memory")
            add.setObjectName("primary")
            add.clicked.connect(window.add_memory)
            edit = QPushButton("Edit selected")
            edit.clicked.connect(window.edit_memory)
            delete = QPushButton("Delete selected")
            delete.setObjectName("dangerButton")
            delete.clicked.connect(window.delete_memory)
            self.export_button = QPushButton("Export JSON")
            self.export_button.clicked.connect(window.export_memories)
            row.addWidget(add)
            row.addWidget(edit)
            row.addWidget(delete)
            row.addWidget(self.export_button)
            row.addStretch()
            layout.addLayout(row)

        def refresh(self, records: tuple[MemoryRecord, ...]) -> None:
            self.table.setRowCount(0)
            for record in records:
                row = self.table.rowCount()
                self.table.insertRow(row)
                values = (
                    record.key,
                    record.value if self.reveal_sensitive.isChecked() or record.sensitivity != "sensitive" else "[protected]",
                    record.memory_type,
                    record.sensitivity,
                    record.updated_at.strftime("%Y-%m-%d %H:%M"),
                )
                item = QTableWidgetItem(record.memory_id)
                item.setData(Qt.ItemDataRole.UserRole, record.memory_id)
                self.table.setVerticalHeaderItem(row, item)
                for col, value in enumerate(values):
                    self.table.setItem(row, col, QTableWidgetItem(str(value)))

        def selected_id(self) -> str | None:
            row = self.table.currentRow()
            if row < 0:
                return None
            header = self.table.verticalHeaderItem(row)
            return header.data(Qt.ItemDataRole.UserRole) if header else None


    class UsersPage(_Page):
        def __init__(self, window: "AsherMainWindow") -> None:
            super().__init__("Users & VoiceGuard", "Manage roles and enrollment state. Recording/training stays delegated to the VoiceGuard adapter.")
            self.window = window
            card, layout = self.add_card("Authorized speakers")
            self.table = QTableWidget(0, 5)
            self.table.setHorizontalHeaderLabels(["Name", "Role", "Enrollment", "Samples", "ID"])
            self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
            layout.addWidget(self.table)
            row = QHBoxLayout()
            enroll = QPushButton("Enroll user")
            enroll.setObjectName("primary")
            enroll.clicked.connect(window.enroll_user)
            sample = QPushButton("Capture sample")
            sample.clicked.connect(window.capture_sample)
            train = QPushButton("Train VoiceGuard")
            train.clicked.connect(window.train_voiceguard)
            revoke = QPushButton("Revoke selected")
            revoke.setObjectName("dangerButton")
            revoke.clicked.connect(window.revoke_user)
            for button in (enroll, sample, train, revoke):
                row.addWidget(button)
            row.addStretch()
            layout.addLayout(row)
            self.note = _label("No classifier result is shown until a real adapter supplies one.", "warning")
            layout.addWidget(self.note)

        def refresh(self, records: tuple[UserRecord, ...]) -> None:
            self.table.setRowCount(0)
            for record in records:
                row = self.table.rowCount()
                self.table.insertRow(row)
                for col, value in enumerate((record.display_name, record.role, record.enrollment_status, record.samples, record.user_id)):
                    item = QTableWidgetItem(str(value))
                    if col == 4:
                        item.setData(Qt.ItemDataRole.UserRole, record.user_id)
                    self.table.setItem(row, col, item)

        def selected_id(self) -> str | None:
            row = self.table.currentRow()
            if row < 0:
                return None
            item = self.table.item(row, 4)
            return item.data(Qt.ItemDataRole.UserRole) if item else None


    class PermissionsPage(_Page):
        def __init__(self, window: "AsherMainWindow") -> None:
            super().__init__("Permission controls", "Risk labels are visible beside every capability; guest permissions are constrained by the controller.")
            self.window = window
            card, layout = self.add_card()
            self.table = QTableWidget(0, 4)
            self.table.setHorizontalHeaderLabels(["Role", "Capability", "Risk", "Allowed"])
            self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            self.table.itemChanged.connect(window.permission_changed)
            layout.addWidget(self.table)
            self._updating = False

        def refresh(self, records: tuple[PermissionRecord, ...]) -> None:
            self._updating = True
            try:
                self.table.setRowCount(0)
                for record in records:
                    row = self.table.rowCount()
                    self.table.insertRow(row)
                    for col, value in enumerate((record.role, record.capability, record.risk.name.replace("_", " "))):
                        self.table.setItem(row, col, QTableWidgetItem(str(value)))
                    allowed = QTableWidgetItem()
                    allowed.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                    allowed.setCheckState(Qt.CheckState.Checked if record.allowed else Qt.CheckState.Unchecked)
                    allowed.setData(
                        Qt.ItemDataRole.UserRole,
                        (record.role, record.capability, record.actor_id),
                    )
                    self.table.setItem(row, 3, allowed)
            finally:
                self._updating = False


    class ActivityPage(_Page):
        def __init__(self, window: "AsherMainWindow") -> None:
            super().__init__("Activity log", "Local audit events are shown with redacted details and no secret payloads.")
            self.window = window
            card, layout = self.add_card()
            self.table = QTableWidget(0, 4)
            self.table.setHorizontalHeaderLabels(["Time", "Event", "Result", "Details"])
            self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
            layout.addWidget(self.table)

        def refresh(self, records: tuple[AuditRecord, ...]) -> None:
            self.table.setRowCount(0)
            for record in records:
                row = self.table.rowCount()
                self.table.insertRow(row)
                values = (
                    record.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    record.event,
                    record.result,
                    json.dumps(dict(record.details), ensure_ascii=False),
                )
                for col, value in enumerate(values):
                    self.table.setItem(row, col, QTableWidgetItem(str(value)))


    class SettingsPage(_Page):
        def __init__(self, window: "AsherMainWindow") -> None:
            super().__init__("Settings", "Switch voice profiles at runtime and control local/API routing preferences.")
            self.window = window
            card, layout = self.add_card("Speech output")
            form = QFormLayout()
            self.voice = QComboBox()
            for name, label in window.controller.voice_profiles():
                self.voice.addItem(label, name)
            self.speed = QDoubleSpinBox()
            self.speed.setRange(0.5, 2.0)
            self.speed.setSingleStep(0.05)
            self.speed.setDecimals(2)
            self.style = QLineEdit()
            form.addRow("Voice profile", self.voice)
            form.addRow("Speed", self.speed)
            form.addRow("Style", self.style)
            layout.addLayout(form)

            privacy_card, privacy_layout = self.add_card("Connectivity and privacy")
            self.offline = QCheckBox("Offline-only mode (never use a network speech provider)")
            self.api = QCheckBox("Allow configured API provider")
            self.mic = QSpinBox()
            self.mic.setRange(-1, 64)
            self.mic.setSpecialValueText("Automatic")
            privacy_layout.addWidget(self.offline)
            privacy_layout.addWidget(self.api)
            row = QHBoxLayout()
            row.addWidget(_label("Microphone index", "muted"))
            row.addWidget(self.mic)
            row.addStretch()
            privacy_layout.addLayout(row)
            comp_card, comp_layout = self.add_card("Companion appearance")
            comp_form = QFormLayout()
            self.companion_appearance = QComboBox()
            self.companion_appearance.addItem(MaleCompanion.label, "male")
            self.companion_appearance.addItem(FemaleCompanion.label, "female")
            comp_form.addRow("Appearance", self.companion_appearance)
            comp_layout.addLayout(comp_form)
            comp_tip = _label("Selects the companion appearance in the Home workspace.", "muted")
            comp_layout.addWidget(comp_tip)

            self.save = QPushButton("Apply settings")
            self.save.setObjectName("primary")
            self.save.clicked.connect(window.apply_settings)
            comp_layout.addWidget(self.save)
            self.message = _label("Changes apply to future speech and workspace presentation without restarting ASHER.", "muted")
            comp_layout.addWidget(self.message)

        def load(self, settings: DesktopSettings) -> None:
            index = self.voice.findData(settings.voice_profile)
            if index >= 0:
                self.voice.setCurrentIndex(index)
            comp_app = getattr(settings, "companion_appearance", "male")
            comp_idx = self.companion_appearance.findData(comp_app)
            if comp_idx >= 0:
                self.companion_appearance.setCurrentIndex(comp_idx)
            self.speed.setValue(settings.speech_speed)
            self.style.setText(settings.speech_style)
            self.offline.setChecked(settings.offline_only)
            self.api.setChecked(settings.api_enabled)
            self.mic.setValue(-1 if settings.microphone_index is None else settings.microphone_index)


    class DiagnosticsPage(_Page):
        def __init__(self, window: "AsherMainWindow") -> None:
            super().__init__("Diagnostics", "Check microphone, transcription, speech, UI, and API readiness without exposing credentials.")
            self.window = window
            card, layout = self.add_card()
            self.summary = _label("Diagnostics have not run yet.", "state")
            layout.addWidget(self.summary)
            self.table = QTableWidget(0, 2)
            self.table.setHorizontalHeaderLabels(["Check", "Status"])
            self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            layout.addWidget(self.table)
            run = QPushButton("Run diagnostics")
            run.setObjectName("primary")
            run.clicked.connect(window.run_diagnostics)
            layout.addWidget(run)

        def refresh(self, report: DiagnosticReport) -> None:
            self.summary.setText(report.summary)
            self.table.setRowCount(0)
            for name, status in report.checks.items():
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(name))
                self.table.setItem(row, 1, QTableWidgetItem(status))


    class AsherMainWindow(QMainWindow):
        """Full desktop shell with responsive, controller-backed views."""

        def __init__(
            self,
            controller: DesktopControllerProtocol | None = None,
            *,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self.controller = controller or DesktopController()
            self._workers: set[FunctionWorker] = set()
            self._nav_buttons: list[QPushButton] = []
            self._pages: dict[str, QWidget] = {}
            self._state_unsubscribe: Callable[[], None] | None = None
            self._pending_view_signature: object = object()
            self._state_bridge = _StateSignalBridge(self)
            self._state_bridge.state_event.connect(self._on_state_event)
            self._companion_fullscreen_active = False
            self._companion_restore_pending = False
            self._restore_maximized_after_companion = False
            self._restore_fullscreen_after_companion = False
            self._audio_poll_enabled = False
            self._compact_shell: bool | None = None
            self.setWindowTitle("Asher — authenticated personal companion")
            self.setMinimumSize(1120, 720)
            self.resize(1500, 900)
            self.setStyleSheet(APP_STYLE + UI5D_STYLE + UI5E_STYLE)
            self._build_shell()
            QTimer.singleShot(0, self._sync_workspace_density)
            self._fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
            self._fullscreen_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            self._fullscreen_shortcut.activated.connect(self._toggle_companion_fullscreen)
            self._windowed_shortcut = QShortcut(QKeySequence("Esc"), self)
            self._windowed_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            self._windowed_shortcut.activated.connect(self._leave_companion_fullscreen)
            subscribe = getattr(self.controller, "subscribe_state", None)
            if callable(subscribe):
                self._state_unsubscribe = subscribe(self._state_bridge.state_event.emit)
            self._refresh_views()
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._refresh_status)
            self._timer.start(500)
            # Full status/view refresh remains deliberately slow.  A small
            # presentation-only poll lets real microphone RMS reach the orb in
            # time to feel continuous without rebuilding any page content.
            self._audio_timer = QTimer(self)
            self._audio_timer.setInterval(66)
            self._audio_timer.timeout.connect(self._refresh_orb_audio)
            self._sync_audio_timer()

        def _build_shell(self) -> None:
            self.mode_stack = QStackedWidget()
            self.setCentralWidget(self.mode_stack)

            workspace = AmbientWorkspace()
            self.workspace = workspace
            workspace.setObjectName("workspace")
            root = QVBoxLayout(workspace)
            self.workspace_root_layout = root
            root.setContentsMargins(24, 22, 24, 22)
            root.setSpacing(0)

            shell = QFrame()
            self.workspace_shell = shell
            shell.setObjectName("workspaceShell")
            shell_layout = QHBoxLayout(shell)
            self.workspace_shell_layout = shell_layout
            shell_layout.setContentsMargins(14, 14, 14, 14)
            shell_layout.setSpacing(12)
            root.addWidget(shell, 1)

            sidebar = QFrame()
            self.sidebar = sidebar
            sidebar.setObjectName("sidebar")
            sidebar.setFixedWidth(190)
            side = QVBoxLayout(sidebar)
            self.sidebar_layout = side
            side.setContentsMargins(14, 19, 14, 16)
            side.setSpacing(4)
            brand_row = QHBoxLayout()
            brand_row.setSpacing(10)
            brand_mark = BrandMark()
            brand_row.addWidget(brand_mark)
            brand_copy = QVBoxLayout()
            brand_copy.setSpacing(1)
            self.brand_label = _label("ASHER", "brand")
            self.brand_label.setWordWrap(False)
            self.brand_subtitle = _label("PRIVATE COMPANION", "brandSub")
            self.brand_subtitle.setWordWrap(False)
            brand_copy.addWidget(self.brand_label)
            brand_copy.addWidget(self.brand_subtitle)
            brand_row.addLayout(brand_copy, 1)
            side.addLayout(brand_row)
            side.addSpacing(18)
            for name, icon_key in NAV_ITEMS:
                display_name = name.replace(" & ", " + ")
                button = VectorNavButton(display_name, icon_key)
                button.setToolTip(name)
                button.clicked.connect(lambda _checked=False, item=name: self.select_page(item))
                self._nav_buttons.append(button)
                side.addWidget(button)
            side.addStretch()
            core = QFrame()
            self.sidebar_core = core
            core.setObjectName("sidebarCore")
            core_layout = QVBoxLayout(core)
            core_layout.setContentsMargins(11, 10, 11, 10)
            core_layout.setSpacing(4)
            core_layout.addWidget(_label("LOCAL CORE", "sidebarCoreTitle"))
            core_layout.addWidget(_label("Memory stays local", "sidebarCoreText"))
            core_layout.addWidget(_label("Guarded actions · auditable steps", "sidebarCoreText"))
            side.addWidget(core)
            shell_layout.addWidget(sidebar)

            right = QVBoxLayout()
            right.setContentsMargins(0, 0, 0, 0)
            right.setSpacing(12)
            header = QFrame()
            self.header = header
            header.setObjectName("header")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(17, 8, 12, 8)
            header_layout.setSpacing(9)

            state_dot = QFrame()
            self.header_state_dot = state_dot
            state_dot.setObjectName("stateDot")
            state_dot.setFixedSize(8, 8)
            state_dot.setAccessibleName("Assistant status indicator")
            header_layout.addWidget(state_dot)
            self.header_state = _label("STANDBY", "state")
            self.header_state.setWordWrap(False)
            header_layout.addWidget(self.header_state)
            self.header_message = _label("Ready for Hey Asher", "muted")
            self.header_message.setWordWrap(False)
            header_layout.addWidget(self.header_message, 1)
            header_layout.addStretch(1)

            trust_cluster = QFrame()
            trust_cluster.setObjectName("trustCluster")
            trust_layout = QHBoxLayout(trust_cluster)
            trust_layout.setContentsMargins(5, 4, 5, 4)
            trust_layout.setSpacing(2)
            self.header_offline = StatusChipLabel("LOCAL ONLY", "#67C8FF")
            trust_layout.addWidget(self.header_offline)
            self.header_api = StatusChipLabel("API DISABLED", "#9386A8")
            trust_layout.addWidget(self.header_api)
            self.header_session = StatusChipLabel("SESSION ACTIVE", "#82E7C7")
            trust_layout.addWidget(self.header_session)
            header_layout.addWidget(trust_cluster)
            self.reauthenticate_button = QPushButton("Re-authenticate")
            self.reauthenticate_button.setObjectName("reauthButton")
            self.reauthenticate_button.setToolTip(
                "Use device authentication to create a fresh owner session"
            )
            self.reauthenticate_button.clicked.connect(self.reauthenticate_owner)
            self.reauthenticate_button.setVisible(False)
            header_layout.addWidget(self.reauthenticate_button)
            stop = QPushButton("EMERGENCY STOP")
            stop.setObjectName("dangerButton")
            stop.setMinimumHeight(38)
            stop.setMaximumWidth(170)
            stop.clicked.connect(self.emergency_stop)
            self.emergency_stop_button = stop
            header_layout.addWidget(stop)
            reset_stop = QPushButton("Reset stop")
            reset_stop.clicked.connect(self.reset_emergency_stop)
            reset_stop.setVisible(False)
            self.reset_stop_button = reset_stop
            header_layout.addWidget(reset_stop)
            right.addWidget(header)

            self.stack = QStackedWidget()
            self.home = HomePage(self)
            self.conversation_page = ConversationPage(self)
            self.confirmation_page = ConfirmationPage(self)
            self.memory_page = MemoryPage(self)
            self.users_page = UsersPage(self)
            self.permissions_page = PermissionsPage(self)
            self.activity_page = ActivityPage(self)
            self.settings_page = SettingsPage(self)
            self.diagnostics_page = DiagnosticsPage(self)
            pages = (
                ("Home", self.home),
                ("Conversation", self.conversation_page),
                ("Confirmation", self.confirmation_page),
                ("Memory", self.memory_page),
                ("Users & VoiceGuard", self.users_page),
                ("Permissions", self.permissions_page),
                ("Activity log", self.activity_page),
                ("Settings", self.settings_page),
                ("Diagnostics", self.diagnostics_page),
            )
            for name, page in pages:
                self._pages[name] = page
                self.stack.addWidget(page)
            right.addWidget(self.stack, 1)
            shell_layout.addLayout(right, 1)

            self.companion_mode = CompanionModePage(self)
            self.mode_stack.addWidget(workspace)
            self.mode_stack.addWidget(self.companion_mode)
            self.mode_stack.setCurrentWidget(workspace)
            # Warm the sealed local WebGL surface while Workspace is visible.
            # Mode switching itself still follows controller truth immediately.
            self.companion_mode.orb.prewarm()
            self.select_page("Home")

        def resizeEvent(self, event: Any) -> None:  # noqa: N802
            super().resizeEvent(event)
            if hasattr(self, "workspace"):
                self._sync_workspace_density()

        def _sync_workspace_density(self) -> None:
            """Use an icon rail and quieter header on constrained viewports."""

            if not hasattr(self, "sidebar"):
                return
            compact = self.width() < 1300 or self.height() < 800
            if compact == self._compact_shell:
                return
            self._compact_shell = compact

            if compact:
                self.workspace_root_layout.setContentsMargins(14, 12, 14, 12)
                self.workspace_shell_layout.setContentsMargins(10, 10, 10, 10)
                self.workspace_shell_layout.setSpacing(10)
                self.sidebar.setFixedWidth(78)
                self.sidebar_layout.setContentsMargins(10, 17, 10, 14)
            else:
                self.workspace_root_layout.setContentsMargins(24, 22, 24, 22)
                self.workspace_shell_layout.setContentsMargins(14, 14, 14, 14)
                self.workspace_shell_layout.setSpacing(12)
                self.sidebar.setFixedWidth(190)
                self.sidebar_layout.setContentsMargins(14, 19, 14, 16)

            self.brand_label.setVisible(not compact)
            self.brand_subtitle.setVisible(not compact)
            self.sidebar_core.setVisible(not compact)
            self.header_message.setVisible(
                not compact or self.header_state.text() == "ERROR"
            )
            for button, (name, _icon_key) in zip(self._nav_buttons, NAV_ITEMS):
                button.setText("" if compact else name.replace(" & ", " + "))
            QTimer.singleShot(0, self.home._sync_layout_mode)

        def _set_companion_mode(self, enabled: bool) -> None:
            target = self.companion_mode if enabled else self.workspace
            if self.mode_stack.currentWidget() is target:
                return

            if enabled:
                if hasattr(self.home, "companion_host"):
                    self.home.companion_host.pause_rendering()
                self._restore_maximized_after_companion = self.isMaximized()
                self._restore_fullscreen_after_companion = self.isFullScreen()
                self._companion_restore_pending = self.isVisible()
                self.mode_stack.setCurrentWidget(target)
                if self._companion_restore_pending:
                    if not self.isFullScreen():
                        self.showFullScreen()
                    self._companion_fullscreen_active = True
                return

            self.mode_stack.setCurrentWidget(target)
            if self._companion_restore_pending:
                if self._restore_fullscreen_after_companion:
                    self.showFullScreen()
                elif self._restore_maximized_after_companion:
                    self.showMaximized()
                else:
                    self.showNormal()
            self._companion_fullscreen_active = False
            self._companion_restore_pending = False
            if hasattr(self, "home") and hasattr(self.home, "companion_host") and hasattr(self, "stack") and self.stack.currentWidget() is self.home:
                self.home.companion_host.resume_rendering()

        def _leave_companion_fullscreen(self) -> bool:
            """Reveal the regular resizable Companion window without ending it."""

            if (
                self.mode_stack.currentWidget() is not self.companion_mode
                or not self.isFullScreen()
            ):
                return False
            if (
                self._restore_maximized_after_companion
                and not self._restore_fullscreen_after_companion
            ):
                self.showMaximized()
            else:
                self.showNormal()
            self._companion_fullscreen_active = False
            return True

        def _toggle_companion_fullscreen(self) -> bool:
            """Toggle F11 presentation mode while preserving the active scene."""

            if self.mode_stack.currentWidget() is not self.companion_mode:
                return False
            if self.isFullScreen():
                return self._leave_companion_fullscreen()
            self.showFullScreen()
            self._companion_fullscreen_active = True
            return True

        def _sync_mode_for_status(self, status: Any) -> None:
            """Follow authoritative voice/session state without renderer timing coupling."""

            self._set_companion_mode(
                should_use_companion_mode(
                    status.state,
                    bool(status.microphone_active),
                )
            )

        def set_companion_appearance(self, appearance: str) -> None:
            clean = "female" if str(appearance).lower() == "female" else "male"
            if hasattr(self.home, "companion_host"):
                self.home.companion_host.set_character(clean)
            if hasattr(self.home, "update_companion_selector"):
                self.home.update_companion_selector(clean)
            if hasattr(self, "settings_page") and hasattr(self.settings_page, "companion_appearance"):
                idx = self.settings_page.companion_appearance.findData(clean)
                if idx >= 0:
                    self.settings_page.companion_appearance.setCurrentIndex(idx)
            self._run(
                self.controller.update_settings,
                on_result=self._settings_result,
                companion_appearance=clean,
            )

        def select_page(self, name: str) -> None:
            page = self._pages.get(name)
            if page is None:
                return
            self.stack.setCurrentWidget(page)
            for button, (item, _icon) in zip(self._nav_buttons, NAV_ITEMS):
                button.setChecked(item == name)
            if hasattr(self.home, "companion_host"):
                if page is self.home:
                    self.home.companion_host.resume_rendering()
                else:
                    self.home.companion_host.pause_rendering()
            self._refresh_views()

        def _run(
            self,
            function: Callable[..., Any],
            *args: Any,
            on_result: Callable[[Any], None] | None = None,
            **kwargs: Any,
        ) -> None:
            worker = FunctionWorker(function, *args, **kwargs)
            self._workers.add(worker)
            if on_result is not None:
                worker.signals.result.connect(on_result)
            worker.signals.error.connect(self._show_error)
            worker.signals.finished.connect(lambda worker=worker: self._workers.discard(worker))
            QThreadPool.globalInstance().start(worker)

        def _show_error(self, message: str) -> None:
            self.header_message.setText(message)
            self.header_message.setVisible(True)
            self.header_state.setText("ERROR")
            self.header_state.setStyleSheet("color: #ffb4b4;")
            self._set_header_state_tone(AssistantState.ERROR, emergency_stopped=True)

        def _set_header_state_tone(
            self,
            state: AssistantState,
            *,
            emergency_stopped: bool = False,
            owner_session_active: bool = True,
            microphone_active: bool = False,
        ) -> None:
            if emergency_stopped or state in {
                AssistantState.ERROR,
                AssistantState.STOPPED,
            }:
                color = QColor("#FF7088")
            elif not owner_session_active or state in {
                AssistantState.LOCKED,
                AssistantState.AWAITING_CONFIRMATION,
            }:
                color = QColor("#F0B46E")
            elif microphone_active or state in {
                AssistantState.WAKE_DETECTED,
                AssistantState.AUTHENTICATING,
                AssistantState.LISTENING,
                AssistantState.TRANSCRIBING,
                AssistantState.THINKING,
                AssistantState.EXECUTING,
                AssistantState.SPEAKING,
            }:
                color = QColor("#66C8FF")
            else:
                color = QColor("#86EBCB")
            rim = color.lighter(145).name()
            self.header_state_dot.setStyleSheet(
                f"background: {color.name()}; border: 1px solid {rim}; border-radius: 4px;"
            )
            self.header_state_dot.setAccessibleDescription(
                state.value.replace("_", " ")
            )

        def _on_state_event(self, event: Any) -> None:
            """Apply real controller state on the Qt thread immediately."""

            self.home.set_state_event(event)
            self.companion_mode.set_state_event(event)
            state = getattr(event, "state", None)
            if isinstance(state, AssistantState):
                self.header_state.setText(state.value.upper().replace("_", " "))
                self.header_state.setStyleSheet("color: #ffb4b4;" if state == AssistantState.ERROR else "")
                self.header_message.setVisible(
                    state == AssistantState.ERROR or not bool(self._compact_shell)
                )
                self._set_header_state_tone(state)
            message = str(getattr(event, "message", "") or "").strip()
            if message:
                self.header_message.setText(message)
            try:
                self._sync_mode_for_status(self.controller.status())
                self._refresh_pending_views()
            except Exception:
                pass

        def _sync_audio_timer(self) -> None:
            """Run the high-rate presentation poll only while a mic is active."""

            if self._audio_poll_enabled:
                if not self._audio_timer.isActive():
                    self._audio_timer.start()
                return
            self._audio_timer.stop()

        def _refresh_orb_audio(self) -> None:
            """Feed only the latest real microphone scalar to both orb views."""

            if not self._audio_poll_enabled:
                return
            try:
                status = self.controller.status()
            except Exception:
                self._audio_poll_enabled = False
                self.home.orb.set_audio_level(0.0)
                if hasattr(self.home, "waveform"):
                    self.home.waveform.set_active(False)
                    self.home.waveform.set_level(0.0)
                self.companion_mode.orb.set_audio_level(0.0)
                self._sync_audio_timer()
                return
            self._audio_poll_enabled = bool(status.microphone_active)
            level = (
                getattr(status, "microphone_level", 0.0)
                if self._audio_poll_enabled
                else 0.0
            )
            self.home.orb.set_audio_level(level)
            if hasattr(self.home, "waveform"):
                self.home.waveform.set_active(self._audio_poll_enabled)
                self.home.waveform.set_level(level)
            self.companion_mode.orb.set_audio_level(level)
            self._sync_audio_timer()

        def _refresh_status(self) -> None:
            try:
                status = self.controller.status()
            except Exception as error:
                self._show_error(f"Status unavailable: {error}")
                return
            self._audio_poll_enabled = bool(status.microphone_active)
            if hasattr(self, "_audio_timer"):
                self._sync_audio_timer()
            state_text = status.state.value.upper().replace("_", " ")
            self.header_state.setText(state_text)
            self.header_state.setStyleSheet("color: #ffb4b4;" if status.state == AssistantState.ERROR else "")
            self.header_message.setText(status.message)
            self.header_message.setVisible(
                status.state == AssistantState.ERROR or not bool(self._compact_shell)
            )
            self._set_header_state_tone(
                status.state,
                emergency_stopped=bool(status.emergency_stopped),
                owner_session_active=bool(status.owner_session_active),
                microphone_active=bool(status.microphone_active),
            )
            if status.offline:
                self.header_offline.setText("LOCAL MODE")
                self.header_offline.set_tone("#67C8FF")
                self.header_api.setText("API DISABLED")
                self.header_api.set_tone("#777185")
            elif status.api_configured:
                self.header_offline.setText("HYBRID MODE")
                self.header_offline.set_tone("#9B83FF")
                self.header_api.setText("API CONFIGURED")
                self.header_api.set_tone("#82E7C7")
            else:
                self.header_offline.setText("LOCAL MODE")
                self.header_offline.set_tone("#67C8FF")
                self.header_api.setText("API UNAVAILABLE")
                self.header_api.set_tone("#C7A46A")
            self.header_session.setText(
                "SESSION ACTIVE" if status.owner_session_active else "SESSION EXPIRED"
            )
            self.header_session.set_tone(
                "#82E7C7" if status.owner_session_active else "#F1A36F"
            )
            self.reauthenticate_button.setEnabled(not status.owner_session_active)
            self.reauthenticate_button.setVisible(not status.owner_session_active)
            self.reset_stop_button.setVisible(status.emergency_stopped)
            self.emergency_stop_button.setEnabled(not status.emergency_stopped)
            self.home.refresh_status(status)
            self.companion_mode.refresh_status(status)
            self._sync_mode_for_status(status)
            try:
                self._refresh_pending_views()
            except Exception:
                # Status and emergency controls must remain live even if an
                # optional controller cannot enumerate confirmations.
                pass

        def _refresh_pending_views(self) -> None:
            pending = self.controller.pending_action()
            signature: object
            if pending is None:
                signature = None
            else:
                signature = (
                    pending.confirmation_id,
                    pending.action,
                    pending.target,
                    pending.effect,
                    pending.risk,
                    pending.expires_at,
                    json.dumps(
                        dict(pending.preview),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
            if signature == self._pending_view_signature:
                return
            self._pending_view_signature = signature
            self.confirmation_page.refresh(pending)
            self.companion_mode.refresh_pending(pending)

        def _refresh_views(self) -> None:
            try:
                self.conversation_page.refresh(self.controller.conversation(), self.controller.live_steps())
                self._refresh_pending_views()
                self.memory_page.refresh(self.controller.list_memories())
                self.users_page.refresh(self.controller.list_users())
                self.permissions_page.refresh(self.controller.list_permissions())
                self.activity_page.refresh(self.controller.audit_records())
                settings = self.controller.settings()
                self.settings_page.load(settings)
                self.home.refresh_settings(settings)
                self.companion_mode.refresh_settings(settings)
            except Exception as error:
                self._show_error(f"View refresh failed: {error}")
            self._refresh_status()

        def submit_text(self) -> None:
            text = self.home.input.text().strip()
            if not text:
                return
            self.home.input.clear()
            self._run(self.controller.submit_text, text, on_result=self._text_result)

        def _text_result(self, turn: ConversationTurn) -> None:
            self.home.show_reply(turn.message)
            self._refresh_views()

        def toggle_listening(self) -> None:
            self._run(self.controller.toggle_listening, on_result=lambda _result: self._refresh_views())

        def approve_pending(self) -> None:
            self._run(self.controller.approve_pending, on_result=lambda _result: self._refresh_views())

        def reject_pending(self) -> None:
            self._run(self.controller.reject_pending, on_result=lambda _result: self._refresh_views())

        @staticmethod
        def _memory_dialog(parent: QWidget, current: MemoryRecord | None = None) -> tuple[str, str, str, str, bool] | None:
            key, ok = QInputDialog.getText(parent, "Memory key", "Key:", text=current.key if current else "")
            if not ok:
                return None
            value, ok = QInputDialog.getText(parent, "Memory value", "Value:", text=current.value if current else "")
            if not ok:
                return None
            types = ["semantic", "preference", "relationship", "goal", "task"]
            type_index = types.index(current.memory_type) if current and current.memory_type in types else 0
            memory_type, ok = QInputDialog.getItem(parent, "Memory type", "Type:", types, current=type_index, editable=False)
            if not ok:
                return None
            sensitivities = ["public", "private", "sensitive"]
            sensitivity_index = sensitivities.index(current.sensitivity) if current and current.sensitivity in sensitivities else 1
            sensitivity, ok = QInputDialog.getItem(parent, "Sensitivity", "Sensitivity:", sensitivities, current=sensitivity_index, editable=False)
            consented = False
            if ok and sensitivity == "sensitive":
                consented = QMessageBox.question(
                    parent,
                    "Sensitive memory consent",
                    "Store this value as sensitive local memory? You can delete it later.",
                ) == QMessageBox.StandardButton.Yes
                if not consented:
                    return None
            return (key, value, memory_type, sensitivity, consented) if ok else None

        def add_memory(self) -> None:
            values = self._memory_dialog(self)
            if values is not None:
                self._run(self.controller.create_memory, *values, on_result=lambda _result: self._refresh_views())

        def edit_memory(self) -> None:
            memory_id = self.memory_page.selected_id()
            if not memory_id:
                self._show_error("Select a memory first")
                return
            current = next((item for item in self.controller.list_memories() if item.memory_id == memory_id), None)
            values = self._memory_dialog(self, current)
            if values is not None:
                _key, value, memory_type, sensitivity, consented = values
                self._run(self.controller.update_memory, memory_id, value, memory_type, sensitivity, consented, on_result=lambda _result: self._refresh_views())

        def delete_memory(self) -> None:
            memory_id = self.memory_page.selected_id()
            if not memory_id:
                self._show_error("Select a memory first")
                return
            answer = QMessageBox.question(self, "Delete memory", "Delete the selected memory? This cannot be undone.")
            if answer == QMessageBox.StandardButton.Yes:
                self._run(self.controller.delete_memory, memory_id, on_result=lambda _result: self._refresh_views())

        def export_memories(self) -> None:
            destination, _selected_filter = QFileDialog.getSaveFileName(
                self,
                "Export local memories",
                "asher-memories.json",
                "JSON files (*.json)",
            )
            destination = destination.strip()
            if not destination:
                return
            if not destination.casefold().endswith(".json"):
                destination += ".json"
            self._run(
                self.controller.export_memories,
                destination,
                on_result=self._memory_export_result,
            )

        def _memory_export_result(self, _destination: Any) -> None:
            self.header_message.setText(
                "Memory export completed to the selected local JSON file."
            )
            self._refresh_views()

        def enroll_user(self) -> None:
            name, ok = QInputDialog.getText(self, "Enroll user", "Display name:")
            if not ok or not name.strip():
                return
            role, ok = QInputDialog.getItem(self, "Enroll user", "Role:", ["owner", "trusted", "guest"], 1, False)
            if ok:
                self._run(self.controller.enroll_user, name, role, on_result=lambda _result: self._refresh_views())

        def capture_sample(self) -> None:
            user_id = self.users_page.selected_id()
            if user_id:
                selected = next((item for item in self.controller.list_users() if item.user_id == user_id), None)
                if selected is not None and selected.enrollment_status == "awaiting_recording_consent":
                    answer = QMessageBox.question(
                        self,
                        "Voice recording consent",
                        "Record and retain this user's voice sample in the private VoiceGuard directory? "
                        "You can revoke enrollment later.",
                    )
                    if answer != QMessageBox.StandardButton.Yes:
                        return
                    confirm = getattr(self.controller, "confirm_voice_recording_consent", None)
                    if not callable(confirm):
                        self._show_error("This controller has no consented VoiceGuard recorder")
                        return
                    self._run(
                        confirm,
                        user_id,
                        on_result=lambda _result: self._run(
                            self.controller.capture_voice_sample,
                            user_id,
                            on_result=lambda _captured: self._refresh_views(),
                        ),
                    )
                else:
                    self._run(self.controller.capture_voice_sample, user_id, on_result=lambda _result: self._refresh_views())
            else:
                self._show_error("Select a user first")

        def train_voiceguard(self) -> None:
            user_id = self.users_page.selected_id()
            if user_id:
                self._run(self.controller.train_voiceguard, user_id, on_result=lambda _result: self._refresh_views())
            else:
                self._show_error("Select a user first")

        def revoke_user(self) -> None:
            user_id = self.users_page.selected_id()
            if user_id and QMessageBox.question(self, "Revoke user", "Revoke this user and their enrollment?") == QMessageBox.StandardButton.Yes:
                self._run(self.controller.revoke_user, user_id, on_result=lambda _result: self._refresh_views())

        def permission_changed(self, item: QTableWidgetItem) -> None:
            if self.permissions_page._updating or item.column() != 3:
                return
            values = item.data(Qt.ItemDataRole.UserRole)
            if not values:
                return
            role, capability, actor_id = (*values, "") if len(values) == 2 else values
            allowed = item.checkState() == Qt.CheckState.Checked
            self._run(
                self.controller.set_permission,
                role,
                capability,
                allowed,
                actor_id=actor_id or None,
                on_result=lambda _result: self._refresh_views(),
            )

        def apply_settings(self) -> None:
            profile = self.settings_page.voice.currentData()
            mic = self.settings_page.mic.value()
            companion = self.settings_page.companion_appearance.currentData()
            changes = {
                "voice_profile": profile,
                "speech_speed": self.settings_page.speed.value(),
                "speech_style": self.settings_page.style.text().strip(),
                "offline_only": self.settings_page.offline.isChecked(),
                "api_enabled": self.settings_page.api.isChecked(),
                "microphone_index": None if mic < 0 else mic,
                "companion_appearance": companion,
            }
            if hasattr(self.home, "companion_host"):
                self.home.companion_host.set_character(companion)
            self._run(self.controller.update_settings, on_result=self._settings_result, **changes)

        def _settings_result(self, settings: DesktopSettings) -> None:
            self.settings_page.message.setText(f"Applied {settings.voice_profile}; future speech uses the new profile.")
            self._refresh_views()

        def run_diagnostics(self) -> None:
            self._run(self.controller.run_diagnostics, on_result=self.diagnostics_page.refresh)

        def reauthenticate_owner(self) -> None:
            self._run(
                self.controller.reauthenticate_owner,
                on_result=self._reauthentication_result,
            )

        def _reauthentication_result(self, _status: Any) -> None:
            self.header_message.setText(
                "Owner session re-authenticated with device credentials."
            )
            self._refresh_views()

        def emergency_stop(self) -> None:
            self._run(self.controller.emergency_stop, on_result=lambda _result: self._refresh_views())

        def reset_emergency_stop(self) -> None:
            self._run(self.controller.reset_emergency_stop, on_result=lambda _result: self._refresh_views())

        def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt callback name
            self._timer.stop()
            self._audio_timer.stop()
            self.companion_mode.orb.shutdown()
            if self._state_unsubscribe is not None:
                self._state_unsubscribe()
                self._state_unsubscribe = None
            close = getattr(self.controller, "close", None)
            if callable(close):
                close()
            # Give short controller jobs a chance to publish their final state;
            # no long-running provider is owned by the widget itself.
            QThreadPool.globalInstance().waitForDone(250)
            super().closeEvent(event)


AsherWindow = AsherMainWindow

__all__ = ["AsherMainWindow", "AsherWindow", "QT_AVAILABLE", "should_use_companion_mode"]
