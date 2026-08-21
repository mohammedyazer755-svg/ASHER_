"""Responsive PySide6 desktop shell for ASHER.

The widgets intentionally contain presentation and event wiring only. All
state changes go through :class:`DesktopControllerProtocol`, and controller
calls are dispatched to ``QThreadPool`` workers so recognition/tool adapters
cannot freeze the GUI.
"""

from __future__ import annotations

import json
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
from asher.ui.workers import FunctionWorker, QT_WORKERS_AVAILABLE


try:
    from PySide6.QtCore import QAbstractAnimation, QEasingCurve, QPropertyAnimation, QThreadPool, QTimer, Qt
    from PySide6.QtGui import QColor, QFont, QIcon, QPalette
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
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


if not QT_AVAILABLE:

    class AsherMainWindow:  # type: ignore[no-redef]
        """Import-safe placeholder when optional Qt is not installed."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_qt()


else:

    APP_STYLE = """
    QWidget { color: #e8eef7; font-family: Segoe UI, Arial; font-size: 10pt; }
    QMainWindow, QDialog { background: #0b1220; }
    QFrame#sidebar { background: #101b2d; border-right: 1px solid #253654; }
    QFrame#header { background: #111d31; border-bottom: 1px solid #263958; }
    QFrame#card, QGroupBox { background: #121f34; border: 1px solid #263b5f; border-radius: 12px; }
    QGroupBox { margin-top: 12px; padding: 14px; font-weight: 600; }
    QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #9eb9dd; }
    QLabel#brand { color: #8ecbff; font-size: 18pt; font-weight: 700; }
    QLabel#pageTitle { color: #f5f8fc; font-size: 19pt; font-weight: 700; }
    QLabel#subtitle, QLabel#muted { color: #91a6c3; }
    QLabel#state { color: #a7f3d0; font-size: 12pt; font-weight: 700; }
    QLabel#statusChip { background: #1b3150; color: #bcd7f7; border-radius: 10px; padding: 5px 10px; }
    QLabel#warning { background: #4b3416; color: #ffd994; border: 1px solid #916721; border-radius: 8px; padding: 8px; }
    QLabel#danger { color: #ffb4b4; }
    QPushButton { background: #1d395f; border: 1px solid #315986; border-radius: 8px; padding: 8px 13px; }
    QPushButton:hover { background: #28527f; }
    QPushButton:pressed { background: #16304f; }
    QPushButton#primary { background: #147d93; border-color: #36b3c8; font-weight: 700; }
    QPushButton#primary:hover { background: #1c9bad; }
    QPushButton#dangerButton { background: #8d2638; border-color: #de6176; font-weight: 700; }
    QPushButton#dangerButton:hover { background: #b3344b; }
    QPushButton#nav { text-align: left; border: 0; background: transparent; padding: 10px 12px; color: #a9bdd8; }
    QPushButton#nav:hover, QPushButton#nav:checked { background: #1d3657; color: #ffffff; }
    QLineEdit, QPlainTextEdit, QTextBrowser, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget, QListWidget, QTreeWidget {
        background: #0e192b; border: 1px solid #2b456b; border-radius: 7px; padding: 6px; selection-background-color: #245c83;
    }
    QHeaderView::section { background: #1a2d49; color: #bcd7f7; padding: 6px; border: 0; }
    QTableWidget, QListWidget, QTreeWidget { gridline-color: #203450; }
    QProgressBar { background: #0e192b; border: 1px solid #2b456b; border-radius: 6px; text-align: center; }
    QProgressBar::chunk { background: #2ca7b8; border-radius: 5px; }
    QScrollBar:vertical { background: #0e192b; width: 10px; }
    QScrollBar::handle:vertical { background: #2b456b; border-radius: 5px; }
    """

    NAV_ITEMS = (
        ("Home", "⌂"),
        ("Conversation", "◌"),
        ("Confirmation", "✓"),
        ("Memory", "▣"),
        ("Users & VoiceGuard", "♙"),
        ("Permissions", "⚿"),
        ("Activity log", "☷"),
        ("Settings", "⚙"),
        ("Diagnostics", "◈"),
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


    class HomePage(_Page):
        def __init__(self, window: "AsherMainWindow") -> None:
            super().__init__("Welcome back", "A calm local-first companion. Use text input while the microphone adapter is being configured.")
            self.window = window
            card, layout = self.add_card()
            state_row = QHBoxLayout()
            self.state = _label("STANDBY", "state")
            self.state.setMinimumHeight(36)
            state_row.addWidget(self.state)
            state_row.addStretch()
            self.listen_button = QPushButton("Start listening")
            self.listen_button.setObjectName("primary")
            self.listen_button.clicked.connect(window.toggle_listening)
            state_row.addWidget(self.listen_button)
            layout.addLayout(state_row)
            self.status = _label("Ready for Hey Asher", "muted")
            layout.addWidget(self.status)

            input_card, input_layout = self.add_card("Text fallback")
            row = QHBoxLayout()
            self.input = QLineEdit()
            self.input.setPlaceholderText("Type a request for Asher…")
            self.input.returnPressed.connect(window.submit_text)
            row.addWidget(self.input, 1)
            send = QPushButton("Send")
            send.setObjectName("primary")
            send.clicked.connect(window.submit_text)
            row.addWidget(send)
            input_layout.addLayout(row)
            self.last_reply = _label("No request submitted yet.", "muted")
            input_layout.addWidget(self.last_reply)

            status_card, status_layout = self.add_card("Runtime status")
            self.offline = _label("Offline mode: checking…", "statusChip")
            self.api = _label("API: checking…", "statusChip")
            status_layout.addWidget(self.offline)
            status_layout.addWidget(self.api)
            disclosure = _label("Online speech is AI-generated; the offline Windows voice stays available.", "muted")
            status_layout.addWidget(disclosure)

        def refresh_status(self, status: Any) -> None:
            self.state.setText(status.state.value.upper().replace("_", " "))
            self.status.setText(status.message)
            self.listen_button.setText("Stop listening" if status.state == AssistantState.LISTENING else "Start listening")
            self.offline.setText("Offline mode: ON" if status.offline else "Offline mode: available fallback")
            self.api.setText("API: configured" if status.api_configured else "API: not configured")


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
            row.addWidget(add)
            row.addWidget(edit)
            row.addWidget(delete)
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
            self.save = QPushButton("Apply settings")
            self.save.setObjectName("primary")
            self.save.clicked.connect(window.apply_settings)
            privacy_layout.addWidget(self.save)
            self.message = _label("Changes apply to future speech without restarting ASHER.", "muted")
            privacy_layout.addWidget(self.message)

        def load(self, settings: DesktopSettings) -> None:
            index = self.voice.findData(settings.voice_profile)
            if index >= 0:
                self.voice.setCurrentIndex(index)
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
            self.setWindowTitle("Asher — authenticated personal companion")
            self.setMinimumSize(1120, 720)
            self.resize(1320, 820)
            self.setStyleSheet(APP_STYLE)
            self._build_shell()
            self._refresh_views()
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._refresh_status)
            self._timer.start(500)

        def _build_shell(self) -> None:
            central = QWidget()
            root = QHBoxLayout(central)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)
            self.setCentralWidget(central)

            sidebar = QFrame()
            sidebar.setObjectName("sidebar")
            sidebar.setFixedWidth(236)
            side = QVBoxLayout(sidebar)
            side.setContentsMargins(15, 24, 15, 18)
            side.setSpacing(4)
            side.addWidget(_label("ASHER", "brand"))
            side.addWidget(_label("Authenticated companion", "muted"))
            side.addSpacing(22)
            for name, icon in NAV_ITEMS:
                button = QPushButton(f"{icon}   {name}")
                button.setObjectName("nav")
                button.setCheckable(True)
                button.clicked.connect(lambda _checked=False, item=name: self.select_page(item))
                self._nav_buttons.append(button)
                side.addWidget(button)
            side.addStretch()
            side.addWidget(_label("Local memory • guarded actions • observable steps", "muted"))
            root.addWidget(sidebar)

            right = QVBoxLayout()
            right.setContentsMargins(0, 0, 0, 0)
            right.setSpacing(0)
            header = QFrame()
            header.setObjectName("header")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(22, 14, 22, 14)
            self.header_state = _label("STANDBY", "state")
            header_layout.addWidget(self.header_state)
            self.header_message = _label("Ready for Hey Asher", "muted")
            header_layout.addWidget(self.header_message, 1)
            self.header_offline = _label("Offline", "statusChip")
            header_layout.addWidget(self.header_offline)
            self.header_api = _label("API not configured", "statusChip")
            header_layout.addWidget(self.header_api)
            stop = QPushButton("EMERGENCY STOP")
            stop.setObjectName("dangerButton")
            stop.setMinimumHeight(38)
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
            root.addLayout(right, 1)
            self.select_page("Home")

        def select_page(self, name: str) -> None:
            page = self._pages.get(name)
            if page is None:
                return
            self.stack.setCurrentWidget(page)
            for button, (item, _icon) in zip(self._nav_buttons, NAV_ITEMS):
                button.setChecked(item == name)
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
            self.header_state.setText("ERROR")
            self.header_state.setStyleSheet("color: #ffb4b4;")

        def _refresh_status(self) -> None:
            try:
                status = self.controller.status()
            except Exception as error:
                self._show_error(f"Status unavailable: {error}")
                return
            state_text = status.state.value.upper().replace("_", " ")
            self.header_state.setText(state_text)
            self.header_state.setStyleSheet("color: #ffb4b4;" if status.state == AssistantState.ERROR else "")
            self.header_message.setText(status.message)
            self.header_offline.setText("OFFLINE MODE" if status.offline else "LOCAL + API")
            self.header_api.setText("API configured" if status.api_configured else "API not configured")
            self.reset_stop_button.setVisible(status.emergency_stopped)
            self.emergency_stop_button.setEnabled(not status.emergency_stopped)
            self.home.refresh_status(status)

        def _refresh_views(self) -> None:
            try:
                self.conversation_page.refresh(self.controller.conversation(), self.controller.live_steps())
                self.confirmation_page.refresh(self.controller.pending_action())
                self.memory_page.refresh(self.controller.list_memories())
                self.users_page.refresh(self.controller.list_users())
                self.permissions_page.refresh(self.controller.list_permissions())
                self.activity_page.refresh(self.controller.audit_records())
                self.settings_page.load(self.controller.settings())
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
            self.home.last_reply.setText(turn.message)
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
            changes = {
                "voice_profile": profile,
                "speech_speed": self.settings_page.speed.value(),
                "speech_style": self.settings_page.style.text().strip(),
                "offline_only": self.settings_page.offline.isChecked(),
                "api_enabled": self.settings_page.api.isChecked(),
                "microphone_index": None if mic < 0 else mic,
            }
            self._run(self.controller.update_settings, on_result=self._settings_result, **changes)

        def _settings_result(self, settings: DesktopSettings) -> None:
            self.settings_page.message.setText(f"Applied {settings.voice_profile}; future speech uses the new profile.")
            self._refresh_views()

        def run_diagnostics(self) -> None:
            self._run(self.controller.run_diagnostics, on_result=self.diagnostics_page.refresh)

        def emergency_stop(self) -> None:
            self._run(self.controller.emergency_stop, on_result=lambda _result: self._refresh_views())

        def reset_emergency_stop(self) -> None:
            self._run(self.controller.reset_emergency_stop, on_result=lambda _result: self._refresh_views())

        def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt callback name
            self._timer.stop()
            close = getattr(self.controller, "close", None)
            if callable(close):
                close()
            # Give short controller jobs a chance to publish their final state;
            # no long-running provider is owned by the widget itself.
            QThreadPool.globalInstance().waitForDone(250)
            super().closeEvent(event)


AsherWindow = AsherMainWindow

__all__ = ["AsherMainWindow", "AsherWindow", "QT_AVAILABLE"]
