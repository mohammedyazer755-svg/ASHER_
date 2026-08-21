"""Small QtConcurrent-style workers used by the desktop views."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from asher.core.redaction import redact_text


try:  # Keep ``import asher.ui`` safe on machines without Qt.
    from PySide6.QtCore import QObject, QRunnable, Signal, Slot

    QT_WORKERS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by dependency-free installs
    QT_WORKERS_AVAILABLE = False


if QT_WORKERS_AVAILABLE:

    class WorkerSignals(QObject):
        result = Signal(object)
        error = Signal(str)
        finished = Signal()


    class FunctionWorker(QRunnable):
        """Run a controller method away from the Qt GUI thread."""

        def __init__(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
            super().__init__()
            self.function = function
            self.args = args
            self.kwargs = kwargs
            self.signals = WorkerSignals()
            self.setAutoDelete(True)

        @Slot()
        def run(self) -> None:
            try:
                self.signals.result.emit(self.function(*self.args, **self.kwargs))
            except Exception as error:  # errors become UI state, never GUI crashes
                self.signals.error.emit(redact_text(f"{type(error).__name__}: {error}"))
            finally:
                self.signals.finished.emit()


else:

    class WorkerSignals:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "PySide6 is required for UI worker signals. Install PySide6 to use the desktop UI."
            )

    class FunctionWorker:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "PySide6 is required for background UI workers. Install PySide6 to use the desktop UI."
            )


TaskWorker = FunctionWorker

__all__ = ["FunctionWorker", "QT_WORKERS_AVAILABLE", "TaskWorker", "WorkerSignals"]
