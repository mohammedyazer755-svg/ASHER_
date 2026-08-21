"""Build the default registry without importing hardware or model providers."""

from __future__ import annotations

import os
from pathlib import Path

from asher.config import AsherConfig
from asher.memory.retrieval import MemoryRetriever
from asher.memory.store import MemoryStore
from asher.security.audit import AuditLog
from asher.security.confirmations import ConfirmationStore
from asher.security.policy import PolicyEngine
from asher.storage import Database
from asher.tools.browser import BrowserTools
from asher.tools.computer import ComputerTools
from asher.tools.files import FileTools
from asher.tools.memory import MemoryTools
from asher.tools.registry import ToolRegistry
from asher.tools.system import SystemTools
from asher.tools.whatsapp import WhatsAppTools
from asher.tools.windows import AppCatalog, WindowsAppTools


def build_registry(config: AsherConfig, database: Database | None = None, *, contact_resolver=None) -> ToolRegistry:
    database = database or Database(config.runtime.database)
    audit = AuditLog(config.runtime.audit_log)
    confirmations = ConfirmationStore(config.confirmation_seconds)
    registry = ToolRegistry(policy=PolicyEngine(), confirmations=confirmations, audit=audit)

    project_root = Path(__file__).resolve().parents[2]
    catalog = AppCatalog(project_root / "data" / "apps.json")
    for definition in WindowsAppTools(catalog).definitions():
        registry.register(definition)
    for definition in BrowserTools().definitions():
        registry.register(definition)
    for definition in SystemTools(config.runtime.screenshots).definitions():
        registry.register(definition)

    roots = [config.runtime.root]
    configured_roots = os.getenv("ASHER_ALLOWED_ROOTS", "").strip()
    if configured_roots:
        roots.extend(Path(item.strip()) for item in configured_roots.split(os.pathsep) if item.strip())
    protected_runtime = [
        config.runtime.database,
        Path(str(config.runtime.database) + "-wal"),
        Path(str(config.runtime.database) + "-shm"),
        config.runtime.audit_log,
    ]
    for definition in FileTools(roots, protected_paths=protected_runtime).definitions():
        registry.register(definition)
    for definition in ComputerTools().definitions():
        registry.register(definition)

    memory_store = MemoryStore(database)
    for definition in MemoryTools(memory_store, MemoryRetriever(memory_store)).definitions():
        registry.register(definition)
    for definition in WhatsAppTools(contact_resolver=contact_resolver).definitions():
        registry.register(definition)
    return registry
