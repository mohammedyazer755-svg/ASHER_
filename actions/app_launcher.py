"""Discover, open and normally close Windows applications."""

import ctypes
import difflib
import json
import os
import subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
APPS_FILE = PROJECT_DIR / "data" / "apps.json"

ALIASES = {
    "browser": "Google Chrome",
    "chrome": "Google Chrome",
    "calculator": "Calculator",
    "calc": "Calculator",
    "cmd": "Command Prompt",
    "command prompt": "Command Prompt",
    "code": "Visual Studio Code",
    "vscode": "Visual Studio Code",
    "vs code": "Visual Studio Code",
    "whatsapp": "WhatsApp",
    "word": "Word",
    "excel": "Excel",
    "powerpoint": "PowerPoint",
}

SPECIAL_APPS = {
    "notepad": ["notepad.exe"],
    "calculator": ["calc.exe"],
    "google chrome": [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "--profile-directory=Default",
    ],
}

_catalog_cache = None


def _normalise(text):
    return " ".join(
        "".join(character.lower() if character.isalnum() else " " for character in text).split()
    )


def _load_start_apps():
    command = "Get-StartApps | Select-Object Name,AppID | ConvertTo-Json -Compress"

    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        check=False,
    )

    if result.returncode != 0 or not result.stdout.strip():
        return []

    data = json.loads(result.stdout)

    if isinstance(data, dict):
        data = [data]

    return [
        {
            "name": item.get("Name", "").strip(),
            "kind": "start_app",
            "command": item.get("AppID", "").strip(),
        }
        for item in data
        if item.get("Name") and item.get("AppID")
    ]


def _load_manual_apps():
    if not APPS_FILE.exists():
        return []

    try:
        with APPS_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(data, dict):
        return []

    return [
        {
            "name": str(name).strip(),
            "kind": "command",
            "command": command,
        }
        for name, command in data.items()
        if str(name).strip() and command
    ]


def _build_catalog(force=False):
    global _catalog_cache

    if _catalog_cache is not None and not force:
        return _catalog_cache

    items = _load_start_apps() + _load_manual_apps()
    unique = {}

    for item in items:
        key = _normalise(item["name"])
        if key and key not in unique:
            unique[key] = item

    _catalog_cache = list(unique.values())
    return _catalog_cache


def _resolved_query(app_name):
    query = _normalise(app_name)
    alias = ALIASES.get(query)
    return _normalise(alias) if alias else query


def _find_app(app_name):
    query = _resolved_query(app_name)
    catalog = _build_catalog()

    for item in catalog:
        if _normalise(item["name"]) == query:
            return item

    contained = [
        item
        for item in catalog
        if query in _normalise(item["name"])
        or _normalise(item["name"]) in query
    ]

    if contained:
        return min(contained, key=lambda item: len(item["name"]))

    best_item = None
    best_score = 0

    for item in catalog:
        score = difflib.SequenceMatcher(
            None,
            query,
            _normalise(item["name"]),
        ).ratio()

        if score > best_score:
            best_score = score
            best_item = item

    return best_item if best_score >= 0.62 else None


def _start_command(command):
    if isinstance(command, list):
        expanded = [os.path.expandvars(str(part)) for part in command]
        subprocess.Popen(expanded)
        return

    expanded = os.path.expandvars(str(command))
    subprocess.Popen(expanded, shell=True)


def open_app(app_name):
    app_name = app_name.strip()

    if not app_name:
        return "Please tell me which application to open."

    special_name = _resolved_query(app_name)
    special_command = SPECIAL_APPS.get(special_name)

    try:
        if special_command:
            executable = special_command[0]
            if executable.endswith(".exe") and "\\" in executable and not Path(executable).exists():
                special_command = None
            else:
                _start_command(special_command)
                return f"Opening {app_name}."

        item = _find_app(app_name)

        if not item:
            return f"I could not find {app_name} in your installed applications."

        if item["kind"] == "start_app":
            subprocess.Popen(
                ["explorer.exe", f"shell:AppsFolder\\{item['command']}"]
            )
        else:
            _start_command(item["command"])

        return f"Opening {item['name']}."

    except Exception as error:
        print(f"App opening error: {error}")
        return f"I could not open {app_name}."


def _visible_windows():
    user32 = ctypes.windll.user32
    windows = []

    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    @callback_type
    def callback(window_handle, _):
        if not user32.IsWindowVisible(window_handle):
            return True

        length = user32.GetWindowTextLengthW(window_handle)

        if length <= 0:
            return True

        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(window_handle, buffer, length + 1)
        title = buffer.value.strip()

        if title:
            windows.append((window_handle, title))

        return True

    user32.EnumWindows(callback, 0)
    return windows


def close_app(app_name):
    app_name = app_name.strip()

    if not app_name:
        return "Please tell me which application to close."

    query = _resolved_query(app_name)
    item = _find_app(app_name)
    names = {query, _normalise(app_name)}

    if item:
        names.add(_normalise(item["name"]))

    matched = []

    try:
        for handle, title in _visible_windows():
            normal_title = _normalise(title)

            if any(name and name in normal_title for name in names):
                matched.append((handle, title))

        if not matched:
            return f"I could not find an open {app_name} window."

        wm_close = 0x0010

        for handle, _ in matched:
            ctypes.windll.user32.PostMessageW(handle, wm_close, 0, 0)

        return f"Closing {app_name}."

    except Exception as error:
        print(f"App closing error: {error}")
        return f"I could not close {app_name}."


def refresh_app_catalog():
    try:
        catalog = _build_catalog(force=True)
        return f"I found {len(catalog)} installed applications."
    except Exception as error:
        print(f"App catalogue error: {error}")
        return "I could not refresh the installed application list."


if __name__ == "__main__":
    print("Dynamic app manager is ready.")
    print("Commands: open APP, close APP, refresh, exit")

    while True:
        command = input("You: ").strip()
        lowered = command.lower()

        if lowered in {"exit", "quit", "bye"}:
            break
        if lowered == "refresh":
            print(refresh_app_catalog())
        elif lowered.startswith("open "):
            print(open_app(command[5:]))
        elif lowered.startswith("close "):
            print(close_app(command[6:]))
        else:
            print("Use open APP, close APP, refresh, or exit.")
