import ctypes
from datetime import datetime
from pathlib import Path

import pyautogui


VOLUME_STEPS = 5

SCREENSHOT_FOLDER = (
    Path.home()
    / "Pictures"
    / "Asher Screenshots"
)


def increase_volume():
    pyautogui.press(
        "volumeup",
        presses=VOLUME_STEPS,
        interval=0.05
    )

    return "Volume increased."


def decrease_volume():
    pyautogui.press(
        "volumedown",
        presses=VOLUME_STEPS,
        interval=0.05
    )

    return "Volume decreased."


def toggle_mute():
    pyautogui.press("volumemute")

    return "Volume mute toggled."


def take_screenshot():
    SCREENSHOT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    filename = f"screenshot_{timestamp}.png"
    screenshot_path = SCREENSHOT_FOLDER / filename

    pyautogui.screenshot(
        str(screenshot_path)
    )

    return (
        f"Screenshot saved as {filename} "
        f"inside the Asher Screenshots folder."
    )


def lock_computer():
    ctypes.windll.user32.LockWorkStation()

    return "Computer locked."


if __name__ == "__main__":
    print("=========================")
    print("  SYSTEM CONTROLS TEST")
    print("=========================")

    print("1. Increase volume")
    print("2. Decrease volume")
    print("3. Toggle mute")
    print("4. Take screenshot")
    print("5. Lock computer")

    choice = input("\nChoose an action: ").strip()

    if choice == "1":
        print(increase_volume())

    elif choice == "2":
        print(decrease_volume())

    elif choice == "3":
        print(toggle_mute())

    elif choice == "4":
        print(take_screenshot())

    elif choice == "5":
        confirmation = input(
            "Type lock to confirm: "
        ).lower().strip()

        if confirmation == "lock":
            print(lock_computer())
        else:
            print("Lock cancelled.")

    else:
        print("Invalid option.")