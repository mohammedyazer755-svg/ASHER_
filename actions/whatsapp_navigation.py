"""Small context-friendly navigation helpers for WhatsApp Desktop."""

import time

from pywinauto import Desktop, keyboard

from actions.app_launcher import open_app
from voice.text_normalizer import normalise_contact_name


def _find_whatsapp_window():
    for window in Desktop(backend="uia").windows():
        try:
            title = window.window_text().strip().lower()
            if "whatsapp" in title and window.is_visible():
                return window
        except Exception:
            continue

    return None


def search_whatsapp_contact(contact):
    contact = normalise_contact_name(contact)

    if not contact:
        return "Please tell me which WhatsApp contact to search for."

    try:
        window = _find_whatsapp_window()

        if window is None:
            open_app("WhatsApp")

            for _ in range(12):
                time.sleep(1)
                window = _find_whatsapp_window()
                if window is not None:
                    break

        if window is None:
            return "I could not find the WhatsApp window."

        window.set_focus()
        time.sleep(0.4)
        keyboard.send_keys("^f")
        time.sleep(0.5)
        keyboard.send_keys("^a{BACKSPACE}")
        keyboard.send_keys(contact, with_spaces=True, pause=0.04)

        return f"Searching WhatsApp for {contact}."

    except Exception as error:
        print(f"WhatsApp navigation error: {error}")
        return f"I could not search WhatsApp for {contact}."
