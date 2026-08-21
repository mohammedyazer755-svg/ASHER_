import os
import time

import pyautogui
import pyperclip


APP_OPEN_WAIT = 4
SEARCH_WAIT = 2
CHAT_OPEN_WAIT = 2


def paste_text(text):
    pyperclip.copy(text)
    pyautogui.hotkey("ctrl", "v")


def send_whatsapp_message(contact_name, message):
    contact_name = contact_name.strip()
    message = message.strip()

    if not contact_name:
        return "Contact name cannot be empty."

    if not message:
        return "Message cannot be empty."

    print("Opening WhatsApp Desktop...")

    # Open installed WhatsApp application
    os.startfile("whatsapp://send")

    time.sleep(APP_OPEN_WAIT)

    # Open the New Chat screen
    pyautogui.hotkey("ctrl", "n")
    time.sleep(1)

    # Search for the contact
    paste_text(contact_name)
    time.sleep(SEARCH_WAIT)

    # Select the first matching contact
    pyautogui.press("enter")
    time.sleep(CHAT_OPEN_WAIT)

    # Type and send the message
    paste_text(message)
    time.sleep(1)
    pyautogui.press("enter")

    return f"Message sent to {contact_name}."


if __name__ == "__main__":
    print("=========================")
    print(" WHATSAPP DESKTOP TEST")
    print("=========================")

    contact = input("Contact name: ").strip()
    message = input("Message: ").strip()

    print("\nPlease verify:")
    print(f"Contact: {contact}")
    print(f"Message: {message}")

    confirmation = input("\nType yes to send: ").lower().strip()

    if confirmation == "yes":
        try:
            result = send_whatsapp_message(contact, message)
            print(result)

        except Exception as error:
            print(f"WhatsApp error: {error}")

    else:
        print("Message cancelled.")