"""Small legacy conversation adapter.

Sensitive credentials are deliberately never returned. New conversations go
through ``asher.agent.controller.CompanionController``; this module remains a
safe compatibility surface for older imports.
"""

from responses import greetings, random_response, thanks
from utils import speak


def handle_conversations(user_input: str) -> bool:
    value = str(user_input).casefold().strip()
    if value in {"hi", "hello", "hey", "yo", "wassup"}:
        speak(random_response(greetings))
        return True
    if value in {"how are you", "how are you doing", "how's it going", "how are u"}:
        speak("I’m doing well and ready to help.")
        return True
    if value in {"who created you", "who is your creator", "creator"}:
        speak("I am the ASHER companion for the authenticated local owner.")
        return True
    if value in {"age", "what is my age", "how old am i"}:
        speak("That private profile value is available only through an authenticated memory view.")
        return True
    if value in {"name", "what is my name", "who am i"}:
        speak("Your profile is available only in the authenticated local memory view.")
        return True
    if value in {"version", "what is your version"}:
        speak("I’m running the current ASHER companion release.")
        return True
    if value in {"what is my password", "password", "what is my passcode", "passcode"}:
        speak("I never store or reveal passwords, passcodes, or PINs.")
        return True
    if value in {"yes", "yeah", "yup", "yep"}:
        speak("Understood.")
        return True
    if value in {"no", "nah", "nope"}:
        speak("Okay.")
        return True
    if value == "help":
        speak("I can chat, search approved local memory, and run authorized tools with previews.")
        return True
    if value in {"thanks", "thank you", "thx", "thank u"}:
        speak(random_response(thanks))
        return True
    if value == "i love you":
        speak("I care about helping you, and I’m here with you.")
        return True
    if value == "who is my mom":
        speak("That private relationship memory is available only after authenticated local retrieval.")
        return True
    return False
