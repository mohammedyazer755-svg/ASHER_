from utils import speak
from config import username, age
from actions.app_launcher import open_app


def hello():
    speak("Hello!")


def show_name():
    speak(f"Your name is {username}")


def show_age():
    speak(f"You are {age} years old")


def show_version():
    speak("I'm running on version 0.2")


def help():
    speak("How can I help you, sir?")


def creator():
    speak(f"I was created by Batman {username}")


def thank():
    speak("You're welcome!")


def launch_notepad():
    response = open_app("notepad")
    speak(response)


def launch_calculator():
    response = open_app("calculator")
    speak(response)


def commands():
    speak(
        "Available commands are: hi, name, age, version, help, "
        "creator, thanks, open notepad, and open calculator."
    )


command_map = {
    "hi": hello,
    "hello": hello,
    "name": show_name,
    "age": show_age,
    "version": show_version,
    "help": help,
    "creator": creator,
    "thanks": thank,
    "commands": commands,
    "open notepad": launch_notepad,
    "open calculator": launch_calculator
}