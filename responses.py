import random
from config import username

greetings=[ 
    f"Hello!{username}",
    f"Hi{username}",
    f"Hey{username}",
    f"Good to see you, {username}",
    f"Welcome Back {username}"
]

thanks =[
    "You're welcome",
    "Happy to help",
    "Anytime",
    "My pleasure!",
    "No Problem!"
]

goodbye = [
    "Bye",
    "See you later",
    "Take care",
    "Have a great day",
    "Catch you later",
    "See you soon",
]
unknown = [
    "I didn't understand that.",
    "Could you say it another way.",
    "I'm still learning that.",
    "I'm not sure what you mean.",
    "I haven't learnt that yet."
]

def random_response(response_list):
    return random.choice(response_list)