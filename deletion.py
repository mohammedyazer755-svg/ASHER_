delete_patterns = {
    "forget my favourite movie": "Favourite_Movie",
    "forget my favorite movie": "Favourite_Movie",
    "delete my favourite movie": "Favourite_Movie",
    "remove my favourite movie": "Favourite_Movie",

    "forget my favourite food": "Favourite_Food",
    "forget my favorite food": "Favourite_Food",
    "delete my favourite food": "Favourite_Food",
    "remove my favourite food": "Favourite_Food",

    "forget my favourite ide": "Favourite_IDE",
    "forget my favorite ide": "Favourite_IDE",
    "delete my favourite ide": "Favourite_IDE",
    "remove my favourite ide": "Favourite_IDE",

    "forget my favourite game": "Favourite_Game",
    "forget my favorite game": "Favourite_Game",
    "delete my favourite game": "Favourite_Game",
    "remove my favourite game": "Favourite_Game",

    "forget my dream company": "Dream_Company",
    "delete my dream company": "Dream_Company",
    "remove my dream company": "Dream_Company",

    "forget my hobby": "Hobby",
    "delete my hobby": "Hobby",
    "remove my hobby": "Hobby",

    "forget my college": "College",
    "delete my college": "College",
    "remove my college": "College",

    "forget my trusted person": "Trusted_Person",
    "delete my trusted person": "Trusted_Person",
    "remove my trusted person": "Trusted_Person",

    "forget my favourite colour": "Favourite_Colour",
    "forget my favorite color": "Favourite_Colour",
    "delete my favourite colour": "Favourite_Colour",
    "remove my favourite colour": "Favourite_Colour",

    "forget my favourite sport": "Favourite_Sport",
    "forget my favorite sport": "Favourite_Sport",
    "delete my favourite sport": "Favourite_Sport",
    "remove my favourite sport": "Favourite_Sport"
}


def detect_delete_intent(sentence):
    sentence = sentence.lower().strip()

    sentence = (
        sentence
        .replace("?", "")
        .replace(".", "")
        .replace("!", "")
    )

    return delete_patterns.get(sentence)