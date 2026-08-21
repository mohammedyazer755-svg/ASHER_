from memory import load_memory, remember
from patterns import memory_patterns

def search_memory(sentence):
    sentence = sentence.lower()
    memory = load_memory()
    for item in memory["memory"]:
        key = item["key"].lower()
        if key in sentence:
            return key, item["value"]
    return None, None

def learn_memory(sentence):
    sentence= sentence.lower()
    for pattern,key in memory_patterns.items():
            if sentence.startswith(pattern):
                 value =sentence.replace(pattern,"").strip()
                 remember(key,value)
                 return key , value
    return None, None
