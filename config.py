from dotenv import load_dotenv
import os

load_dotenv()
username = os.getenv("NAME")
age = os.getenv("AGE")
mom = os.getenv("MOTHER")