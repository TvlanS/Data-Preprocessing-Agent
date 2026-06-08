import os
from dotenv import load_dotenv
from pyprojroot import here

load_dotenv(here(".env"))

api = os.getenv("DEEPSEEK_API_KEY")
print(api)