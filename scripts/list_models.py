"""Print the Gemini models your API key can use.

    python scripts/list_models.py

Pick a 'flash' model from the output and set GEMINI_MODEL in .env.
"""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not key:
    raise SystemExit("Set GEMINI_API_KEY in your .env file first.")

client = genai.Client(api_key=key)
for model in client.models.list():
    print(model.name)
