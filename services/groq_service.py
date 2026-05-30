import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
URL = "https://api.groq.com/openai/v1/chat/completions"


def ask_ai(prompt):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in environment variables")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model":    "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(URL, headers=headers, json=payload, timeout=30)

    if response.status_code == 401:
        raise ValueError("Invalid GROQ_API_KEY")
    if response.status_code != 200:
        raise ValueError(f"Groq API error {response.status_code}: {response.text[:200]}")

    data    = response.json()
    choices = data.get("choices")
    if not choices:
        raise ValueError(f"Groq returned no choices: {data}")

    return choices[0]["message"]["content"]
