import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
URL = "https://api.groq.com/openai/v1/chat/completions"


def ask_ai(prompt, json_mode=False):
    """
    Call Groq LLaMA API.
    json_mode=True only for whiteboard/RAG which need JSON output.
    ask/tutor use plain text — never json_mode.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in environment variables")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json"
    }

    payload = {
        "model":       "llama-3.1-8b-instant",
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }

    # Only add json response format for whiteboard route
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    response = requests.post(URL, headers=headers, json=payload, timeout=60)

    if response.status_code == 401:
        raise ValueError("Invalid GROQ_API_KEY — check Render environment variables")
    if response.status_code != 200:
        raise ValueError(f"Groq API error {response.status_code}: {response.text[:200]}")

    data    = response.json()
    choices = data.get("choices")
    if not choices:
        raise ValueError(f"Groq returned no choices: {data}")

    print("[groq] Response received successfully")
    return choices[0]["message"]["content"]
