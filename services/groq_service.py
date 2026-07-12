import os
import time
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
URL          = "https://api.groq.com/openai/v1/chat/completions"
logger       = logging.getLogger("maitrilearn")

MAX_TOKENS = {
    "ask":        600,
    "tutor":      800,
    "whiteboard": 3500,
    "terminal":   400,
    "default":    600,
}


def ask_ai(prompt: str, json_mode: bool = False, route: str = "default") -> str:
    """
    Call Groq LLaMA API with exponential backoff retry on rate limit (429).
    Retries up to 3 times with 2s → 4s → 8s delays.
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
        "max_tokens":  MAX_TOKENS.get(route, MAX_TOKENS["default"]),
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    # ── Retry with exponential backoff on 429 ────────────────────────────────
    max_retries = 3
    delays      = [2, 4, 8]  # seconds

    for attempt in range(max_retries + 1):
        t0       = time.time()
        response = requests.post(URL, headers=headers, json=payload, timeout=60)
        elapsed  = round((time.time() - t0) * 1000)

        if response.status_code == 429:
            # Rate limited — check Retry-After header first
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else delays[min(attempt, len(delays)-1)]

            if attempt < max_retries:
                logger.warning(f"[groq] 429 rate limited (attempt {attempt+1}/{max_retries}) "
                               f"— retrying in {wait}s route={route}")
                time.sleep(wait)
                continue
            else:
                logger.error(f"[groq] 429 rate limited after {max_retries} retries route={route}")
                raise ValueError("Rate limit exceeded. Please wait a moment and try again.")

        if response.status_code == 401:
            raise ValueError("Invalid GROQ_API_KEY — check Render environment variables")

        if response.status_code != 200:
            raise ValueError(f"Groq API error {response.status_code}: {response.text[:200]}")

        data    = response.json()
        choices = data.get("choices")
        if not choices:
            raise ValueError(f"Groq returned no choices: {data}")

        content = choices[0]["message"]["content"]
        logger.info(f"[groq] route={route} attempt={attempt+1} "
                    f"tokens={data.get('usage',{}).get('completion_tokens','?')} time={elapsed}ms")
        return content

    raise ValueError("AI service unavailable after retries. Please try again.")
