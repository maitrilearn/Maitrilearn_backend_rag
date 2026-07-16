"""
Multi-provider LLM service with automatic fallback.

Problem this solves: the backend previously called Groq exclusively. Once
Groq's free-tier quota is hit, every /ask, /tutor, /whiteboard/lesson and
/terminal/run request fails with a raw rate-limit error — even though the
platform has zero redundancy built in.

Fix: try providers in order (Groq -> Cerebras -> Gemini). Each is called
through a normalized OpenAI-compatible chat-completions call so the calling
code (routes/*.py) never has to know which provider actually answered.
A provider is skipped entirely if its API key isn't set in the environment,
so this works fine with just GROQ_API_KEY configured (current behavior) and
gets progressively more resilient as CEREBRAS_API_KEY / GEMINI_API_KEY are
added.

IMPORTANT — gunicorn worker timeout:
render.yaml runs gunicorn with --timeout 60. If ask_ai() blocks a worker for
close to (or over) 60s, Render/gunicorn kills the worker mid-request, which
looks like a random 500 on whatever request happened to be in flight (this
bit the original single-provider version too — see the comment history in
the old groq_service.py). Trying three providers back-to-back with full
timeouts each could easily exceed that. So this module enforces a hard
overall time budget (GLOBAL_BUDGET_SECONDS) across ALL providers and
retries combined — once the budget is spent, it fails fast with a normal
503/429-style error instead of risking the worker.
"""
import os
import time
import logging
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("maitrilearn")

MAX_TOKENS = {
    "ask":        600,
    "tutor":      800,
    "whiteboard": 3500,
    "terminal":   400,
    "default":    600,
}

# Keep the whole call chain safely under gunicorn's --timeout 60 (render.yaml),
# leaving buffer for request parsing / response serialization / network hops.
GLOBAL_BUDGET_SECONDS = 45
MIN_USEFUL_TIMEOUT    = 4   # don't even attempt a provider with less budget than this

# ── Provider registry ────────────────────────────────────────────────────────
# All three speak the OpenAI chat-completions schema (Gemini via its official
# OpenAI-compatibility endpoint), so one call path handles all of them.
def _providers():
    """Built lazily so env vars set after import (e.g. in tests) are honored."""
    return [
        {
            "name":    "groq",
            "api_key": os.getenv("GROQ_API_KEY", "").strip(),
            "url":     "https://api.groq.com/openai/v1/chat/completions",
            "model":   os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            "retries": 1,   # Groq is usually first/fastest — worth one quick retry on 429
        },
        {
            "name":    "cerebras",
            "api_key": os.getenv("CEREBRAS_API_KEY", "").strip(),
            "url":     "https://api.cerebras.ai/v1/chat/completions",
            "model":   os.getenv("CEREBRAS_MODEL", "llama-3.3-70b"),
            "retries": 0,
        },
        {
            "name":    "gemini",
            "api_key": os.getenv("GEMINI_API_KEY", "").strip(),
            "url":     "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "model":   os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            "retries": 0,
        },
    ]


class LLMError(Exception):
    """Raised for any provider-level failure (bad key, rate limit, bad response)."""
    def __init__(self, message, retryable=False):
        self.retryable = retryable
        super().__init__(message)


def _call_provider(provider: dict, prompt: str, max_tokens: int,
                    json_mode: bool, timeout_seconds: float) -> str:
    headers = {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       provider["model"],
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens":  max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    # connect timeout small & fixed; read timeout is whatever budget remains
    response = requests.post(
        provider["url"], headers=headers, json=payload,
        timeout=(5, max(timeout_seconds, MIN_USEFUL_TIMEOUT))
    )

    if response.status_code == 429:
        raise LLMError(f"{provider['name']} rate limited (429)", retryable=True)
    if response.status_code == 401:
        raise LLMError(f"{provider['name']} rejected the API key (401) — check env var", retryable=False)
    if response.status_code != 200:
        raise LLMError(f"{provider['name']} error {response.status_code}: {response.text[:200]}", retryable=False)

    data    = response.json()
    choices = data.get("choices")
    if not choices:
        raise LLMError(f"{provider['name']} returned no choices: {str(data)[:200]}", retryable=False)

    content = choices[0]["message"]["content"]
    if not content or not content.strip():
        raise LLMError(f"{provider['name']} returned empty content", retryable=False)
    return content


def ask_ai(prompt: str, json_mode: bool = False, route: str = "default") -> str:
    """
    Call the first available/working LLM provider. Tries Groq, then Cerebras,
    then Gemini — skipping any whose API key isn't configured. Bounded by
    GLOBAL_BUDGET_SECONDS so a single request can never hang the worker.
    """
    max_tokens = MAX_TOKENS.get(route, MAX_TOKENS["default"])
    providers  = [p for p in _providers() if p["api_key"]]

    if not providers:
        raise ValueError(
            "No LLM provider configured. Set at least one of GROQ_API_KEY, "
            "CEREBRAS_API_KEY, GEMINI_API_KEY in the environment."
        )

    start  = time.time()
    errors = []

    for provider in providers:
        attempts = provider.get("retries", 0) + 1
        for attempt in range(attempts):
            remaining = GLOBAL_BUDGET_SECONDS - (time.time() - start)
            if remaining < MIN_USEFUL_TIMEOUT:
                logger.warning(
                    f"[llm] time budget exhausted before trying {provider['name']} "
                    f"(route={route}) — failing fast rather than risking worker timeout"
                )
                raise ValueError(
                    "AI service is slow to respond right now. Please try again in a moment."
                )

            try:
                t0      = time.time()
                content = _call_provider(provider, prompt, max_tokens, json_mode, remaining)
                elapsed = round((time.time() - t0) * 1000)
                logger.info(
                    f"[llm] provider={provider['name']} route={route} "
                    f"attempt={attempt+1} time={elapsed}ms OK"
                )
                return content

            except LLMError as e:
                logger.warning(f"[llm] {provider['name']} failed (attempt {attempt+1}/{attempts}, route={route}): {e}")
                errors.append(str(e))
                remaining_after = GLOBAL_BUDGET_SECONDS - (time.time() - start)
                if e.retryable and attempt < attempts - 1 and remaining_after > MIN_USEFUL_TIMEOUT + 2:
                    time.sleep(2)
                    continue
                break  # move on to next provider

            except requests.exceptions.RequestException as e:
                logger.warning(f"[llm] {provider['name']} network error (route={route}): {e}")
                errors.append(f"{provider['name']} network error: {e}")
                break

    logger.error(f"[llm] all providers exhausted for route={route}: {errors}")
    combined = "; ".join(errors)
    if any("429" in e or "rate limited" in e for e in errors):
        raise ValueError("Too many requests — all configured AI providers are rate-limited right now. Please wait a moment and try again.")
    raise ValueError(f"AI service unavailable after trying all providers. Please try again in a moment. ({combined})")
