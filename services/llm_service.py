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
            # llama-3.3-70b was deprecated by Cerebras (Feb 2026) and
            # llama3.1-8b was deprecated May 27 2026 — both now 404.
            # gpt-oss-120b is Cerebras's current recommended migration
            # target for both. If this 404s again in the logs, override
            # with CEREBRAS_MODEL without a redeploy — check
            # https://inference-docs.cerebras.ai/models/overview first.
            "model":     os.getenv("CEREBRAS_MODEL", "gpt-oss-120b"),
            "retries":   0,
            # gpt-oss-120b is a reasoning model: it spends part of max_tokens
            # on hidden "thinking" before the visible answer, so a small
            # max_tokens can produce a 200 OK with NO content at all (verified
            # against the live API — see conversation notes). Disable
            # reasoning so the full token budget goes to the answer.
            "reasoning": True,
        },
        {
            "name":    "gemini",
            "api_key": os.getenv("GEMINI_API_KEY", "").strip(),
            "url":     "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            # gemini-2.5-flash was retired for new/existing callers ahead
            # of its published Oct 2026 date (Google pulled it early —
            # see the Gemini API deprecations page). gemini-3-flash-preview
            # is the current official successor. Same deal: override with
            # GEMINI_MODEL if this one rots too, check
            # https://ai.google.dev/gemini-api/docs/deprecations first.
            "model":     os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
            "retries":   0,
            # Same reasoning-token issue as Cerebras — gemini-3-flash-preview
            # returned finish_reason="length" with 0 completion tokens on a
            # 10-token budget in live testing.
            "reasoning": True,
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
    if provider.get("reasoning"):
        # Ask the model to skip hidden "thinking" tokens and go straight to
        # the answer — otherwise max_tokens can be entirely consumed by
        # reasoning with nothing left for visible content (confirmed live:
        # both gpt-oss-120b and gemini-3-flash-preview did exactly this on a
        # 10-token budget). Not every provider honors this param, so it's
        # combined with the token padding below as a belt-and-suspenders fix.
        payload["reasoning_effort"] = "none"
        max_tokens = int(max_tokens * 1.5) + 200
        payload["max_tokens"] = max_tokens

    # connect timeout small & fixed; read timeout is whatever budget remains
    response = requests.post(
        provider["url"], headers=headers, json=payload,
        timeout=(5, max(timeout_seconds, MIN_USEFUL_TIMEOUT))
    )

    if response.status_code == 429:
        raise LLMError(f"{provider['name']} rate limited (429)", retryable=True)
    if response.status_code == 401:
        raise LLMError(f"{provider['name']} rejected the API key (401) — check env var", retryable=False)
    if response.status_code == 404:
        # Almost always means the model ID is stale/deprecated, not that the
        # provider is down. Providers rotate model names every few months
        # (this exact thing happened to both Cerebras and Gemini here) — flag
        # it distinctly so it doesn't get read as "just another 429-ish blip"
        # in the logs.
        raise LLMError(
            f"{provider['name']} model '{provider['model']}' not found (404) — "
            f"likely deprecated, override with {provider['name'].upper()}_MODEL env var",
            retryable=False,
        )
    if response.status_code != 200:
        raise LLMError(f"{provider['name']} error {response.status_code}: {response.text[:200]}", retryable=False)

    data    = response.json()
    choices = data.get("choices")
    if not choices:
        raise LLMError(f"{provider['name']} returned no choices: {str(data)[:200]}", retryable=False)

    content = choices[0].get("message", {}).get("content")
    if not content or not content.strip():
        finish_reason = choices[0].get("finish_reason", "?")
        raise LLMError(
            f"{provider['name']} returned empty/missing content "
            f"(finish_reason={finish_reason}) — likely ran out of tokens on "
            f"hidden reasoning before producing an answer",
            retryable=False,
        )
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
