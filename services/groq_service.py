"""
Backward-compatible shim.

The multi-provider fallback logic (Groq -> Cerebras -> Gemini) now lives in
services/llm_service.py. This file is kept so existing imports across the
codebase (routes/ask.py, routes/tutor.py, routes/whiteboard.py,
routes/terminal.py, and the legacy services/*.py duplicates) don't need to
change.
"""
from services.llm_service import ask_ai  # noqa: F401
