import logging
import re
from flask import Blueprint, request
from services.groq_service import ask_ai
from services.rag_service import search_chunks
from utils.validator import validate_topic, looks_like_gibberish, ValidationError
from utils.limiter import limiter

tutor_bp = Blueprint("tutor", __name__)
logger = logging.getLogger("maitrilearn")

# ── Question vs. topic detection ────────────────────────────────────────────
# The single input box on the AI Tutor UI is labelled "topic", but students
# type specific questions into it too ("What is useState?", "Solve 2x+5=13").
# QA audit CRITICAL finding: every input, question or not, was forced through
# a fixed "WHAT IT IS / HOW IT WORKS / EXAMPLE / KEY POINTS" topic-summary
# template, so a direct question never actually got answered — e.g. asking
# "What is useState?" returned a generic React description with no mention
# of useState at all. We now detect which case we're in and build a prompt
# that actually addresses what was typed either way.
_QUESTION_WORDS = (
    "what", "why", "how", "when", "where", "which", "who", "whom",
    "does", "do", "did", "is", "are", "was", "were", "can", "could",
    "should", "would", "will", "explain", "solve", "difference between",
)


def _looks_like_question(text: str) -> bool:
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    first_word = re.split(r"\s+", stripped.lower(), maxsplit=1)[0] if stripped else ""
    first_word = re.sub(r"[^a-z]", "", first_word)
    if first_word in _QUESTION_WORDS:
        return True
    if "difference between" in stripped.lower():
        return True
    return False


@tutor_bp.route("/tutor", methods=["POST"])
@limiter.limit("30 per minute", override_defaults=True)  # was 10/min — QA flagged as too strict for classroom use
def tutor():
    data = request.get_json(silent=True) or {}

    try:
        topic = validate_topic(data.get("topic", ""))
    except ValidationError as e:
        return {"error": e.message, "field": e.field}, 400

    if looks_like_gibberish(topic):
        logger.info(f"[tutor] Rejected gibberish input: {topic[:40]!r}")
        return {
            "error": "That doesn't look like a real topic. Please enter a subject "
                     "or question you'd like to learn about (e.g. 'Photosynthesis' or 'Docker').",
            "field": "topic"
        }, 400

    # ── Ground the answer in the student's uploaded notes when available ────
    rag_context = ""
    try:
        chunks = search_chunks(topic, topic=None, top_k=4, threshold=0.35)
        if chunks:
            contents = [c.get("content", "") if isinstance(c, dict) else c for c in chunks]
            rag_context = "\n\n---\n\n".join(c for c in contents if c)[:3000]
    except Exception as e:
        logger.warning(f"[tutor] RAG lookup error (non-fatal): {e}")

    is_question = _looks_like_question(topic)
    context_section = (
        f'\nSTUDENT\'S UPLOADED NOTES (use as primary source if relevant):\n"""\n{rag_context}\n"""\n'
        if rag_context else ""
    )

    if is_question:
        # Direct question — answer it specifically. Do NOT fall back to a
        # generic description of the wider subject.
        prompt = f"""A student asked this exact question: "{topic}"
{context_section}
Answer this SPECIFIC question directly and completely — do not describe the
general subject area instead of answering it. If it's a coding/math question,
show the actual working (code, steps, or calculation) that applies to THIS
question, not a generic Hello World or textbook blurb.

Format your answer like this:
DIRECT ANSWER: (1-2 sentences that directly answer "{topic}")
EXPLANATION: (2-4 sentences of specific reasoning/steps that apply to this exact question)
EXAMPLE: (a concrete example — real code, real numbers, or a real command — specific to "{topic}")
KEY POINTS: (2-3 bullet points specific to this question, not generic subject trivia)

Be concise and stay strictly on "{topic}"."""
    else:
        # General topic — a structured intro is appropriate here.
        prompt = f"""Explain the topic "{topic}" to a student in this exact format:
{context_section}
WHAT IT IS: (1 sentence)
HOW IT WORKS: (2-3 sentences)
REAL EXAMPLE: (1 concrete example specific to {topic})
KEY POINTS: (3 bullet points)

Be concise."""

    try:
        answer = ask_ai(prompt, route="tutor")
        logger.info(f"[tutor] topic={topic[:40]} question={is_question} rag={bool(rag_context)} len={len(answer)}")
        return {"answer": answer, "rag_used": bool(rag_context)}
    except Exception as e:
        logger.error(f"[tutor] Error: {e}")
        error_msg = str(e)
        if "Rate limit" in error_msg or "429" in error_msg:
            return {"error": "Too many requests — please wait 10 seconds and try again."}, 429
        return {"error": "AI service unavailable. Please try again in a moment."}, 503
