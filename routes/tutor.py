import logging
from flask import Blueprint, request
from services.groq_service import ask_ai
from utils.validator import validate_topic, looks_like_gibberish, ValidationError
from utils.limiter import limiter

tutor_bp = Blueprint("tutor", __name__)
logger = logging.getLogger("maitrilearn")


@tutor_bp.route("/tutor", methods=["POST"])
@limiter.limit("10 per minute")
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

    # SPEED FIX: shorter, focused prompt
    prompt = f"""Explain {topic} to a student in this exact format:

WHAT IT IS: (1 sentence)
HOW IT WORKS: (2-3 sentences)
REAL EXAMPLE: (1 concrete example)
KEY POINTS: (3 bullet points)

Be concise."""

    try:
        answer = ask_ai(prompt, route="tutor")
        logger.info(f"[tutor] topic={topic[:40]} len={len(answer)}")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"[tutor] Error: {e}")
        error_msg = str(e)
        if "Rate limit" in error_msg or "429" in error_msg:
            return {"error": "Too many requests — please wait 10 seconds and try again."}, 429
        return {"error": "AI service unavailable. Please try again in a moment."}, 503
