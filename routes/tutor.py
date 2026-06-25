import logging
from flask import Blueprint, request
from services.groq_service import ask_ai
from utils.validator import validate_topic, ValidationError

tutor_bp = Blueprint("tutor", __name__)
logger = logging.getLogger("maitrilearn")


@tutor_bp.route("/tutor", methods=["POST"])
def tutor():
    data = request.get_json(silent=True) or {}

    try:
        topic = validate_topic(data.get("topic", ""))
    except ValidationError as e:
        return {"error": e.message, "field": e.field}, 400

    prompt = f"""Teach this topic simply to a student:

{topic}

Include:
- Clear explanation in simple language
- Real-world example
- Key points to remember"""

    try:
        answer = ask_ai(prompt)
        logger.info(f"[tutor] topic={topic[:40]} len={len(answer)}")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"[tutor] Error: {e}")
        return {"error": "AI service unavailable. Please try again."}, 503
