import logging
from flask import Blueprint, request
from services.groq_service import ask_ai
from utils.validator import validate_question, validate_text, ValidationError

ask_bp = Blueprint("ask", __name__)
logger = logging.getLogger("maitrilearn")


@ask_bp.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}

    try:
        question = validate_question(data.get("question", ""))
        subject  = validate_text(data.get("subject", "General"),
                                 field="subject", min_len=0, max_len=100) \
                   if data.get("subject") else "General"
        topic    = validate_text(data.get("topic", ""),
                                 field="topic", min_len=0, max_len=200) \
                   if data.get("topic") else ""
    except ValidationError as e:
        return {"error": e.message, "field": e.field}, 400

    # SPEED FIX: concise prompt — no verbose instructions
    if subject and subject != "General" or topic:
        prompt = f"Subject: {subject}\nTopic: {topic}\nQ: {question}\n\nAnswer clearly in 2-4 sentences."
    else:
        prompt = f"Q: {question}\n\nAnswer clearly in 2-4 sentences."

    try:
        answer = ask_ai(prompt, route="ask")
        logger.info(f"[ask] subject={subject} len={len(answer)}")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"[ask] Error: {e}")
        error_msg = str(e)
        if "Rate limit" in error_msg or "429" in error_msg:
            return {"error": "Too many requests — please wait 10 seconds and try again."}, 429
        return {"error": "AI service unavailable. Please try again in a moment."}, 503
