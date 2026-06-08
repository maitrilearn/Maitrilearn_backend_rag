from flask import Blueprint, request
from services.groq_service import ask_ai

# BUG FIX: Now uses shared groq_service instead of duplicating HTTP call
tutor_bp = Blueprint("tutor", __name__)


@tutor_bp.route("/tutor", methods=["POST"])
def tutor():
    data = request.json

    if not data or not data.get("topic"):
        return {"error": "topic is required"}, 400

    topic = data["topic"]

    prompt = f"""Teach this topic simply:

{topic}

Include:
- explanation
- examples
- key points"""

    try:
        answer = ask_ai(prompt)
        return {"answer": answer}
    except Exception as e:
        # BUG FIX: Return proper error response instead of crashing with KeyError
        print(f"[tutor] Error: {e}")
        return {"error": "AI service unavailable. Please try again."}, 503
