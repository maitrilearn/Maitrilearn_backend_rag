from flask import Blueprint, request
from services.groq_service import ask_ai

# BUG FIX: Now uses shared groq_service instead of duplicating HTTP call
ask_bp = Blueprint("ask", __name__)


@ask_bp.route("/ask", methods=["POST"])
def ask():
    data = request.json

    if not data or not data.get("question"):
        return {"error": "question is required"}, 400

    prompt = data["question"]
    subject = data.get("subject", "")
    topic = data.get("topic", "")

    if subject or topic:
        prompt = f"Subject: {subject}\nTopic: {topic}\n\nQuestion: {prompt}"

    try:
        answer = ask_ai(prompt)
        return {"answer": answer}
    except Exception as e:
        # BUG FIX: Return proper error response instead of crashing with KeyError
        print(f"[ask] Error: {e}")
        return {"error": "AI service unavailable. Please try again."}, 503
