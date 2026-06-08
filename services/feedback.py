from flask import Blueprint, request
from services.telegram_service import send_feedback

feedback_bp = Blueprint("feedback", __name__)


@feedback_bp.route("/feedback", methods=["POST"])
def feedback():
    data = request.json

    if not data or not data.get("feedback"):
        return {"error": "feedback text is required"}, 400

    try:
        send_feedback(data["feedback"])
        return {"success": True}
    except Exception as e:
        # BUG FIX: Don't crash if Telegram is misconfigured
        print(f"[feedback] Error: {e}")
        return {"error": "Could not send feedback."}, 503
