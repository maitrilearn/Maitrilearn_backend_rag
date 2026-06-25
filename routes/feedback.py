import logging
from flask import Blueprint, request
from services.telegram_service import send_feedback
from utils.validator import validate_feedback, ValidationError

feedback_bp = Blueprint("feedback", __name__)
logger = logging.getLogger("maitrilearn")


@feedback_bp.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}

    try:
        text = validate_feedback(data.get("feedback", ""))
    except ValidationError as e:
        return {"error": e.message, "field": e.field}, 400

    try:
        send_feedback(text)
        logger.info(f"[feedback] len={len(text)}")
        return {"success": True}
    except Exception as e:
        logger.error(f"[feedback] Error: {e}")
        return {"error": "Could not send feedback. Please try again."}, 503
