import logging
import time
from threading import Lock
from flask import Blueprint, request
from services.telegram_service import send_feedback
from utils.validator import validate_feedback, ValidationError

feedback_bp = Blueprint("feedback", __name__)
logger = logging.getLogger("maitrilearn")

# QA audit HIGH finding (H-04 / admin panel): feedback was only ever sent to
# Telegram — nothing was stored anywhere the app itself could read back, so
# there was no way to build a "view submitted feedback" admin screen at all.
# This is a small bounded in-memory log (last 200 submissions) that
# routes/admin.py's GET /admin/feedback reads from. NOTE: same caveat as the
# whiteboard lesson cache — per-worker-process memory, cleared on restart/
# deploy, not shared across gunicorn workers. Good enough to make the admin
# panel actually show recent feedback; a real production version would move
# this to Supabase alongside the RAG tables.
_recent_feedback = []
_recent_feedback_lock = Lock()
_MAX_FEEDBACK_ENTRIES = 200


def _log_feedback(text: str, rating=None):
    with _recent_feedback_lock:
        _recent_feedback.append({
            "text":      text,
            "rating":    rating,
            "timestamp": time.time(),
        })
        if len(_recent_feedback) > _MAX_FEEDBACK_ENTRIES:
            del _recent_feedback[0]


def get_recent_feedback(limit: int = 50):
    with _recent_feedback_lock:
        return list(reversed(_recent_feedback))[:limit]


@feedback_bp.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}

    try:
        # BUG FIX: accept both "feedback" and "message" field names
        raw_text = data.get("feedback") or data.get("message") or ""
        text = validate_feedback(raw_text)
    except ValidationError as e:
        return {"error": e.message, "field": e.field}, 400

    rating = data.get("rating")
    try:
        rating = int(rating) if rating is not None else None
    except (TypeError, ValueError):
        rating = None

    try:
        send_feedback(text)
        _log_feedback(text, rating)
        logger.info(f"[feedback] len={len(text)} rating={rating}")
        return {"success": True}
    except Exception as e:
        logger.error(f"[feedback] Error: {e}")
        return {"error": "Could not send feedback. Please try again."}, 503
