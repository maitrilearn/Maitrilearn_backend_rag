import os
import time
import logging
from datetime import datetime
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from routes.ask import ask_bp
from routes.tutor import tutor_bp
from routes.feedback import feedback_bp
from routes.whiteboard import whiteboard_bp
from routes.rag import rag_bp
from routes.terminal import terminal_bp
from utils.limiter import limiter

app = Flask(__name__)

# ── RATE LIMITING ───────────────────────────────────────────────────────────
# Was defined in utils/limiter.py but never wired in — every endpoint was
# unlimited. 200/day + 50/hour + 20/minute per IP as a sane default; the
# LLM-backed endpoints (/tutor, /ask, /whiteboard/lesson) additionally get a
# tighter per-route limit since those are the expensive/abusable ones.
limiter.init_app(app)

# ── CORS ──────────────────────────────────────────────────────────────────────
CORS(app, origins=[
    "https://maitrilearn.github.io",
    "https://maitrilearn.com",
    "https://www.maitrilearn.com",
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:5500",
    "http://localhost:9000",
    "null"
])

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("maitrilearn")


@app.before_request
def before_request():
    """Record request start time."""
    g.start_time = time.time()


@app.after_request
def after_request(response):
    """Log every request with timing."""
    # g.start_time may not be set if an earlier before_request hook (e.g.
    # Flask-Limiter aborting with 429) short-circuited before ours ran.
    # Falling back to 0ms rather than crashing — this was silently turning
    # every rate-limited request into a raw 500 instead of the intended
    # clean 429 (confirmed in production logs).
    start_time  = getattr(g, "start_time", None)
    duration_ms = round((time.time() - start_time) * 1000) if start_time else 0
    ip          = request.headers.get("X-Forwarded-For", request.remote_addr)
    origin      = request.headers.get("Origin", "-")

    # Skip logging health checks to reduce noise
    if request.path != "/health":
        logger.info(
            f"{request.method} {request.path} "
            f"→ {response.status_code} "
            f"({duration_ms}ms) "
            f"ip={ip} origin={origin}"
        )
    return response


# ── BLUEPRINTS ─────────────────────────────────────────────────────────────────
app.register_blueprint(ask_bp)
app.register_blueprint(tutor_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(whiteboard_bp)
app.register_blueprint(rag_bp)
app.register_blueprint(terminal_bp)


# ── HEALTH CHECK ──────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    """
    Health check endpoint for Render uptime monitoring.
    Returns 200 with service status.
    """
    return jsonify({
        "status":    "ok",
        "service":   "MaitriLearn Backend",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version":   "1.0.0"
    }), 200


# ── HOME ──────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({
        "message":   "MaitriLearn Backend Running ✅",
        "rag":       "enabled",
        "endpoints": ["/ask", "/tutor", "/feedback",
                      "/whiteboard/lesson", "/terminal/run",
                      "/rag/ingest", "/rag/search", "/rag/topics",
                      "/health"]
    }), 200


# ── ERROR HANDLERS ─────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    logger.warning(f"404 — {request.method} {request.path}")
    return jsonify({
        "error":   "Endpoint not found",
        "path":    request.path,
        "method":  request.method,
        "hint":    "Check the API documentation at /"
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    logger.warning(f"405 — {request.method} {request.path}")
    return jsonify({
        "error":  "Method not allowed",
        "method": request.method,
        "path":   request.path
    }), 405


@app.errorhandler(429)
def rate_limited(e):
    logger.warning(f"429 — {request.method} {request.path} rate limit exceeded")
    return jsonify({
        "error":   "Too many requests. Please slow down and try again shortly.",
        "message": str(getattr(e, "description", "Rate limit exceeded"))
    }), 429


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"500 — {request.method} {request.path} — {str(e)}")
    return jsonify({
        "error":   "Internal server error",
        "message": "Something went wrong on our end. Please try again."
    }), 500


@app.errorhandler(Exception)
def unhandled_exception(e):
    logger.error(
        f"Unhandled exception on {request.method} {request.path}: "
        f"{type(e).__name__}: {str(e)}"
    )
    return jsonify({
        "error":   "Unexpected error",
        "message": "Please try again or contact support."
    }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info(f"MaitriLearn Backend starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
