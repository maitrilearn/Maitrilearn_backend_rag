"""
QA audit HIGH finding (H-04): "Many admin endpoints missing (return 404)" —
GET /admin/topics, /admin/stats, /admin/feedback were all 404. admin.html
itself doesn't currently call these (it uses /rag/topics directly, which
works), but they're reasonable admin capabilities in their own right — in
particular there was previously no way to read back submitted feedback at
all. Implemented here rather than removed, per the audit's own suggested
fix ("implement... or remove UI references").

NOT implemented: /admin/users, /admin/usage, /terminal/session/start,
/terminal/session/end. There's no user-account system in this app at all
(no Supabase Auth, no session model — the terminal is stateless with the
client echoing cwd back, see routes/terminal.py), so building these would
mean returning fabricated data with nothing real behind it. Better to leave
them absent than fake.
"""
import os
import logging
import requests
from flask import Blueprint, request, jsonify
from routes.feedback import get_recent_feedback
from utils.auth import require_admin_key, set_session_cookie, clear_session_cookie
from utils.limiter import limiter

admin_bp = Blueprint("admin", __name__)
logger = logging.getLogger("maitrilearn")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


# ── Session login/logout (post-audit fix, finding H-04) ─────────────────────
# admin.html POSTs the raw key here ONCE per login instead of keeping it in
# localStorage forever. On success it gets back an HttpOnly session cookie
# and never has to touch (or store) the raw key again for the rest of the
# session. Rate-limited tightly since this is effectively a login endpoint
# and is the one place the raw key is still typed/sent.
@admin_bp.route("/admin/login", methods=["POST"])
@limiter.limit("10 per minute")
def admin_login():
    configured_key = os.getenv("ADMIN_API_KEY", "").strip()
    if not configured_key:
        return {"error": "Admin endpoint is not configured. Contact the site administrator."}, 503

    data          = request.get_json(silent=True) or {}
    provided_key  = (data.get("key") or "").strip()

    if provided_key != configured_key:
        logger.warning(
            f"[auth] Rejected admin login attempt "
            f"ip={request.headers.get('X-Forwarded-For', request.remote_addr)}"
        )
        return {"error": "Invalid admin key"}, 401

    resp = jsonify({"success": True})
    return set_session_cookie(resp)


@admin_bp.route("/admin/logout", methods=["POST"])
def admin_logout():
    resp = jsonify({"success": True})
    return clear_session_cookie(resp)


@admin_bp.route("/admin/session", methods=["GET"])
@require_admin_key
def admin_session_check():
    """Lets admin.html silently check 'am I still logged in?' on page load
    without prompting for the key every time (cookie lasts 12h)."""
    return {"authenticated": True}


@admin_bp.route("/admin/topics", methods=["GET"])
@require_admin_key
def admin_topics():
    """Like /rag/topics, but with a per-topic chunk count instead of just
    a flat topic list — what an admin dashboard actually wants to show."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"error": "Supabase not configured"}, 500
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    res = requests.get(
        f"{SUPABASE_URL}/rest/v1/documents?select=topic",
        headers=headers, timeout=10
    )
    if res.status_code != 200:
        return {"error": "Could not fetch topics"}, 500

    rows = res.json()
    counts = {}
    for r in rows:
        t = r.get("topic")
        if t:
            counts[t] = counts.get(t, 0) + 1

    topics = [{"topic": t, "chunks": c} for t, c in sorted(counts.items())]
    return {"topics": topics, "total_topics": len(topics), "total_chunks": len(rows)}


@admin_bp.route("/admin/stats", methods=["GET"])
@require_admin_key
def admin_stats():
    """High-level knowledge-base + feedback counters for an admin dashboard."""
    stats = {
        "topics":            None,
        "total_chunks":      None,
        "recent_feedback_count": len(get_recent_feedback(limit=200)),
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
    }

    if SUPABASE_URL and SUPABASE_KEY:
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
        try:
            res = requests.get(
                f"{SUPABASE_URL}/rest/v1/documents?select=topic",
                headers=headers, timeout=10
            )
            if res.status_code == 200:
                rows = res.json()
                stats["topics"] = len(set(r["topic"] for r in rows if r.get("topic")))
                stats["total_chunks"] = len(rows)
        except Exception as e:
            logger.warning(f"[admin/stats] Supabase lookup failed: {e}")

    return stats


@admin_bp.route("/admin/feedback", methods=["GET"])
@require_admin_key
def admin_feedback():
    """Recent feedback submissions. See routes/feedback.py for the storage
    caveat (in-memory, per-process, not durable across restarts)."""
    entries = get_recent_feedback(limit=50)
    return {"feedback": entries, "count": len(entries)}
