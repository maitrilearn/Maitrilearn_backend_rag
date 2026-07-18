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
from flask import Blueprint
from routes.feedback import get_recent_feedback
from utils.auth import require_admin_key

admin_bp = Blueprint("admin", __name__)
logger = logging.getLogger("maitrilearn")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


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
