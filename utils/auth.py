"""
Lightweight shared-secret auth for admin/mutating endpoints
(/rag/ingest, /rag/delete). Not a full auth system — just a gate so these
endpoints aren't wide open to anonymous internet traffic, per the QA
security finding.

Set ADMIN_API_KEY in the Render environment to enable enforcement.
If it's not set, the endpoint stays open (so existing deployments and local
dev don't break) but a warning is logged on every request — this makes the
gap visible in logs rather than silently insecure forever.
"""
import os
import logging
from functools import wraps
from flask import request

logger = logging.getLogger("maitrilearn")


def require_admin_key(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        configured_key = os.getenv("ADMIN_API_KEY", "").strip()

        if not configured_key:
            logger.warning(
                f"[auth] ADMIN_API_KEY not set — {request.path} is running "
                "WITHOUT authentication. Set ADMIN_API_KEY in Render env vars."
            )
            return f(*args, **kwargs)

        provided_key = request.headers.get("X-Admin-Key", "")
        if provided_key != configured_key:
            logger.warning(f"[auth] Rejected unauthenticated request to {request.path} "
                           f"ip={request.headers.get('X-Forwarded-For', request.remote_addr)}")
            return {"error": "Unauthorized — missing or invalid X-Admin-Key header"}, 401

        return f(*args, **kwargs)
    return wrapper
