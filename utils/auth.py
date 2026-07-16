"""
Shared-secret auth for admin/mutating endpoints (/rag/ingest, /rag/delete).

FAIL-CLOSED BY DEFAULT. This directly addresses the QA audit's top finding:
these endpoints were reachable by anyone on the public internet with no key,
no JWT, no IP allowlist. Any request without a valid X-Admin-Key is now
rejected with 401/503 unless ADMIN_API_KEY is set in the environment AND
matches.

Set ADMIN_API_KEY in the Render environment (Dashboard -> your backend
service -> Environment) to enable these endpoints. Until it's set, they will
correctly refuse ALL requests, including legitimate ones -- this is
intentional: an admin write endpoint that's reachable with no key at all is
worse than one that's temporarily unusable.

Local development escape hatch: set ALLOW_UNSAFE_ADMIN=true in your local
.env if you need to hit these routes without configuring a key. This must
NEVER be set in a production/Render environment.
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
        allow_unsafe    = os.getenv("ALLOW_UNSAFE_ADMIN", "").strip().lower() == "true"

        if not configured_key:
            if allow_unsafe:
                logger.warning(
                    f"[auth] ADMIN_API_KEY not set - {request.path} allowed through "
                    "because ALLOW_UNSAFE_ADMIN=true (local dev only - never set this in production)."
                )
                return f(*args, **kwargs)
            logger.error(
                f"[auth] Rejected request to {request.path} - ADMIN_API_KEY is not "
                "configured on this server. Set it in the Render environment to enable this endpoint."
            )
            return {
                "error": "Admin endpoint is not configured. Contact the site administrator."
            }, 503

        provided_key = request.headers.get("X-Admin-Key", "")
        if provided_key != configured_key:
            logger.warning(
                f"[auth] Rejected unauthenticated request to {request.path} "
                f"ip={request.headers.get('X-Forwarded-For', request.remote_addr)}"
            )
            return {"error": "Unauthorized - missing or invalid X-Admin-Key header"}, 401

        return f(*args, **kwargs)
    return wrapper
