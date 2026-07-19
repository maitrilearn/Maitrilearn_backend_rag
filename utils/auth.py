"""
Auth for admin/mutating endpoints (/rag/ingest, /rag/delete, /admin/*).

FAIL-CLOSED BY DEFAULT. This directly addresses the QA audit's top finding:
these endpoints were reachable by anyone on the public internet with no key,
no JWT, no IP allowlist. Any request without valid credentials is rejected
with 401/503 unless ADMIN_API_KEY is set in the environment AND matches.

Set ADMIN_API_KEY in the Render environment (Dashboard -> your backend
service -> Environment) to enable these endpoints. Until it's set, they will
correctly refuse ALL requests, including legitimate ones -- this is
intentional: an admin write endpoint that's reachable with no key at all is
worse than one that's temporarily unusable.

Local development escape hatch: set ALLOW_UNSAFE_ADMIN=true in your local
.env if you need to hit these routes without configuring a key. This must
NEVER be set in a production/Render environment.

── Session cookie (added post-audit, finding H-04) ──────────────────────────
Previously admin.html stored the raw admin key in localStorage and sent it
as X-Admin-Key on every request. Any XSS on the admin page (or a malicious
browser extension, or a shared/public machine) could read localStorage
directly and walk off with the permanent key. That raw header path is kept
here for non-browser callers (curl, scripts, CI) since there's no browser
storage to protect for those, but admin.html itself no longer touches the
raw key after the initial login:

  1. POST /admin/login with the raw key once -> server verifies it against
     ADMIN_API_KEY and, if it matches, issues a short-lived signed token in
     an HttpOnly, Secure, SameSite=None cookie. HttpOnly means JS (and
     therefore XSS) can never read it back out.
  2. Every subsequent admin request is sent with credentials included; the
     browser attaches the cookie automatically. require_admin_key() accepts
     either a valid session cookie OR a valid X-Admin-Key header.
  3. The token is itsdangerous-signed and expires (SESSION_MAX_AGE_SECONDS)
     -- even a stolen cookie is only useful for a limited window, unlike the
     raw key which worked forever until manually rotated.

This does not require a database: the token is stateless (its own signature
+ embedded timestamp is what's verified), so it works fine on Render's free
tier with no session store.
"""
import os
import time
import logging
from functools import wraps
from flask import request, make_response
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

logger = logging.getLogger("maitrilearn")

SESSION_COOKIE_NAME     = "maitrilearn_admin_session"
SESSION_MAX_AGE_SECONDS = 12 * 60 * 60  # 12 hours


def _serializer():
    # Salted with ADMIN_API_KEY itself so the signing secret rotates for
    # free whenever the admin key is rotated -- no separate SECRET_KEY to
    # manage/leak. If ADMIN_API_KEY isn't set there's nothing to sign
    # against anyway (endpoints are fail-closed below), so this is safe.
    secret = os.getenv("ADMIN_API_KEY", "").strip()
    return URLSafeTimedSerializer(secret_key=secret, salt="maitrilearn-admin-session")


def _issue_session_token() -> str:
    return _serializer().dumps({"admin": True, "iat": int(time.time())})


def _verify_session_token(token: str) -> bool:
    if not token:
        return False
    try:
        _serializer().loads(token, max_age=SESSION_MAX_AGE_SECONDS)
        return True
    except (BadSignature, SignatureExpired):
        return False


def set_session_cookie(response):
    """Attach a fresh signed session cookie to `response`. Called by
    /admin/login after the raw key has been verified."""
    token = _issue_session_token()
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,      # never readable from JS -- the XSS mitigation
        secure=True,         # HTTPS only (Render + GitHub Pages are both HTTPS)
        samesite="None",     # cookie must ride cross-site: GH Pages -> Render
        path="/",
    )
    return response


def clear_session_cookie(response):
    response.delete_cookie(SESSION_COOKIE_NAME, path="/", samesite="None", secure=True)
    return response


def require_admin_key(f):
    """Accepts EITHER:
       - a valid X-Admin-Key header matching ADMIN_API_KEY (scripts/CI/curl), or
       - a valid signed session cookie issued by /admin/login (browser UI).
    """
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
        if provided_key and provided_key == configured_key:
            return f(*args, **kwargs)

        session_token = request.cookies.get(SESSION_COOKIE_NAME, "")
        if _verify_session_token(session_token):
            return f(*args, **kwargs)

        logger.warning(
            f"[auth] Rejected unauthenticated request to {request.path} "
            f"ip={request.headers.get('X-Forwarded-For', request.remote_addr)}"
        )
        return {"error": "Unauthorized - missing or invalid admin credentials"}, 401

    return wrapper
