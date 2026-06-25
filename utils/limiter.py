from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Shared limiter instance — import this in app.py
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",    # use Redis URI in production
    strategy="fixed-window"
)
