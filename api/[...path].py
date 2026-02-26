"""
Vercel serverless catch-all for FastAPI. Handles all /api/* routes.
"""
import os
import sys

# Ensure backend package is importable (repo root = parent of api/)
_root = os.path.join(os.path.dirname(__file__), "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

from mangum import Mangum
from backend.api.main import app

# Mangum adapts ASGI (FastAPI) to Lambda-style (event, context). Export for Vercel.
# lifespan="off" so we use ensure_loaded() on first request instead.
handler = Mangum(app, lifespan="off")
