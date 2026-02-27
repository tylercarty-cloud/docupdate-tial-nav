"""
Vercel serverless catch-all for FastAPI. Handles all /api/* routes.
Vercel runs ASGI apps when it finds an `app` export (no Mangum).
"""
import os
import sys

# Repo root = parent of api/ so "backend" package can be imported
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from backend.api.main import app
