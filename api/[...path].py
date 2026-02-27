"""
Vercel serverless catch-all for FastAPI. Handles all /api/* routes.
Vercel expects a class "handler(BaseHTTPRequestHandler)" - we proxy to the FastAPI ASGI app.
"""
import asyncio
import os
import sys
from http.server import BaseHTTPRequestHandler

# Repo root = parent of api/ so "backend" package can be imported
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from backend.api.main import app


def _run_asgi(scope, body: bytes):
    """Run the ASGI app once and return (status, headers, body)."""
    status = 500
    headers: list[tuple[bytes, bytes]] = []
    response_body = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        nonlocal status, headers
        if message["type"] == "http.response.start":
            status = message.get("status", 500)
            headers = [(k.lower(), v) for k, v in message.get("headers", [])]
        elif message["type"] == "http.response.body":
            response_body.append(message.get("body", b""))

    async def run():
        await app(scope, receive, send)

    asyncio.run(run())
    return status, headers, b"".join(response_body)


class handler(BaseHTTPRequestHandler):
    def _respond(self, status: int, headers: list[tuple[bytes, bytes]], body: bytes):
        self.send_response(status)
        for name, value in headers:
            if name != b"content-length":
                self.send_header(name.decode("latin-1"), value.decode("latin-1"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle(self):
        # Read body for POST/PUT/PATCH
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b""
        # Build ASGI scope
        path = self.path.split("?")[0] if "?" in self.path else self.path
        query_string = self.path.split("?", 1)[1].encode() if "?" in self.path else b""
        scope = {
            "type": "http",
            "method": self.command,
            "path": path,
            "root_path": "",
            "scheme": "https",
            "query_string": query_string,
            "headers": [
                (k.encode("latin-1"), v.encode("latin-1"))
                for k, v in self.headers.items()
            ],
            "client": None,
            "server": None,
            "asgi": {"version": "3.0", "spec_version": "2.0"},
        }
        status, headers, body = _run_asgi(scope, body)
        self._respond(status, headers, body)

    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()

    def do_PUT(self):
        self._handle()

    def do_PATCH(self):
        self._handle()

    def do_DELETE(self):
        self._handle()

    def do_OPTIONS(self):
        self._handle()

    def log_message(self, format, *args):
        pass  # Suppress default request logging
