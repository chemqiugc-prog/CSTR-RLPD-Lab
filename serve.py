#!/usr/bin/env python3
"""Tiny static file server for the CSTR-RLPD Lab web app (frontend/).
Sends Cache-Control: no-store so the browser always loads the latest files —
handy for development and harmless for the (static, dependency-free) app.
Stdlib only; no install needed.

    python3 serve.py [port]      # default 8010
"""
import http.server
import os
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8010
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"[CSTR-RLPD Lab] serving frontend/ at http://127.0.0.1:{PORT}")
    httpd.serve_forever()
