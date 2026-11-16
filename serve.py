#!/usr/bin/env python3
"""Local server for the model pages.

Like `python3 -m http.server`, but tells the browser never to cache, so a plain
reload always shows the latest files. It also answers POST /ask, the questions
a viewer asks about a page (see ask.py); for that it needs the 'anthropic'
package and an API key:

    .venv/bin/python serve.py 5173
"""
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

MAX_BODY = 1_000_000  # bytes of one /ask request


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _hidden(self):
        # Nothing whose name starts with a dot is served: .venv, .claude, a stray .env.
        return any(part.startswith(".") for part in urlsplit(self.path).path.split("/") if part)

    def do_GET(self):
        if self._hidden():
            return self.send_error(404)
        super().do_GET()

    def do_HEAD(self):
        if self._hidden():
            return self.send_error(404)
        super().do_HEAD()

    def do_POST(self):
        if urlsplit(self.path).path != "/ask":
            return self.send_error(404)
        # Only this server's own pages may ask: a question spends the key's credits.
        # A JSON content type cannot be sent from another site without a preflight, which is never granted.
        origin = self.headers.get("Origin")
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self.send_error(415, "send JSON")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            return self.send_error(403, "questions are only accepted from this server's own pages")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if not 0 < length <= MAX_BODY:
            return self.send_error(413 if length > MAX_BODY else 400)
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError:
            return self.send_error(400, "not JSON")

        import ask  # here, so the pages are served even if this part cannot load

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.end_headers()
        events = ask.events(body)
        try:
            # One JSON object per line, sent as it happens; the response ends when the connection closes.
            for event in events:
                self.wfile.write((json.dumps(event) + "\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the viewer pressed Stop or left the page
        finally:
            events.close()  # ends the request to the model, so nothing more is spent


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5173
    print(f"Serving on http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
