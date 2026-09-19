from __future__ import annotations

import sys
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _Handler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.last_request = time.monotonic()

    def process_request(self, request: object, client_address: object) -> None:
        self.last_request = time.monotonic()
        super().process_request(request, client_address)


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    port = int(sys.argv[2])
    idle_timeout = float(sys.argv[3])
    handler = partial(_Handler, directory=str(root))
    server = _Server(("127.0.0.1", port), handler)
    server.timeout = 1.0
    while time.monotonic() - server.last_request < idle_timeout:
        server.handle_request()
    server.server_close()


if __name__ == "__main__":
    main()
