from __future__ import annotations

import json
import mimetypes
from urllib.parse import unquote, urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .explanations import explain_sample
from .runs import (
    RunStore,
    predict_table,
    read_predictions,
    read_profile_for_run,
    read_saved_samples,
    read_summary,
)


OLLAMA = "http://127.0.0.1:11434"
MAX_UPLOAD = 10 * 1024 * 1024


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(detail)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, url):
        return None


def ollama_request(path: str, payload: dict | None = None, timeout: int = 5) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{OLLAMA}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def ollama_models() -> list[str]:
    try:
        return sorted(
            {
                name
                for model in ollama_request("/api/tags").get("models", [])
                if (name := str(model.get("name") or model.get("model") or ""))
                and int(model.get("size") or 0) > 0
                and "cloud" not in name.casefold()
            }
        )
    except (HTTPError, URLError, TimeoutError, ValueError, TypeError, KeyError):
        return []


def create_server(
    root: Path, port: int = 8765, static: Path | None = None
) -> ThreadingHTTPServer:
    store = RunStore(root)
    if static is None:
        raise ValueError("Provide the path to the interactive frontend assets.")
    static = static.resolve()
    if (
        not (static / "index.html").is_file()
        or not (static / "inference" / "index.html").is_file()
    ):
        raise FileNotFoundError("The original interactive frontend has not been built.")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def send(
            self, status: int, payload: object, content_type: str = "application/json"
        ) -> None:
            body = (
                json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
                if content_type == "application/json"
                else payload
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def serve_request(self) -> None:
            host = self.headers.get("Host", "").lower()
            bound_port = self.server.server_address[1]
            if host not in {f"127.0.0.1:{bound_port}", f"localhost:{bound_port}"}:
                raise ApiError(403, "Only local requests are allowed.")
            if self.command == "POST":
                origin = self.headers.get("Origin")
                if origin and origin.lower() not in {
                    f"http://127.0.0.1:{bound_port}",
                    f"http://localhost:{bound_port}",
                }:
                    raise ApiError(403, "Cross-origin requests are not allowed.")
                if self.headers.get_content_type() != "application/json":
                    raise ApiError(415, "Expected application/json.")
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise ApiError(400, "Invalid request length.") from exc
                if not 0 < length <= MAX_UPLOAD + 1024 * 1024:
                    raise ApiError(413, "Request exceeds the 10 MB upload limit.")
                try:
                    body = json.loads(self.rfile.read(length))
                except (ValueError, UnicodeError) as exc:
                    raise ApiError(400, "Invalid JSON request.") from exc
                if not isinstance(body, dict):
                    raise ApiError(400, "Expected a JSON object.")
            else:
                body = {}

            path = urlsplit(self.path).path
            if self.command == "GET" and not path.startswith("/api/"):
                requested = unquote(path)
                target = (static / requested.lstrip("/")).resolve()
                if target.is_dir():
                    target = target / "index.html"
                elif not target.suffix and (target / "index.html").is_file():
                    target = target / "index.html"
                if not target.is_relative_to(static) or not target.is_file():
                    raise ApiError(404, "Asset not found.")
                content_type = (
                    mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                )
                self.send(200, target.read_bytes(), content_type)
                return
            if self.command == "GET" and path == "/api/runs":
                self.send(200, [read_summary(run) for run in store.runs.values()])
                return
            if self.command == "GET" and path == "/api/chat/models":
                self.send(200, {"models": ollama_models()})
                return
            if self.command == "POST" and path == "/api/chat":
                model = body.get("model")
                question = body.get("question")
                context = body.get("context", "")
                if not isinstance(model, str) or model not in ollama_models():
                    raise ApiError(422, "Select a locally installed Ollama model.")
                if (
                    not isinstance(question, str)
                    or not 0 < len(question.strip()) <= 3000
                    or not isinstance(context, str)
                    or len(context) > 6000
                ):
                    raise ApiError(422, "Invalid chat request.")
                prompt = "You explain microbiome research model outputs. Avoid clinical recommendations. Distinguish evidence from interpretation. Never claim a diagnosis."
                try:
                    reply = ollama_request(
                        "/api/chat",
                        {
                            "model": model,
                            "stream": False,
                            "messages": [
                                {"role": "system", "content": prompt},
                                {
                                    "role": "user",
                                    "content": f"Results:\n{context}\n\nQuestion:\n{question}",
                                },
                            ],
                        },
                        timeout=120,
                    )
                    self.send(200, {"answer": reply["message"]["content"]})
                except (
                    HTTPError,
                    URLError,
                    TimeoutError,
                    ValueError,
                    KeyError,
                    TypeError,
                ) as exc:
                    raise ApiError(503, "Local Ollama chat is unavailable.") from exc
                return

            parts = path.strip("/").split("/")
            if len(parts) != 4 or parts[:2] != ["api", "runs"]:
                raise ApiError(404, "Endpoint not found.")
            try:
                run = store.get(parts[2])
            except KeyError as exc:
                raise ApiError(404, "Unknown experiment target.") from exc
            action = parts[3]
            if self.command == "GET" and action == "predictions":
                self.send(200, read_predictions(run))
                return
            if self.command == "GET" and action == "artifacts":
                directory = run.path / "explainability"
                artifacts = (
                    sorted(
                        str(p.relative_to(directory))
                        for p in directory.rglob("*")
                        if p.is_file() and p.suffix in {".parquet", ".tsv", ".json"}
                    )[:100]
                    if directory.is_dir()
                    else []
                )
                self.send(200, artifacts)
                return
            if self.command == "POST" and action in {"predict", "explain"}:
                source = body.get("source", "uploaded")
                if action == "predict" or source == "uploaded":
                    content = body.get("file")
                    if (
                        not isinstance(content, str)
                        or len(content.encode("utf-8")) > MAX_UPLOAD
                    ):
                        raise ApiError(413, "Provide a data file smaller than 10 MB.")
                    X, names, samples = read_profile_for_run(
                        run, content.encode("utf-8"), str(body.get("format", "wide"))
                    )
                    if len(samples) > 500:
                        raise ApiError(413, "Limit uploads to 500 samples.")
                elif source == "saved":
                    X, names, samples = read_saved_samples(run)
                else:
                    raise ApiError(422, "Invalid sample source.")
                strategy = str(body.get("strategy", "MPMA-B"))
                if action == "predict":
                    self.send(200, predict_table(run, X, names, samples, strategy))
                else:
                    self.send(
                        200,
                        explain_sample(
                            run,
                            X,
                            names,
                            samples,
                            str(body.get("sample_id", "")),
                            strategy,
                            str(body.get("method", "shap")),
                        ),
                    )
                return
            raise ApiError(404, "Endpoint not found.")

        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond()

        def respond(self):
            try:
                self.serve_request()
            except ApiError as exc:
                self.send(exc.status, {"detail": exc.detail})
            except (ValueError, FileNotFoundError) as exc:
                self.send(422, {"detail": str(exc)})
            except Exception:
                self.send(
                    500,
                    {"detail": "Interactive inference encountered an internal error."},
                )

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
