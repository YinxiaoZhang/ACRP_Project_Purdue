from __future__ import annotations

from base64 import b64decode
from io import BytesIO
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any

from PIL import Image

from .service import ParkingVisionService


class ParkingVisionAPI:
    def __init__(self, default_config_path: str | Path | None = None) -> None:
        self._service = ParkingVisionService()
        self._default_config_path = Path(default_config_path).expanduser().resolve() if default_config_path else None

    def analyze_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        config_path = payload.get("config_path")
        if config_path is None:
            if self._default_config_path is None:
                raise ValueError("Request must include config_path, or start the server with --default-config.")
            config_path = self._default_config_path

        backend = str(payload.get("backend", "heuristic"))
        config = self._service.load_config(config_path)

        if payload.get("image_path"):
            with Image.open(Path(payload["image_path"]).expanduser().resolve()) as image:
                return self._service.analyze(image=image, config=config, backend=backend)

        if payload.get("image_base64"):
            raw_bytes = b64decode(payload["image_base64"])
            with Image.open(BytesIO(raw_bytes)) as image:
                return self._service.analyze(image=image, config=config, backend=backend)

        raise ValueError("Request must include either image_path or image_base64.")


def create_handler(app: ParkingVisionAPI) -> type[BaseHTTPRequestHandler]:
    class RequestHandler(BaseHTTPRequestHandler):
        def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._write_json(HTTPStatus.OK, {"status": "ok"})
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/analyze":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
                result = app.analyze_request(payload)
            except Exception as exc:  # noqa: BLE001
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            self._write_json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return RequestHandler


def serve(host: str, port: int, default_config_path: str | Path | None = None) -> None:
    app = ParkingVisionAPI(default_config_path=default_config_path)
    server = ThreadingHTTPServer((host, port), create_handler(app))
    print(f"Parking vision API listening on http://{host}:{port}")
    server.serve_forever()
