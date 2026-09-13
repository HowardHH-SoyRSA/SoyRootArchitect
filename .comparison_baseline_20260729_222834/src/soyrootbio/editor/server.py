from __future__ import annotations

import argparse
import hmac
import ipaddress
import os
from pathlib import Path
import secrets
import threading
from urllib.parse import urlsplit
import webbrowser

from flask import Flask, jsonify, make_response, request, send_file, send_from_directory

from .session import (
    EditorRevisionConflict,
    EditorSession,
    EditorValidationError,
)


def create_editor_app(
    output_dir: str | Path,
    *,
    session_dir: str | Path | None = None,
    static_dir: str | Path | None = None,
) -> Flask:
    session = EditorSession(output_dir, session_dir=session_dir)
    resolved_static = _resolve_static_dir(static_dir)
    app = Flask(
        "soyrootbio-editor",
        static_folder=None,
    )
    app.config["EDITOR_SESSION"] = session
    app.config["EDITOR_STATIC_DIR"] = resolved_static
    app.config["EDITOR_SESSION_TOKEN"] = secrets.token_urlsafe(32)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

    @app.after_request
    def add_local_security_headers(response):
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "connect-src 'self'; worker-src 'self' blob:; "
            "object-src 'none'; frame-ancestors 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.before_request
    def protect_local_api():
        hostname = urlsplit(f"//{request.host}").hostname or ""
        if not _is_loopback_host(hostname):
            return jsonify({"error": "Non-loopback host rejected.", "kind": "security"}), 403
        if not request.path.startswith("/api/"):
            return None
        expected = app.config["EDITOR_SESSION_TOKEN"]
        supplied = request.cookies.get("soyrootbio_editor_token", "")
        if not supplied or not hmac.compare_digest(supplied, expected):
            return jsonify({"error": "Editor session token is missing.", "kind": "security"}), 403
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
            return jsonify({"error": "Cross-origin editor request rejected.", "kind": "security"}), 403
        return None

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "state": session.public_state()})

    @app.get("/api/state")
    def state():
        return jsonify(session.public_state())

    @app.get("/api/mesh")
    def mesh():
        return send_file(
            session.mesh.path,
            mimetype="application/octet-stream",
            conditional=True,
            etag=True,
        )

    @app.get("/api/mesh-labels")
    def mesh_labels():
        payload, revision = session.labels_snapshot()
        return app.response_class(
            payload,
            mimetype="application/octet-stream",
            headers={
                "X-Vertex-Count": str(session.mesh.vertex_count),
                "X-Label-Revision": str(revision),
            },
        )

    @app.get("/api/point-patches/<patch_id>/indices")
    def point_patch_indices(patch_id: str):
        raw_revision = request.args.get("revision")
        if raw_revision is None:
            raise EditorValidationError("Point-patch revision is required.")
        try:
            expected_revision = int(raw_revision)
        except ValueError as exc:
            raise EditorValidationError(
                "Point-patch revision must be an integer."
            ) from exc
        payload, count, revision = session.point_patch_indices_snapshot(
            patch_id,
            expected_revision=expected_revision,
        )
        return app.response_class(
            payload,
            mimetype="application/octet-stream",
            headers={
                "X-Point-Count": str(count),
                "X-Label-Revision": str(revision),
            },
        )

    @app.get("/api/operations")
    def operations():
        content = session.operation_log_text()
        return app.response_class(content, mimetype="application/x-ndjson")

    @app.post("/api/operations")
    def apply_operation():
        payload = request.get_json(force=True, silent=False) or {}
        result = session.apply_operation(
            str(payload.get("type", "")),
            dict(payload.get("arguments") or {}),
        )
        return jsonify(result)

    @app.post("/api/undo")
    def undo():
        return jsonify(session.undo())

    @app.post("/api/redo")
    def redo():
        return jsonify(session.redo())

    @app.post("/api/export")
    def export():
        payload = request.get_json(silent=True) or {}
        target = payload.get("target_dir")
        if target is not None:
            resolved_target = Path(str(target)).resolve()
            if (
                resolved_target != session.session_dir
                and session.session_dir not in resolved_target.parents
            ):
                raise EditorValidationError(
                    "Exports are restricted to the editor session directory."
                )
            target = resolved_target
        export_dir = session.export_materialised(target)
        return jsonify({"export_dir": str(export_dir)})

    @app.errorhandler(EditorValidationError)
    @app.errorhandler(ValueError)
    def validation_error(error):
        return jsonify({"error": str(error), "kind": "validation"}), 400

    @app.errorhandler(EditorRevisionConflict)
    def revision_conflict(error):
        return (
            jsonify(
                {
                    "error": str(error),
                    "kind": "revision_conflict",
                    "current_revision": session.label_revision,
                }
            ),
            409,
        )

    @app.errorhandler(FileNotFoundError)
    def missing_file(error):
        return jsonify({"error": str(error), "kind": "missing_file"}), 404

    @app.get("/")
    def index():
        if resolved_static and (resolved_static / "index.html").is_file():
            response = make_response(
                send_from_directory(resolved_static, "index.html")
            )
        else:
            response = make_response(
                jsonify(
                    {
                        "name": "SoyRootBio 3D Editor API",
                        "message": "Build the editor frontend or run its development server.",
                    }
                )
            )
        response.set_cookie(
            "soyrootbio_editor_token",
            app.config["EDITOR_SESSION_TOKEN"],
            httponly=True,
            samesite="Strict",
            secure=False,
            path="/",
        )
        return response

    @app.get("/<path:path>")
    def static_files(path: str):
        if resolved_static:
            candidate = resolved_static / path
            if candidate.is_file():
                return send_from_directory(resolved_static, path)
            index_path = resolved_static / "index.html"
            if index_path.is_file():
                return send_from_directory(resolved_static, "index.html")
        return jsonify({"error": "Editor frontend is not built."}), 404

    return app


def launch_editor(
    output_dir: str | Path,
    *,
    session_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    static_dir: str | Path | None = None,
) -> None:
    if not _is_loopback_host(host):
        raise ValueError(
            "The editor contains mutation endpoints and may only bind to a loopback host."
        )
    app = create_editor_app(
        output_dir,
        session_dir=session_dir,
        static_dir=static_dir,
    )
    if open_browser:
        threading.Timer(
            0.8,
            lambda: webbrowser.open(f"http://{host}:{port}/"),
        ).start()
    app.run(host=host, port=port, threaded=True, use_reloader=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open the SoyRootBio 3D graph editor.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--session-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--static-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    launch_editor(
        args.output,
        session_dir=args.session_dir,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        static_dir=args.static_dir,
    )
    return 0


def _resolve_static_dir(static_dir: str | Path | None) -> Path | None:
    candidates: list[Path] = []
    if static_dir is not None:
        candidates.append(Path(static_dir))
    if os.environ.get("SOYROOTBIO_EDITOR_STATIC"):
        candidates.append(Path(os.environ["SOYROOTBIO_EDITOR_STATIC"]))
    repository_root = Path(__file__).resolve().parents[3]
    candidates.extend(
        [
            repository_root / "editor" / "dist" / "client",
            repository_root / "editor" / "dist",
            Path(__file__).resolve().parent / "web",
        ]
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_dir():
            return candidate
    return None


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
