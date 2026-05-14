"""
Local web UI for the Badminton Analytics project (Flask edition).

Run locally on the laptop while the SSH tunnel to the Grafilabs GPU is open.
Visit http://127.0.0.1:7860 to upload a match clip; the file is forwarded to
the remote inference server through the tunnel, tracking + shuttle detection
run on the RTX 4090, and the annotated video is offered for download here.

Architecture
------------
    Browser ──▶ Flask (this file, :7860)
                     │  uploads video
                     ▼
                Job runner (thread, jobs.py)
                     │  state ←→ Postgres (db.py)
                     │  POST /track over the tunnel
                     ▼
            Remote inference server (127.0.0.1:8000 -> GPU box)
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

import requests
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
)
from werkzeug.exceptions import RequestEntityTooLarge

import db
from jobs import Job, run_job, snapshot


UI_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("UI_DATA_DIR", str(UI_DIR / "data")))
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
for d in (UPLOAD_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)

INFER_URL = os.environ.get("BADMINTON_INFER_URL", "http://127.0.0.1:8000")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv"}

app = Flask(
    __name__,
    template_folder=str(UI_DIR / "templates"),
    static_folder=str(UI_DIR / "static"),
    static_url_path="/static",
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

db.init_schema()


@app.route("/")
def index():
    return render_template("upload.html", infer_url=INFER_URL, max_mb=MAX_UPLOAD_MB)


@app.route("/health")
def health():
    try:
        r = requests.get(f"{INFER_URL}/health", timeout=5)
        return jsonify({"ui": "ok", "tunnel": r.ok, "inference": r.json()})
    except Exception as exc:
        return jsonify({"ui": "ok", "tunnel": False, "error": str(exc)}), 502


@app.route("/upload", methods=["POST"])
def upload():
    video = request.files.get("video")
    if not video or not video.filename:
        return jsonify({"error": "no file"}), 400

    suffix = Path(video.filename).suffix.lower() or ".mp4"
    if suffix not in ALLOWED_EXT:
        return jsonify({"error": f"unsupported extension {suffix}"}), 400

    job_id = uuid.uuid4().hex[:12]
    saved = UPLOAD_DIR / f"{job_id}{suffix}"
    video.save(str(saved))

    try:
        conf = float(request.form.get("conf", "0.4"))
        iou = float(request.form.get("iou", "0.5"))
    except ValueError:
        saved.unlink(missing_ok=True)
        return jsonify({"error": "conf/iou must be numeric"}), 400

    do_shuttle = request.form.get("do_shuttle", "false").lower() in ("true", "on", "1", "yes")

    job = Job(
        id=job_id,
        input_path=saved,
        result_dir=RESULT_DIR / job_id,
        infer_url=INFER_URL,
        conf=conf,
        iou=iou,
        do_shuttle=do_shuttle,
    )
    job.result_dir.mkdir(parents=True, exist_ok=True)
    job.persist()

    threading.Thread(target=run_job, args=(job_id,), daemon=True).start()
    return redirect(f"/jobs/{job_id}", code=303)


@app.errorhandler(RequestEntityTooLarge)
def _too_large(_exc):
    return jsonify({"error": f"file exceeds {MAX_UPLOAD_MB} MB cap"}), 413


@app.route("/jobs")
def jobs_index():
    rows = db.list_jobs()
    return render_template("jobs.html", jobs=rows)


@app.route("/jobs/<job_id>")
def job_page(job_id: str):
    row = db.get_job(job_id)
    if not row:
        return f"<h1>unknown job {job_id}</h1>", 404
    return render_template("job.html", job_id=job_id, filename=row["input_filename"])


@app.route("/jobs/<job_id>/status")
def job_status(job_id: str):
    snap = snapshot(job_id)
    if snap is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(snap)


@app.route("/jobs/<job_id>/stream")
def job_stream(job_id: str):
    """Server-Sent Events stream of job progress so the page updates live."""
    if snapshot(job_id) is None:
        return jsonify({"error": "unknown job"}), 404

    def gen():
        last = None
        while True:
            snap = snapshot(job_id)
            if snap is None:
                break
            if snap != last:
                yield f"data: {json.dumps(snap)}\n\n"
                last = snap
            if snap["status"] in ("done", "error"):
                break
            time.sleep(0.5)

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/results/<path:filename>")
def results(filename: str):
    """Serve any artifact written into ui/data/results/<job_id>/..."""
    return send_from_directory(RESULT_DIR, filename, conditional=True)


if __name__ == "__main__":
    app.run(
        host=os.environ.get("UI_HOST", "127.0.0.1"),
        port=int(os.environ.get("UI_PORT", "7860")),
        threaded=True,
        debug=False,
    )
