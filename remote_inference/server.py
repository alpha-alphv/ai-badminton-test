"""
Remote YOLO inference server for the Badminton Analytics Project.

Runs on a GPU host (e.g. a GraphLabs / Grafilabs GPU instance) and is exposed
to the local machine through an SSH local-port-forward, so the notebook keeps
talking to ``http://localhost:8000`` while the heavy lifting happens on CUDA.

Endpoints
---------
GET  /health
    Returns the model load status and CUDA availability.

POST /predict
    Multipart form: ``image`` (jpeg/png bytes), ``model`` (detect|pose|shuttle),
    optional ``conf`` and ``iou`` floats. Returns JSON with boxes, keypoints,
    and class names mimicking ``ultralytics.Results``.

POST /track
    Multipart form: ``video`` (file), ``conf``, ``iou``, ``tracker``.
    Streams NDJSON, one line per frame, each line is the same shape as
    ``/predict`` plus a ``frame`` index and tracking ``id`` on each box.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from ultralytics import YOLO


DETECT_WEIGHTS = os.environ.get("DETECT_WEIGHTS", "yolov8n.pt")
POSE_WEIGHTS = os.environ.get("POSE_WEIGHTS", "yolov8n-pose.pt")
SHUTTLE_WEIGHTS = os.environ.get("SHUTTLE_WEIGHTS", "weights/best.pt")
DEVICE = os.environ.get("INFER_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")

MODELS: dict[str, YOLO] = {}


def _load_models() -> None:
    """Eagerly load every model onto the GPU at startup so the first request is fast."""
    MODELS["detect"] = YOLO(DETECT_WEIGHTS)
    MODELS["pose"] = YOLO(POSE_WEIGHTS)
    MODELS["shuttle"] = YOLO(SHUTTLE_WEIGHTS)
    for name, m in MODELS.items():
        try:
            m.to(DEVICE)
        except Exception as exc:
            print(f"[warn] could not move {name} to {DEVICE}: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield
    MODELS.clear()


app = FastAPI(title="Badminton Analytics Remote Inference", lifespan=lifespan)


def _decode_image(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(img)[:, :, ::-1]  # RGB -> BGR for ultralytics/opencv parity
    return np.ascontiguousarray(arr)


def _result_to_dict(result, frame_idx: Optional[int] = None) -> dict:
    payload: dict = {"names": {int(k): v for k, v in result.names.items()}}
    if frame_idx is not None:
        payload["frame"] = frame_idx

    if result.boxes is not None and len(result.boxes) > 0:
        data = result.boxes.data.detach().cpu().numpy().tolist()
        payload["boxes"] = data
        ids = getattr(result.boxes, "id", None)
        if ids is not None:
            payload["ids"] = ids.detach().cpu().numpy().tolist()
    else:
        payload["boxes"] = []

    if getattr(result, "keypoints", None) is not None and result.keypoints.xy is not None:
        kp_xy = result.keypoints.xy.detach().cpu().numpy().tolist()
        kp_conf = (
            result.keypoints.conf.detach().cpu().numpy().tolist()
            if result.keypoints.conf is not None
            else None
        )
        payload["keypoints"] = {"xy": kp_xy, "conf": kp_conf}

    return payload


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "models": sorted(MODELS.keys()),
        "weights": {
            "detect": DETECT_WEIGHTS,
            "pose": POSE_WEIGHTS,
            "shuttle": SHUTTLE_WEIGHTS,
        },
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    model: str = Form("detect"),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
):
    if model not in MODELS:
        raise HTTPException(400, f"unknown model '{model}'; choose detect|pose|shuttle")

    raw = await image.read()
    frame = _decode_image(raw)
    results = MODELS[model].predict(frame, conf=conf, iou=iou, device=DEVICE, verbose=False)
    return JSONResponse(_result_to_dict(results[0]))


@app.post("/track")
async def track(
    video: UploadFile = File(...),
    model: str = Form("detect"),
    conf: float = Form(0.4),
    iou: float = Form(0.5),
    tracker: str = Form("bytetrack.yaml"),
):
    """Run tracking server-side over an uploaded video and stream per-frame results."""
    if model not in MODELS:
        raise HTTPException(400, f"unknown model '{model}'")

    suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await video.read())
        tmp.flush()
        tmp.close()

        def gen():
            stream = MODELS[model].track(
                source=tmp.name,
                conf=conf,
                iou=iou,
                persist=True,
                tracker=tracker,
                stream=True,
                show=False,
                verbose=False,
                device=DEVICE,
            )
            for i, r in enumerate(stream):
                yield json.dumps(_result_to_dict(r, frame_idx=i)) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        workers=1,
    )
