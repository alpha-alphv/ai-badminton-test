"""
Drop-in ``ultralytics.YOLO``-compatible client that talks to the remote
inference server over an SSH-tunneled HTTP connection.

Notebook usage
--------------
The notebook does three things with ultralytics:

>>> model = YOLO("yolov8n.pt")                  # generic detector
>>> results = model(frame)                       # single image
>>> stream = model.track(source=video_path, stream=True, ...)  # tracking

All three are covered here. Swap ``YOLO(...)`` for ``RemoteYOLO(...)`` with a
``role`` argument indicating which server-side model to invoke:

>>> from remote_inference.remote_yolo import RemoteYOLO
>>> model = RemoteYOLO(role="detect")            # generic person detector
>>> pose_model = RemoteYOLO(role="pose")         # YOLOv8-pose
>>> shuttle_model = RemoteYOLO(role="shuttle")   # custom YOLO11 (best.pt)

The default endpoint is ``http://127.0.0.1:8000``. Override per instance or via
the ``BADMINTON_INFER_URL`` environment variable.
"""
from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np
import requests


DEFAULT_URL = os.environ.get("BADMINTON_INFER_URL", "http://127.0.0.1:8000")


@dataclass
class _Boxes:
    """Mimics ``ultralytics.engine.results.Boxes`` for the fields used in the notebook."""

    data: np.ndarray  # (N, 6) for predict; (N, 7) for track ([x1,y1,x2,y2,id,score,cls])
    id: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.data.shape[0])

    # The notebook calls ``.cpu().numpy()`` on this. We're already numpy, so make
    # the calls no-ops by returning self/array.
    def cpu(self):
        return self

    def numpy(self):
        return self.data


class _NPWrap:
    """Numpy array wearing ``.cpu().numpy()`` so result code is identical."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


@dataclass
class _Keypoints:
    xy: Optional[_NPWrap]
    conf: Optional[_NPWrap]


class _Result:
    """Mimics ``ultralytics.engine.results.Results`` for the fields used in the notebook."""

    def __init__(self, payload: dict):
        data = np.asarray(payload.get("boxes", []), dtype=np.float32)
        if data.size == 0:
            data = np.zeros((0, 6), dtype=np.float32)
        ids = payload.get("ids")
        if ids is not None:
            ids_arr = np.asarray(ids, dtype=np.float32).reshape(-1, 1)
            # Reorder predict-shape (x1,y1,x2,y2,score,cls) into track-shape
            # (x1,y1,x2,y2,id,score,cls) so notebook unpacking still works.
            if data.shape[1] == 6:
                data = np.concatenate([data[:, :4], ids_arr, data[:, 4:6]], axis=1)
            self.boxes = _Boxes(data=data, id=ids_arr.flatten())
        else:
            self.boxes = _Boxes(data=data)

        self.names = {int(k): v for k, v in payload.get("names", {}).items()}

        kp = payload.get("keypoints")
        if kp:
            xy = np.asarray(kp.get("xy", []), dtype=np.float32) if kp.get("xy") else None
            conf = np.asarray(kp.get("conf", []), dtype=np.float32) if kp.get("conf") else None
            self.keypoints = _Keypoints(
                xy=_NPWrap(xy) if xy is not None else None,
                conf=_NPWrap(conf) if conf is not None else None,
            )
        else:
            self.keypoints = None

        self.frame_idx = payload.get("frame")

    # Some downstream code calls ``r.plot()`` — keep a stub so it fails loudly only
    # if it's actually used.
    def plot(self):  # pragma: no cover
        raise NotImplementedError("plot() runs on the server-side ultralytics object; render locally.")


class RemoteYOLO:
    """Minimal ultralytics.YOLO-compatible HTTP client."""

    def __init__(
        self,
        weights: Optional[str] = None,
        role: str = "detect",
        url: str = DEFAULT_URL,
        conf: float = 0.25,
        iou: float = 0.45,
        timeout: float = 120.0,
    ):
        # ``weights`` is accepted for drop-in symmetry with YOLO("xxx.pt") but is
        # ignored — the server decides which weights to load at startup. The
        # ``role`` is what selects detect / pose / shuttle on the server side.
        self.weights = weights
        self.role = role
        self.url = url.rstrip("/")
        self.conf = conf
        self.iou = iou
        self.timeout = timeout

    def __call__(self, frame, conf: Optional[float] = None, iou: Optional[float] = None):
        return self.predict(frame, conf=conf, iou=iou)

    def predict(self, frame, conf: Optional[float] = None, iou: Optional[float] = None):
        if isinstance(frame, str):
            with open(frame, "rb") as f:
                img_bytes = f.read()
        elif isinstance(frame, np.ndarray):
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                raise RuntimeError("failed to JPEG-encode frame for remote inference")
            img_bytes = buf.tobytes()
        else:
            raise TypeError(f"unsupported frame type: {type(frame)!r}")

        files = {"image": ("frame.jpg", img_bytes, "image/jpeg")}
        data = {
            "model": self.role,
            "conf": str(self.conf if conf is None else conf),
            "iou": str(self.iou if iou is None else iou),
        }
        resp = requests.post(f"{self.url}/predict", files=files, data=data, timeout=self.timeout)
        resp.raise_for_status()
        return [_Result(resp.json())]

    def track(
        self,
        source: str,
        conf: float = 0.4,
        iou: float = 0.5,
        persist: bool = True,
        tracker: str = "bytetrack.yaml",
        stream: bool = True,
        show: bool = False,
        verbose: bool = False,
        **_unused,
    ) -> Iterator[_Result]:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"video not found: {source}")

        data = {
            "model": self.role,
            "conf": str(conf),
            "iou": str(iou),
            "tracker": tracker,
        }
        with open(source, "rb") as fh:
            files = {"video": (os.path.basename(source), fh, "video/mp4")}
            with requests.post(
                f"{self.url}/track",
                files=files,
                data=data,
                stream=True,
                timeout=self.timeout,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    yield _Result(json.loads(line))

    def to(self, *_args, **_kwargs):
        # Device placement is the server's job — keep the call signature.
        return self

    def health(self) -> dict:
        r = requests.get(f"{self.url}/health", timeout=10)
        r.raise_for_status()
        return r.json()
