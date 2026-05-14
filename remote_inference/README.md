# Remote GPU Inference (Grafilabs)

This folder turns the notebook's local YOLO inference into a small HTTP service
that runs on a Grafilabs / GraphLabs GPU instance. The local notebook keeps
talking to `http://127.0.0.1:8000` while an SSH tunnel forwards the traffic to
the GPU box, so nothing else in the analytics pipeline changes.

```
┌──────────────────┐    SSH -L 8000:127.0.0.1:8000     ┌────────────────────────┐
│  Local notebook  │ ───────────────────────────────▶  │  Grafilabs GPU host    │
│  RemoteYOLO()    │                                   │  uvicorn server.py     │
│  127.0.0.1:8000  │  ◀── streamed JSON detections ──  │  CUDA · ultralytics    │
└──────────────────┘                                   └────────────────────────┘
```

## 1. Provision the GPU instance

On the Grafilabs GPU box (Ubuntu + CUDA), once you SSH in:

```bash
sudo apt-get update && sudo apt-get install -y python3.11-venv ffmpeg
git clone <your-fork-of-this-repo> badminton && cd badminton

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r remote_inference/requirements-server.txt
```

Copy your custom shuttlecock weights up to the box, e.g. from your laptop:

```bash
scp weights/best.pt user@gpu.grafilabs.example:/home/user/badminton/weights/best.pt
```

The pose / generic weights (`yolov8n.pt`, `yolov8n-pose.pt`) are downloaded
automatically by `ultralytics` on first use.

## 2. Start the inference server on the GPU box

Bind to **localhost only** so it is reachable only via the SSH tunnel:

```bash
cd ~/badminton
source .venv/bin/activate
export HOST=127.0.0.1
export PORT=8000
export DETECT_WEIGHTS=yolov8n.pt
export POSE_WEIGHTS=yolov8n-pose.pt
export SHUTTLE_WEIGHTS=weights/best.pt
python remote_inference/server.py
```

For long sessions wrap it in `tmux` or a `systemd --user` unit so it survives
SSH disconnects.

## 3. Open the SSH tunnel from your laptop

```powershell
# Windows / PowerShell
$env:GRAFI_HOST = "user@gpu.grafilabs.example"
$env:GRAFI_KEY  = "$HOME\.ssh\grafilabs_id_ed25519"
.\remote_inference\start_tunnel.ps1
```

```bash
# macOS / Linux
export GRAFI_HOST=user@gpu.grafilabs.example
export GRAFI_KEY=~/.ssh/grafilabs_id_ed25519
./remote_inference/start_tunnel.sh
```

Leave that terminal open. Verify the tunnel:

```bash
curl http://127.0.0.1:8000/health
# {"ok": true, "device": "cuda:0", "cuda": true, "models": ["detect","pose","shuttle"], ...}
```

## 4. Switch the notebook to remote inference

Add this near the top of `notebooks/Badminton_Analysis.ipynb` (after the
`from ultralytics import YOLO` import):

```python
import sys, os
sys.path.append(os.path.abspath("remote_inference"))
from remote_yolo import RemoteYOLO

# point to the tunneled endpoint (default 127.0.0.1:8000)
os.environ.setdefault("BADMINTON_INFER_URL", "http://127.0.0.1:8000")
```

Then replace each model instantiation:

| Original (cell)                          | Remote replacement                                |
| ---------------------------------------- | ------------------------------------------------- |
| `model = YOLO("yolov8n.pt")` *(cell 4)*  | `model = RemoteYOLO(role="detect")`               |
| `pose_model = YOLO("yolov8n-pose.pt")` *(cells 10, 20, 21)* | `pose_model = RemoteYOLO(role="pose")` |
| `model = YOLO(weights_path)` *(cell 22)* | `model = RemoteYOLO(role="shuttle")`              |

No other lines change — `model(frame)`, `model.track(source=video_path, stream=True, ...)`,
`results[0].boxes.data.cpu().numpy()`, `results[0].names`, and pose `keypoints`
all keep working because `RemoteYOLO` returns `_Result` objects with the same
attributes ultralytics uses.

> **Note on `model.track` over the tunnel:** the entire video file is uploaded
> once to the GPU box, tracking runs server-side with ByteTrack, and per-frame
> JSON is streamed back. That is faster than per-frame uploads and preserves
> ByteTrack's temporal state.

## 5. Tear down

Stop the server with `Ctrl-C` in its tmux pane (or `systemctl --user stop ...`),
then close the local tunnel terminal. No local CUDA / Ultralytics install is
required anymore — the notebook only needs `requests`, `numpy`, `opencv-python`
(see `requirements-client.txt`).

## Troubleshooting

- **`Connection refused` on `curl /health`** — the server isn't listening on
  `127.0.0.1:8000`, or the tunnel died. `ssh -v` will show forward errors.
- **`CUDA out of memory`** — drop to `yolov8s-pose.pt`, or set
  `INFER_DEVICE=cpu` on the server side to fall back.
- **Notebook hangs on `model.track`** — the video upload is in progress; the
  first NDJSON line only arrives once the server starts emitting frames. Watch
  the server log for `ultralytics` progress.
- **TLS / auth** — the tunnel already gives you authenticated, encrypted
  transport. Do not bind the server to `0.0.0.0` on a public Grafilabs IP.
