# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Computer-vision badminton analytics. A Google-Colab Jupyter notebook (`notebooks/Badminton Analysis` — note the file has **no `.ipynb` extension**, see Gotchas) does the original analysis: YOLOv8 person detection + ByteTrack tracking, YOLOv8-pose for joints, and a custom-trained YOLO11 shuttlecock detector (`weights/best.pt`) for shot detection. The repo has since grown two extra components that move inference off the notebook host and onto a remote GPU, fronted by a local web UI.

## Three-tier architecture

```
Browser ── http ──▶ ui/  (laptop container, Flask :7860)
                     │ video upload + analytics rendering
                     │ job state ←→ Postgres (sibling container)
                     ▼
           RemoteYOLO HTTP client
                     │ POST /predict, /track
                     ▼  ssh -L 8000:127.0.0.1:8000
           remote_inference/server.py  (GPU box, FastAPI :8000)
                     │
                     ▼
           ultralytics.YOLO on CUDA
              (detect | pose | shuttle)
```

- **`remote_inference/server.py`** runs on the Grafilabs GPU box and is **GPU-only by design** — `_require_cuda()` aborts startup if `torch.cuda.is_available()` is `False`. It loads three models keyed by role: `detect` (`yolov8n.pt`), `pose` (`yolov8n-pose.pt`), `shuttle` (`weights/best.pt`). FP16 is on by default (`INFER_HALF=1`). Server binds to `127.0.0.1` only; exposure is through the SSH tunnel. **Not containerized** — the SSH-tunnel + GPU host model doesn't fit a single-host compose file.
- **`remote_inference/remote_yolo.py`** is a drop-in for `ultralytics.YOLO` that mirrors the subset of the ultralytics API the notebook uses: `model(frame)`, `model.track(source=..., stream=True, ...)`, `results[0].boxes.data.cpu().numpy()`, `results[0].names`, `keypoints.xy`. Calls go over HTTP to the tunneled server. Used by both the notebook (after the three-line swap) and the UI.
- **`ui/app.py`** is the Flask web app. `POST /upload` saves the video, inserts the job row in Postgres, spawns a daemon thread that runs `Job` from `ui/jobs.py`, and redirects to a job page that streams progress over SSE (`/jobs/{id}/stream`). Flask is started with `threaded=True` so SSE doesn't starve other requests. The UI never imports `torch` or `ultralytics` — all heavy inference is remote.
- **`ui/jobs.py`** orchestrates one job: tracking via `/track` (single video upload, NDJSON stream back), sampled pose, per-frame shuttle detection, then locally rendered analytics + annotated MP4. Every `job.update(...)` and `job.add_artifact(...)` writes through to Postgres via `ui/db.py`.
- **`ui/db.py`** owns the schema (`jobs` + `artifacts`) and a `psycopg_pool.ConnectionPool`. `db.init_schema()` is called once at Flask startup; the SSE stream reads the latest snapshot from the DB on every tick.
- **`ui/analytics.py`** is the notebook's matplotlib analytics ported to headless functions (`render_*` write PNGs into `ui/data/results/<job_id>/`).
- **`docker-compose.yml`** runs `ui` and `postgres` together. `ui` reaches the SSH tunnel on the host via `host.docker.internal` (mapped to `host-gateway` for Linux). Artifacts and Postgres data live in named volumes (`uidata`, `pgdata`).

## Common commands

### One-time install

On the **GPU box** (CUDA toolkit version must be ≤ driver's `CUDA Version` from `nvidia-smi`):

```bash
pip uninstall -y torch torchvision torchaudio                              # avoid mismatched default wheel
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
pip install -r remote_inference/requirements-server.txt
python -c "import torch; assert torch.cuda.is_available()"                 # must print nothing
```

On the **laptop**:

```powershell
pip install -r ui/requirements.txt                                         # UI + client only, no torch
```

Or skip the local install entirely and use Docker (see below).

### Run the system (Docker — recommended)

Two terminals on the laptop, one on the GPU box. The UI + Postgres run together in compose; the inference server still lives on the GPU host and is reached through the SSH tunnel.

```bash
# 1. GPU box — start inference (use tmux so it survives ssh)
export INFER_DEVICE=cuda:0 INFER_HALF=1
export SHUTTLE_WEIGHTS=weights/best.pt
python remote_inference/server.py
```

```powershell
# 2. Laptop — open the SSH tunnel (leave open)
$env:GRAFI_HOST = "root@118.107.222.200"; $env:GRAFI_PORT = "39225"
.\remote_inference\start_tunnel.ps1
```

```powershell
# 3. Laptop — bring up the UI + Postgres
docker compose up -d --build              # http://127.0.0.1:7860, Postgres on 127.0.0.1:5432
docker compose logs -f ui                 # follow logs
docker compose down                       # stop; add -v to wipe pgdata + uidata
```

`DATABASE_URL` and `BADMINTON_INFER_URL` are overridable via env (see `docker-compose.yml`). The UI reaches the tunnel via `host.docker.internal:8000`; on Linux this is mapped to `host-gateway` automatically.

### Run the system (bare metal)

Same three pieces, but UI and Postgres run on the host:

```powershell
# Laptop — Postgres (any way you like; e.g. a one-off container)
docker run -d --name pg -e POSTGRES_USER=badminton -e POSTGRES_PASSWORD=badminton `
  -e POSTGRES_DB=badminton -p 127.0.0.1:5432:5432 postgres:16-alpine

# Laptop — UI
$env:DATABASE_URL = "postgresql://badminton:badminton@127.0.0.1:5432/badminton"
.\ui\start_ui.ps1                          # http://127.0.0.1:7860
```

### Smoke-test the tunnel end-to-end

```powershell
curl http://127.0.0.1:8000/health          # expect {"cuda": true, "name": "NVIDIA GeForce RTX 4090", ...}
python remote_inference\smoke_test.py "videos\Video Project.mp4" --frame 200
# writes remote_inference/smoke_test_result.png — 3 panels (detect/pose/shuttle)
```

### Working with the notebook

The notebook itself is Colab-flavored (`from google.colab import drive, files`, `drive.mount(...)`) and was not designed to run locally. For remote-GPU mode swap the three `YOLO(...)` constructors per `remote_inference/notebook_patch.md`:

```python
model      = RemoteYOLO(role="detect")     # cell 4: YOLO("yolov8n.pt")
pose_model = RemoteYOLO(role="pose")       # cells 10/20/21: YOLO("yolov8n-pose.pt")
model      = RemoteYOLO(role="shuttle")    # cell 22: YOLO(weights_path)
```

No tests / linters / build system. Validate Python edits with `python -m py_compile <file>`.

## Gotchas

- **The notebook file lacks the `.ipynb` extension** — it's at `notebooks/Badminton Analysis`. Renaming to `.ipynb` is fine for editors but the path is the literal one. If you need to programmatically read it as JSON, copy with the extension first.
- **`pip install torch` defaults to the newest cu### wheel.** If the GPU driver doesn't support that CUDA runtime, you get `RuntimeError: The NVIDIA driver on your system is too old` even on a modern driver. Always pin via `--index-url https://download.pytorch.org/whl/cuXXX` matching `nvidia-smi`. `cu121` is the safe baseline for any driver advertising CUDA 12.1+.
- **The inference server has no CPU fallback.** This is intentional — see `_require_cuda()` in `remote_inference/server.py`. Don't add a CPU branch "for testing"; spin up a CUDA box or mock `RemoteYOLO`.
- **`RemoteYOLO` normalises track output to predict's shape.** When tracking, the server returns boxes as `[x1,y1,x2,y2,score,cls]` with a separate `ids` list; `_Result.__init__` interleaves them to `[x1,y1,x2,y2,id,score,cls]` so the notebook's tuple unpacking still works. Keep this contract if you change either side.
- **Court polygon differs between notebook and UI.** Notebook uses an interactive ipywidgets slider (cell 3). UI auto-fits a 10%-inset rectangle (`_default_court` in `ui/jobs.py`). The latter is a placeholder for browser-based corner picking — if you add click-to-mark, update `court_coords.npy` in the job dir before analytics run.
- **Per-frame shuttle detection is the slow stage in the UI pipeline.** It calls `/predict` once per frame because `/track` with the shuttle model gives unreliable IDs for the small object. If you batch frames or use a different strategy, keep the output schema in `analytics.run_shuttle_detection` (`frame, x, y, score`) so the rest of the pipeline stays compatible.
- **UI job state lives in Postgres** (`jobs` + `artifacts` tables, schema in `ui/db.py`). Every `Job.update(...)` and `Job.add_artifact(...)` writes through; `snapshot(job_id)` reads back. The in-memory `JOBS` dict is gone — look jobs up via `db.get_job(id)`. Restarting the UI preserves DB rows but **does not resume in-flight workers** — threads die with the process. If you need resumption, the data is there to rebuild from; nobody's asked for it yet.
- **Files still live on disk**, not in the DB. Uploads go to `ui/data/uploads/<id>.<ext>` and results to `ui/data/results/<id>/...`. The `artifacts.path` column stores the absolute path for bookkeeping; `artifacts.url` is what the browser fetches via `/results/<...>`. In Docker both directories are inside the `uidata` named volume.
- **Don't bind any server to `0.0.0.0`.** Both `remote_inference/server.py` and `ui/app.py` should stay on `127.0.0.1`; remote access goes through SSH tunnels.
