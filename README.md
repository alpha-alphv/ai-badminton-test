# Badminton Analytics

Computer-vision badminton analytics. Upload a match clip in the web UI, the
video is streamed to a remote GPU (Grafilabs / GraphLabs) over an SSH tunnel
for tracking + shuttle detection, and the laptop renders the annotated video
and analytics charts. Job state lives in Postgres so past runs are browseable
at `/jobs`.

```
Browser ── http ──▶ ui/  (laptop container, Flask :7860)
                     │ video upload + analytics rendering
                     │ job state ←→ Postgres (sibling container)
                     ▼
           RemoteYOLO HTTP client
                     │ POST /predict, /track
                     ▼  ssh -L 8000:127.0.0.1:8000
           remote_inference/server.py  (Grafilabs GPU box, FastAPI :8000)
                     │
                     ▼
           ultralytics.YOLO on CUDA   (detect | pose | shuttle)
```

Two pieces to run:

1. **Inference server** on the Grafilabs GPU box (FastAPI + CUDA).
2. **Web UI + Postgres** on your laptop (Docker Compose), plus an SSH tunnel
   that exposes the GPU box's `:8000` as `127.0.0.1:8000` locally.

The UI never imports `torch` / `ultralytics`; the GPU box never accepts public
traffic. Everything goes through the tunnel.

---

## Prerequisites

**Laptop**
- Docker Desktop (Windows / macOS) or Docker Engine + Compose plugin (Linux).
- OpenSSH client (`ssh` on PATH). PowerShell 5.1+ on Windows.
- Your Grafilabs SSH key.

**Grafilabs GPU instance**
- Ubuntu 22.04 with an NVIDIA driver. `nvidia-smi` must work.
- Python 3.10 or 3.11.
- Outbound HTTPS so `pip` and the ultralytics weight downloader can fetch
  packages and `yolov8n*.pt` weights.

You do **not** need to install Python, CUDA, or PyTorch on the laptop. The UI
ships as a container; only the GPU box runs torch.

---

## 1. Provision the Grafilabs GPU box

SSH into your instance and clone the repo:

```bash
sudo apt-get update && sudo apt-get install -y python3.11-venv ffmpeg git
git clone <your-fork-of-this-repo> badminton
cd badminton

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install the CUDA-enabled PyTorch wheel **first**, pinned to a `cu###` that
matches the driver's CUDA version reported by `nvidia-smi`. `cu121` is the
safe baseline for any driver advertising CUDA 12.1+:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
pip install -r remote_inference/requirements-server.txt
```

Verify CUDA is wired up — this must print the device name and nothing else:

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Upload your custom shuttlecock weights from the laptop:

```powershell
# from the laptop, in the project root
scp weights\best.pt user@gpu.grafilabs.example:/home/user/badminton/weights/best.pt
```

`yolov8n.pt` and `yolov8n-pose.pt` are downloaded by `ultralytics` on first
use, so no manual upload needed for those.

> **Gotcha:** `pip install torch` without `--index-url` pulls the newest
> CPU/CUDA wheel and you'll get `RuntimeError: The NVIDIA driver on your
> system is too old`. Always pin the index URL.

---

## 2. Start the inference server on the GPU box

Bind to **localhost only** so it's reachable only through the SSH tunnel:

```bash
cd ~/badminton
source .venv/bin/activate

export HOST=127.0.0.1
export PORT=8000
export INFER_DEVICE=cuda:0          # cuda:1 on multi-GPU
export INFER_HALF=1                 # fp16 on
export SHUTTLE_WEIGHTS=weights/best.pt

# Run inside tmux so it survives SSH disconnects
tmux new -s infer
python remote_inference/server.py
# Ctrl-b d to detach; `tmux attach -t infer` to come back
```

Startup aborts loudly if `torch.cuda.is_available()` is `False`. There is
**no CPU fallback** — that's intentional.

---

## 3. Open the SSH tunnel from your laptop

Leave this terminal open for as long as you want the UI to work.

**Windows / PowerShell**

```powershell
$env:GRAFI_HOST = "user@gpu.grafilabs.example"
$env:GRAFI_PORT = "22"                              # if Grafilabs gave you a non-22 port
$env:GRAFI_KEY  = "$HOME\.ssh\grafilabs_id_ed25519"

# Bind '*' is required so the Docker UI container (which reaches the host as
# host.docker.internal — NOT loopback) can see the tunnel. Only do this on a
# trusted LAN; anyone reachable on :8000 can POST to your GPU.
$env:GRAFI_TUNNEL_BIND = "*"

.\remote_inference\start_tunnel.ps1
```

**macOS / Linux**

```bash
export GRAFI_HOST=user@gpu.grafilabs.example
export GRAFI_PORT=22
export GRAFI_KEY=~/.ssh/grafilabs_id_ed25519
./remote_inference/start_tunnel.sh
```

Verify in another terminal:

```bash
curl http://127.0.0.1:8000/health
# {"ok": true, "device": "cuda:0", "cuda": true, "half": true,
#  "gpu": {"index": 0, "name": "NVIDIA GeForce RTX 4090", ...}}
```

If `cuda` is `false` or you get `Connection refused`, fix that before
bringing up the UI — the UI's health badge will just say "tunnel down".

---

## 4. Bring up the web UI + Postgres

In the project root on the laptop:

```bash
docker compose up -d --build
```

This starts two containers:

| Container | Purpose | Address |
| --- | --- | --- |
| `ui` | Flask app (`ui/app.py`) | http://127.0.0.1:7860 |
| `postgres` | Job + artifact state | compose-internal `postgres:5432` |

Volumes:

- `uidata` — uploads at `ui/data/uploads/` and results at `ui/data/results/`.
- `pgdata` — Postgres data dir.

Both survive `docker compose down`; add `-v` to wipe them.

Useful commands:

```bash
docker compose logs -f ui                    # tail UI logs
docker compose exec postgres psql -U badminton   # ad-hoc SQL
docker compose restart ui                    # after editing code
docker compose down                          # stop (data preserved)
docker compose down -v                       # stop + wipe pgdata, uidata
```

Environment overrides (set before `docker compose up`):

| Var | Default | Purpose |
| --- | --- | --- |
| `BADMINTON_INFER_URL` | `http://host.docker.internal:8000` | Where the UI reaches the tunnel. |
| `MAX_UPLOAD_MB` | `500` | Per-upload size cap. |

---

## 5. Use it

Open http://127.0.0.1:7860.

- The GPU health badge in the navbar should read `✓ <GPU name>`. If it
  shows `✗ tunnel down`, step 3 is broken.
- Drop a clip onto the upload card. The job page streams progress over
  Server-Sent Events; when it finishes you get the annotated MP4, nine
  analytics PNGs, and five CSVs.
- Past jobs are at http://127.0.0.1:7860/jobs.

---

## Daily restart

Once the box is provisioned (steps 1 & 2's `pip install` are one-time), a
normal session is just:

```bash
# on the GPU box (via SSH; tmux pane survives logout)
tmux attach -t infer     # or start fresh if the pane is gone

# on the laptop
.\remote_inference\start_tunnel.ps1     # leave running
docker compose up -d                    # leave running
```

Open http://127.0.0.1:7860.

---

## Bare-metal alternative (no Docker)

If you'd rather run the UI without containers:

```powershell
# Postgres in a one-off container
docker run -d --name pg `
  -e POSTGRES_USER=badminton -e POSTGRES_PASSWORD=badminton -e POSTGRES_DB=badminton `
  -p 127.0.0.1:5432:5432 postgres:16-alpine

# Python deps
pip install -r ui\requirements.txt

# Start the UI
$env:DATABASE_URL = "postgresql://badminton:badminton@127.0.0.1:5432/badminton"
$env:BADMINTON_INFER_URL = "http://127.0.0.1:8000"
.\ui\start_ui.ps1
```

In this mode the tunnel can stay on the default `127.0.0.1` bind — no
`GRAFI_TUNNEL_BIND=*` needed.

---

## Troubleshooting

- **UI says `✗ tunnel down`** — the SSH tunnel terminal closed, or the
  bind address is wrong for Docker. From the container, the host's
  loopback is *not* `127.0.0.1`; you need `GRAFI_TUNNEL_BIND=*` (or the
  bare-metal path above).
- **`/health` shows `"cuda": false`** — the server fell back to CPU
  (shouldn't happen — startup is supposed to abort). Re-check that you
  installed the `cuXXX` torch wheel before `requirements-server.txt`.
- **`CUDA out of memory`** — keep `INFER_HALF=1`, or request a larger GPU.
- **Job stuck at "uploading video"** — large clip; the entire file is
  posted to `/track` once. Watch `docker compose logs -f ui` and the
  server's tmux pane.
- **`docker compose up` errors on port 7860** — something else is bound
  to it; either kill it or change the host-side port mapping in
  `docker-compose.yml`.
- **Postgres won't start on Windows because 5432 is reserved** — the
  default compose file doesn't expose the port on the host, so this only
  affects the bare-metal path. Use a different host port (e.g. 55432).

For lower-level / notebook usage of the same inference server, see
[`remote_inference/README.md`](remote_inference/README.md).

---

## Repo layout

```
ui/                      # Flask web app (containerized)
  app.py                 # routes: /, /upload, /jobs, /jobs/<id>, /results/...
  jobs.py                # per-job worker (track → pose → shuttle → analytics)
  analytics.py           # matplotlib charts ported from the notebook
  db.py                  # Postgres schema + CRUD
  templates/, static/    # UI
remote_inference/        # GPU-side FastAPI server + SSH tunnel helpers
  server.py              # POST /predict, /track  (GPU-only)
  remote_yolo.py         # ultralytics.YOLO drop-in over HTTP
  start_tunnel.ps1, .sh
notebooks/               # Original Colab analysis notebook
weights/best.pt          # Custom shuttlecock detector
docker-compose.yml       # ui + postgres
```

No tests / linters / build system. Validate Python edits with
`python -m py_compile <file>`.
