# Local Upload UI

A small Flask web app that runs on your laptop. You drag a match clip into it,
it ships the video over the SSH tunnel to the Grafilabs GPU box for inference,
then renders all the notebook analytics locally and shows them on a result page.

```
Browser  ──▶  UI  (laptop, :7860)  ──ssh -L 8000──▶  inference server (GPU)
                │
                └─▶ renders annotated.mp4 + analytics PNGs locally
```

## Prereqs

1. SSH tunnel to the GPU box is open (`remote_inference/start_tunnel.ps1`).
2. Inference server running on the GPU box (`python remote_inference/server.py`).
3. `curl http://127.0.0.1:8000/health` returns `cuda: true`.

## Install

```powershell
cd C:\Users\alpha\Documents\alphv\Badminton_Analytics_Project
pip install -r ui\requirements.txt
```

You don't need `torch` / `ultralytics` locally — only the client-side libs.

## Run

```powershell
.\ui\start_ui.ps1
```

Then open <http://127.0.0.1:7860>. The badge in the top-right turns green and
shows the GPU name once the tunnel is healthy.

## What it produces

Per upload (job id is a 12-char hex), `ui/data/results/<job_id>/`:

- `annotated.mp4` — boxes for P1/P2 plus shuttle marker
- `P1_trajectory.csv`, `P2_trajectory.csv` — frame-by-frame positions
- `shuttle_trajectory.csv` — shuttle position when detected
- `shots.csv` — frame, hitter, shot type (smash / drive / clear / net-drop)
- `pose_metrics.csv` — sampled leg stretch and body angle
- `court_coords.npy` — polygon used for the run
- Analytics PNGs: `trajectories`, `heatmap`, `court_dominance`,
  `convex_hull`, `zone_transitions`, `speed_weighted_court`,
  `recovery_position`, `speed_over_time`, `shuttle_trajectory`

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `BADMINTON_INFER_URL` | `http://127.0.0.1:8000` | Tunneled inference endpoint. |
| `UI_HOST` | `127.0.0.1` | Don't bind to `0.0.0.0` unless you trust the LAN. |
| `UI_PORT` | `7860` | |
| `MAX_UPLOAD_MB` | `500` | Hard cap on uploaded file size. |

## How requests flow

1. `POST /upload` saves the video under `ui/data/uploads/<job>.mp4` and spawns
   a daemon `threading.Thread` running `run_job`. Flask's dev server is started
   with `threaded=True` so the SSE progress stream doesn't block other requests.
2. The task calls `RemoteYOLO(role=...).track(source=...)` — this uploads the
   video to the GPU box once and streams per-frame NDJSON back.
3. Pose runs as a sampled set (60 frames spaced through the clip) over
   `/predict`.
4. Shuttle detection runs frame-by-frame over `/predict`. (Frame-by-frame is
   slower than a single track call but more reliable for the small shuttle.)
5. Analytics run on the resulting dataframes — matplotlib renders PNGs into
   the job's result directory, served as static files.
6. The annotated MP4 is encoded locally with OpenCV using the cached track +
   shuttle results — no second remote round-trip.

## Court polygon

The notebook had an interactive slider to set court corners. The UI auto-fits
a 10%-inset rectangle and saves it as `court_coords.npy` next to the outputs.
If you want a precise polygon, drop your own 4×2 int array into the result
folder before re-running, or open an issue and we can add a click-to-mark step.

## Troubleshooting

- **Health badge red** — tunnel is down or the server isn't running.
- **Job stuck in `tracking`** — first NDJSON line only arrives once ultralytics
  starts emitting frames; for a 5-minute 1080p clip that's ~20 s after submit.
- **Empty `shots.csv`** — shuttle detector didn't fire enough boxes. Check
  `shuttle_trajectory.csv`; if it's empty, lower the conf threshold on the
  shuttle role in `jobs.py` (it's currently `0.25`).
- **OOM rendering plots locally** — `matplotlib` uses the Agg backend; should
  be fine on a laptop. Reduce gallery image DPI in `analytics.py` (default 150)
  if you really need to.
