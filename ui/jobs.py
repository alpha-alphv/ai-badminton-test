"""
Background job runner: drives the remote inference server through the SSH
tunnel, then runs all of the notebook analytics locally on the resulting
trajectories.

Job state lives in Postgres (see ui/db.py). The in-process Job object is a
thin handle that writes through to the database on every update so the UI's
SSE stream can serve fresh snapshots even after a process restart.
"""
from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd

UI_DIR = Path(__file__).parent
sys.path.insert(0, str((UI_DIR.parent / "remote_inference").resolve()))
from remote_yolo import RemoteYOLO  # noqa: E402

import analytics  # noqa: E402
import db  # noqa: E402


@dataclass
class Job:
    id: str
    input_path: Path
    result_dir: Path
    infer_url: str
    conf: float = 0.4
    iou: float = 0.5
    do_shuttle: bool = True

    status: str = "queued"
    message: str = ""
    progress: float = 0.0
    total_frames: int = 0
    frames_done: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: Optional[str] = None

    def persist(self) -> None:
        """Insert the initial row for this job."""
        db.insert_job({
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "total_frames": self.total_frames,
            "frames_done": self.frames_done,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "input_filename": self.input_path.name,
            "input_path": str(self.input_path),
            "result_dir": str(self.result_dir),
            "infer_url": self.infer_url,
            "conf": self.conf,
            "iou": self.iou,
            "do_shuttle": self.do_shuttle,
        })

    def update(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)
        db.update_job(self.id, kw)

    def add_artifact(self, key: str, path: Path) -> None:
        url = f"/results/{self.id}/{path.name}"
        db.upsert_artifact(self.id, key, url, str(path))


def snapshot(job_id: str) -> Optional[dict]:
    row = db.get_job(job_id)
    if row is None:
        return None
    finished_at = row["finished_at"]
    return {
        "id": row["id"],
        "status": row["status"],
        "message": row["message"],
        "progress": round(row["progress"], 3),
        "total_frames": row["total_frames"],
        "frames_done": row["frames_done"],
        "started_at": row["started_at"],
        "finished_at": finished_at,
        "elapsed_s": round((finished_at or time.time()) - row["started_at"], 1),
        "error": row["error"],
        "artifacts": db.list_artifacts(row["id"]),
    }


def load_job(job_id: str) -> Optional[Job]:
    row = db.get_job(job_id)
    if row is None:
        return None
    return Job(
        id=row["id"],
        input_path=Path(row["input_path"]),
        result_dir=Path(row["result_dir"]),
        infer_url=row["infer_url"],
        conf=row["conf"],
        iou=row["iou"],
        do_shuttle=row["do_shuttle"],
        status=row["status"],
        message=row["message"],
        progress=row["progress"],
        total_frames=row["total_frames"],
        frames_done=row["frames_done"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        error=row["error"],
    )


def run_job(job_id: str) -> None:
    job = load_job(job_id)
    if not job:
        return
    try:
        _run(job)
    except Exception as exc:
        job.update(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            message=traceback.format_exc().splitlines()[-1],
            finished_at=time.time(),
        )


def _probe_video(path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return total, fps, w, h


def _default_court(w: int, h: int) -> np.ndarray:
    """Inset polygon (10% from each edge) used when no user-supplied court coords exist."""
    dx, dy = int(w * 0.10), int(h * 0.10)
    return np.array(
        [[dx, dy], [w - dx, dy], [w - dx, h - dy], [dx, h - dy]],
        dtype=np.int32,
    )


def _run(job: Job) -> None:
    total, fps, w, h = _probe_video(job.input_path)
    job.update(total_frames=total, message=f"{total} frames @ {fps:.1f} fps {w}x{h}")

    court = _default_court(w, h)
    np.save(job.result_dir / "court_coords.npy", court)

    detect_client = RemoteYOLO(role="detect", url=job.infer_url, conf=job.conf, iou=job.iou)
    pose_client = RemoteYOLO(role="pose", url=job.infer_url, conf=0.3, iou=0.5)
    shuttle_client = RemoteYOLO(role="shuttle", url=job.infer_url, conf=0.25, iou=0.45)

    # 1. health check
    health = detect_client.health()
    if not health.get("cuda"):
        raise RuntimeError(f"remote inference is not on CUDA: {health}")

    # 2. tracking ------------------------------------------------------------
    job.update(status="tracking", message="streaming /track from GPU box")
    traj_rows = []
    track_results: list[dict] = []
    player_map: dict[int, str] = {}

    for i, r in enumerate(detect_client.track(
        source=str(job.input_path),
        conf=job.conf,
        iou=job.iou,
        tracker="bytetrack.yaml",
        stream=True,
    )):
        track_results.append({
            "frame": i,
            "boxes": r.boxes.data.tolist(),
            "names": r.names,
        })
        # boxes shape: x1,y1,x2,y2,id,score,cls
        for box in r.boxes.data:
            if len(box) < 7:
                continue
            x1, y1, x2, y2, tid, score, cls = box[:7]
            if int(cls) != 0:
                continue
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            tid = int(tid)
            if tid not in player_map:
                player_map[tid] = "P1" if cy < h / 2 else "P2"
            traj_rows.append((i, player_map[tid], float(cx), float(cy)))

        job.update(frames_done=i + 1, progress=(i + 1) / max(total, 1) * 0.55)

    if not traj_rows:
        raise RuntimeError("no player detections returned from tracker")

    traj_df = pd.DataFrame(traj_rows, columns=["frame", "player", "x", "y"])
    traj_df = traj_df.groupby(["player", "frame"], as_index=False).mean(numeric_only=True)
    for label, df in traj_df.groupby("player"):
        out = job.result_dir / f"{label}_trajectory.csv"
        df[["frame", "x", "y"]].to_csv(out, index=False)
        job.add_artifact(f"trajectory_{label}", out)

    # 3. pose (sampled) -------------------------------------------------------
    job.update(status="pose", message="sampling pose frames", progress=0.6)
    pose_rows = analytics.run_pose_on_samples(
        video=job.input_path,
        client=pose_client,
        sample_count=min(60, total),
    )
    pose_csv = job.result_dir / "pose_metrics.csv"
    pose_rows.to_csv(pose_csv, index=False)
    job.add_artifact("pose_metrics", pose_csv)

    # 4. shuttle detection ---------------------------------------------------
    shuttle_df = pd.DataFrame(columns=["frame", "x", "y", "score"])
    if job.do_shuttle:
        job.update(status="shuttle", message="running shuttle detector frame-by-frame", progress=0.65)
        shuttle_df = analytics.run_shuttle_detection(
            video=job.input_path,
            client=shuttle_client,
            progress_cb=lambda done, total_: job.update(
                frames_done=done,
                progress=0.65 + (done / max(total_, 1)) * 0.15,
            ),
        )
        shuttle_csv = job.result_dir / "shuttle_trajectory.csv"
        shuttle_df.to_csv(shuttle_csv, index=False)
        job.add_artifact("shuttle_trajectory", shuttle_csv)

    # 5. analytics -----------------------------------------------------------
    job.update(status="analytics", message="rendering plots", progress=0.82)
    analytics.render_all_plots(
        traj_df=traj_df,
        shuttle_df=shuttle_df,
        court_coords=court,
        frame_shape=(h, w),
        fps=fps,
        out_dir=job.result_dir,
        add_artifact=job.add_artifact,
    )

    if job.do_shuttle and not shuttle_df.empty:
        shots_df = analytics.detect_shots(shuttle_df, traj_df, fps=fps, frame_w=w)
        shots_csv = job.result_dir / "shots.csv"
        shots_df.to_csv(shots_csv, index=False)
        job.add_artifact("shots", shots_csv)

    # 6. annotated mp4 -------------------------------------------------------
    job.update(status="rendering", message="encoding annotated video", progress=0.9)
    annotated = job.result_dir / "annotated.mp4"
    analytics.render_annotated_video(
        video=job.input_path,
        track_results=track_results,
        player_map=player_map,
        shuttle_df=shuttle_df,
        court_coords=court,
        out_path=annotated,
        progress_cb=lambda done, total_: job.update(
            frames_done=done,
            progress=0.9 + (done / max(total_, 1)) * 0.10,
        ),
    )
    job.add_artifact("annotated_video", annotated)

    job.update(
        status="done",
        message="complete",
        progress=1.0,
        finished_at=time.time(),
    )
