"""
Notebook analytics ported to a function library so the UI can render them
headlessly. Each ``render_*`` writes a PNG to ``out_dir`` and registers it as an
artifact via ``add_artifact``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


COLORS = {"P1": "lime", "P2": "darkorange"}
CV_COLORS = {"P1": (80, 255, 80), "P2": (255, 140, 40)}


# ----------------------------------------------------------------------------
# Pose
# ----------------------------------------------------------------------------
def run_pose_on_samples(video: Path, client, sample_count: int = 60) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return pd.DataFrame()

    sample_count = min(sample_count, total)
    indices = np.linspace(0, total - 1, sample_count, dtype=int)
    rows = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        r = client(frame)[0]
        if r.keypoints is None or r.keypoints.xy is None:
            continue
        kps_all = r.keypoints.xy.numpy()
        for person_idx, kps in enumerate(kps_all):
            # COCO-17: 5/6 shoulders, 11/12 hips, 13/14 knees, 15/16 ankles
            if kps.shape[0] < 17:
                continue
            l_sh, r_sh = kps[5], kps[6]
            l_hp, r_hp = kps[11], kps[12]
            l_an, r_an = kps[15], kps[16]
            leg_stretch = float(np.linalg.norm(l_an - r_an))
            shoulder_mid = (l_sh + r_sh) / 2
            hip_mid = (l_hp + r_hp) / 2
            torso = shoulder_mid - hip_mid
            body_angle = float(np.degrees(np.arctan2(torso[0], -torso[1])))
            rows.append({
                "frame": int(idx),
                "person": person_idx,
                "leg_stretch": leg_stretch,
                "body_angle": body_angle,
            })
    cap.release()
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Shuttle detection (per-frame remote call)
# ----------------------------------------------------------------------------
def run_shuttle_detection(video: Path, client, progress_cb: Callable[[int, int], None] | None = None) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = client(frame)[0]
        # take highest-score box whose class name contains "shuttle"
        best = None
        for box in r.boxes.data:
            x1, y1, x2, y2 = box[:4]
            score = box[-2]
            cls = int(box[-1])
            name = r.names.get(cls, "").lower()
            if "shuttle" not in name and "cock" not in name:
                continue
            if best is None or score > best["score"]:
                best = {
                    "frame": i,
                    "x": float((x1 + x2) / 2),
                    "y": float((y1 + y2) / 2),
                    "score": float(score),
                }
        if best is not None:
            rows.append(best)
        i += 1
        if progress_cb and i % 10 == 0:
            progress_cb(i, total)
    cap.release()
    if progress_cb:
        progress_cb(i, total)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _normalize(df: pd.DataFrame, x_min, x_max, y_min, y_max):
    nx = (df["x"] - x_min) / max(x_max - x_min, 1e-6)
    ny = (df["y"] - y_min) / max(y_max - y_min, 1e-6)
    return nx.clip(0, 1).values, ny.clip(0, 1).values


# ----------------------------------------------------------------------------
# Top-level renderer
# ----------------------------------------------------------------------------
def render_all_plots(
    traj_df: pd.DataFrame,
    shuttle_df: pd.DataFrame,
    court_coords: np.ndarray,
    frame_shape: tuple[int, int],
    fps: float,
    out_dir: Path,
    add_artifact: Callable[[str, Path], None],
) -> None:
    h, w = frame_shape

    per_player: dict[str, pd.DataFrame] = {
        label: df.sort_values("frame").reset_index(drop=True)
        for label, df in traj_df.groupby("player")
    }

    # speeds (m/s assuming 13 m court width across frame)
    scale_m_per_px = 13.0 / w
    speeds: dict[str, pd.DataFrame] = {}
    for label, df in per_player.items():
        x = df["x"].astype(float).values
        y = df["y"].astype(float).values
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        dist_m = np.sqrt(dx * dx + dy * dy) * scale_m_per_px
        speed_mps = dist_m * fps
        speeds[label] = pd.DataFrame({
            "frame": df["frame"].values,
            "dist_m": dist_m,
            "speed_mps": speed_mps,
        })

    _trajectories_over_court(per_player, court_coords, frame_shape, out_dir, add_artifact)
    _movement_heatmap(per_player, frame_shape, out_dir, add_artifact)
    _speed_over_time(speeds, fps, out_dir, add_artifact)
    _convex_hull(per_player, frame_shape, out_dir, add_artifact)
    _court_dominance(per_player, frame_shape, out_dir, add_artifact)
    _zone_transitions(per_player, frame_shape, out_dir, add_artifact)
    _speed_weighted_court(per_player, speeds, frame_shape, out_dir, add_artifact)
    _recovery_position(per_player, frame_shape, out_dir, add_artifact)

    if not shuttle_df.empty:
        _shuttle_trajectory(shuttle_df, frame_shape, out_dir, add_artifact)


def _trajectories_over_court(per_player, court, frame_shape, out_dir, add_artifact):
    h, w = frame_shape
    fig, ax = plt.subplots(figsize=(10, 6))
    poly = np.vstack([court, court[:1]])
    ax.plot(poly[:, 0], poly[:, 1], color="white", lw=2)
    ax.set_facecolor("#1a4d2e")
    for label, df in per_player.items():
        ax.plot(df["x"], df["y"], color=COLORS[label], lw=1.5, alpha=0.85, label=label)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_title("Player Trajectories Over Court")
    ax.legend(loc="upper right")
    out = out_dir / "trajectories.png"
    _save(fig, out)
    add_artifact("trajectories", out)


def _movement_heatmap(per_player, frame_shape, out_dir, add_artifact):
    h, w = frame_shape
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, label in zip(axes, ["P1", "P2"]):
        if label not in per_player:
            ax.set_visible(False)
            continue
        df = per_player[label]
        hist, xe, ye = np.histogram2d(df["x"], df["y"], bins=100, range=[[0, w], [0, h]])
        ax.imshow(np.power(hist.T, 0.5), extent=[0, w, h, 0], cmap="magma", aspect="auto")
        ax.set_title(f"{label} Movement Heatmap")
        ax.axis("off")
    out = out_dir / "heatmap.png"
    _save(fig, out)
    add_artifact("heatmap", out)


def _speed_over_time(speeds, fps, out_dir, add_artifact):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, label in zip(axes, ["P1", "P2"]):
        if label not in speeds:
            ax.set_visible(False)
            continue
        sdf = speeds[label]
        ax.plot(sdf["frame"] / fps, sdf["speed_mps"], color=COLORS[label], lw=1.8)
        ax.set_title(f"{label} Speed Over Time")
        ax.set_ylabel("m/s")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("seconds")
    out = out_dir / "speed_over_time.png"
    _save(fig, out)
    add_artifact("speed_over_time", out)


def _convex_hull(per_player, frame_shape, out_dir, add_artifact):
    h, w = frame_shape
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("#1a4d2e")
    for label, df in per_player.items():
        pts = df[["x", "y"]].values
        if len(pts) < 3:
            continue
        try:
            hull = ConvexHull(pts)
            poly = pts[hull.vertices]
            ax.fill(poly[:, 0], poly[:, 1], alpha=0.35, color=COLORS[label], label=f"{label} hull")
            ax.scatter(pts[:, 0], pts[:, 1], s=4, color=COLORS[label], alpha=0.6)
        except Exception:
            pass
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_title("Convex Hull — Court Coverage")
    ax.legend()
    out = out_dir / "convex_hull.png"
    _save(fig, out)
    add_artifact("convex_hull", out)


def _zone_grid(frame_shape, n_x: int = 3, n_y: int = 3):
    h, w = frame_shape
    xs = np.linspace(0, w, n_x + 1)
    ys = np.linspace(0, h, n_y + 1)
    return xs, ys


def _zone_of(point, xs, ys):
    x, y = point
    ix = max(0, min(len(xs) - 2, int(np.searchsorted(xs, x) - 1)))
    iy = max(0, min(len(ys) - 2, int(np.searchsorted(ys, y) - 1)))
    return iy * (len(xs) - 1) + ix


def _court_dominance(per_player, frame_shape, out_dir, add_artifact):
    h, w = frame_shape
    bins_x, bins_y = 30, 20
    h1, _, _ = np.histogram2d(
        per_player.get("P1", pd.DataFrame(columns=["x", "y"]))["x"],
        per_player.get("P1", pd.DataFrame(columns=["x", "y"]))["y"],
        bins=[bins_x, bins_y], range=[[0, w], [0, h]],
    )
    h2, _, _ = np.histogram2d(
        per_player.get("P2", pd.DataFrame(columns=["x", "y"]))["x"],
        per_player.get("P2", pd.DataFrame(columns=["x", "y"]))["y"],
        bins=[bins_x, bins_y], range=[[0, w], [0, h]],
    )
    diff = (h1 - h2)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(diff.T, extent=[0, w, h, 0], cmap="RdYlGn", aspect="auto")
    ax.set_title("Court Dominance Map (P1 − P2)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, label="P1 ← dominance → P2")
    out = out_dir / "court_dominance.png"
    _save(fig, out)
    add_artifact("court_dominance", out)


def _zone_transitions(per_player, frame_shape, out_dir, add_artifact):
    xs, ys = _zone_grid(frame_shape)
    n = (len(xs) - 1) * (len(ys) - 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, label in zip(axes, ["P1", "P2"]):
        if label not in per_player:
            ax.set_visible(False)
            continue
        df = per_player[label].sort_values("frame")
        zones = [_zone_of((x, y), xs, ys) for x, y in df[["x", "y"]].values]
        mat = np.zeros((n, n), dtype=float)
        for a, b in zip(zones[:-1], zones[1:]):
            mat[a, b] += 1
        row_sums = mat.sum(axis=1, keepdims=True)
        prob = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)
        im = ax.imshow(prob, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{label} Zone Transitions")
        ax.set_xlabel("to zone")
        ax.set_ylabel("from zone")
        fig.colorbar(im, ax=ax, fraction=0.04)
    out = out_dir / "zone_transitions.png"
    _save(fig, out)
    add_artifact("zone_transitions", out)


def _speed_weighted_court(per_player, speeds, frame_shape, out_dir, add_artifact):
    h, w = frame_shape
    bins_x, bins_y = 40, 25
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, label in zip(axes, ["P1", "P2"]):
        if label not in per_player or label not in speeds:
            ax.set_visible(False)
            continue
        df = per_player[label].sort_values("frame")
        s = speeds[label]
        merged = df.merge(s, on="frame", how="left")
        H, _, _ = np.histogram2d(
            merged["x"], merged["y"], bins=[bins_x, bins_y],
            range=[[0, w], [0, h]], weights=merged["speed_mps"].fillna(0),
        )
        C, _, _ = np.histogram2d(
            merged["x"], merged["y"], bins=[bins_x, bins_y], range=[[0, w], [0, h]],
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(C > 0, H / C, 0)
        ax.imshow(avg.T, extent=[0, w, h, 0], cmap="inferno", aspect="auto")
        ax.set_title(f"{label} Speed-Weighted Court (m/s)")
        ax.axis("off")
    out = out_dir / "speed_weighted_court.png"
    _save(fig, out)
    add_artifact("speed_weighted_court", out)


def _recovery_position(per_player, frame_shape, out_dir, add_artifact):
    h, w = frame_shape
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("#1a4d2e")
    for label, df in per_player.items():
        mean_x = df["x"].mean()
        mean_y = df["y"].mean()
        std_x = df["x"].std()
        std_y = df["y"].std()
        ax.scatter(df["x"], df["y"], s=3, color=COLORS[label], alpha=0.25)
        ax.scatter([mean_x], [mean_y], s=180, color=COLORS[label], edgecolor="white",
                   lw=2, zorder=5, label=f"{label} mean")
        circle = plt.Circle((mean_x, mean_y), max(std_x, std_y), color=COLORS[label],
                            alpha=0.15)
        ax.add_patch(circle)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_title("Recovery Position (mean + dispersion)")
    ax.legend()
    out = out_dir / "recovery_position.png"
    _save(fig, out)
    add_artifact("recovery_position", out)


def _shuttle_trajectory(shuttle_df, frame_shape, out_dir, add_artifact):
    h, w = frame_shape
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#1a4d2e")
    ax.plot(shuttle_df["x"], shuttle_df["y"], color="yellow", lw=1, alpha=0.7)
    ax.scatter(shuttle_df["x"], shuttle_df["y"], s=8, c=shuttle_df["frame"],
               cmap="plasma")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_title("Shuttle Trajectory (time-coloured)")
    out = out_dir / "shuttle_trajectory.png"
    _save(fig, out)
    add_artifact("shuttle_trajectory", out)


# ----------------------------------------------------------------------------
# Shot detection (ported from notebook cell 22)
# ----------------------------------------------------------------------------
def detect_shots(shuttle_df: pd.DataFrame, traj_df: pd.DataFrame, fps: float, frame_w: int) -> pd.DataFrame:
    if shuttle_df.empty:
        return pd.DataFrame(columns=["Shot_Number", "Frame", "Shot_Type", "Player"])

    sdf = shuttle_df.sort_values("frame").reset_index(drop=True)
    frames = sdf["frame"].values
    x = sdf["x"].values
    y = sdf["y"].values
    if len(frames) < 4:
        return pd.DataFrame(columns=["Shot_Number", "Frame", "Shot_Type", "Player"])

    vx = np.diff(x)
    vy = np.diff(y)
    speed_px = np.sqrt(vx * vx + vy * vy)
    scale_m_per_px = 13.0 / frame_w
    speed_mps = speed_px * scale_m_per_px * fps
    accel = np.abs(np.diff(speed_mps))

    thresh = accel.mean() + 1.6 * accel.std()
    spikes = np.where(accel > thresh)[0]

    merged = [0]
    last = -999
    for s in spikes:
        if s - last > 4:
            merged.append(int(s))
            last = int(s)

    by_frame: dict[int, dict[str, tuple[float, float]]] = {}
    for _, row in traj_df.iterrows():
        by_frame.setdefault(int(row["frame"]), {})[row["player"]] = (row["x"], row["y"])

    records = []
    prev_player = None
    for i, eidx in enumerate(merged):
        fnum = int(frames[min(eidx, len(frames) - 1)])
        sx, sy = float(x[min(eidx, len(x) - 1)]), float(y[min(eidx, len(y) - 1)])

        nearest = None
        for offset in range(0, 4):
            for f_try in (fnum - offset, fnum + offset):
                if f_try in by_frame:
                    nearest = by_frame[f_try]
                    break
            if nearest:
                break

        hitter = "Unknown"
        if nearest:
            dists = {p: np.hypot(sx - px, sy - py) for p, (px, py) in nearest.items()}
            if dists:
                hitter = min(dists, key=dists.get)

        if hitter == prev_player or hitter == "Unknown":
            hitter = "P1" if prev_player == "P2" else "P2"
        prev_player = hitter

        dxn = x[min(eidx + 1, len(x) - 1)] - x[eidx]
        dyn = y[min(eidx + 1, len(y) - 1)] - y[eidx]
        angle_deg = abs(np.degrees(np.arctan2(dyn, dxn)))
        v_next = speed_mps[min(eidx, len(speed_mps) - 1)]

        if v_next > np.percentile(speed_mps, 75):
            stype = "smash"
        elif angle_deg < 20:
            stype = "drive"
        elif angle_deg > 45:
            stype = "clear"
        else:
            stype = "net/drop"

        records.append({
            "Shot_Number": i + 1,
            "Frame": fnum,
            "Shot_Type": stype,
            "Player": hitter,
        })
    return pd.DataFrame(records)


# ----------------------------------------------------------------------------
# Annotated video
# ----------------------------------------------------------------------------
def render_annotated_video(
    video: Path,
    track_results: list[dict],
    player_map: dict[int, str],
    shuttle_df: pd.DataFrame,
    court_coords: np.ndarray,
    out_path: Path,
    progress_cb: Callable[[int, int], None] | None = None,
) -> None:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    by_frame = {tr["frame"]: tr for tr in track_results}
    shuttle_by_frame = {
        int(row["frame"]): (float(row["x"]), float(row["y"]))
        for _, row in shuttle_df.iterrows()
    } if not shuttle_df.empty else {}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # court polygon
        cv2.polylines(frame, [court_coords.reshape(-1, 1, 2).astype(np.int32)],
                      True, (200, 200, 200), 2)
        tr = by_frame.get(i)
        if tr:
            for box in tr["boxes"]:
                if len(box) < 7:
                    continue
                x1, y1, x2, y2, tid, score, cls = box[:7]
                if int(cls) != 0:
                    continue
                label = player_map.get(int(tid))
                if not label:
                    continue
                color = CV_COLORS[label]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label}", (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        shuttle = shuttle_by_frame.get(i)
        if shuttle:
            cv2.circle(frame, (int(shuttle[0]), int(shuttle[1])), 8, (0, 255, 255), -1)
            cv2.circle(frame, (int(shuttle[0]), int(shuttle[1])), 14, (0, 255, 255), 2)
        writer.write(frame)
        i += 1
        if progress_cb and i % 30 == 0:
            progress_cb(i, total)
    cap.release()
    writer.release()
    if progress_cb:
        progress_cb(i, total)
