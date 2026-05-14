"""Quick end-to-end check that the SSH-tunneled remote inference works.

Usage:
    python remote_inference/smoke_test.py
    python remote_inference/smoke_test.py path/to/video.mp4 --frame 100
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from remote_yolo import RemoteYOLO  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default="videos/Video Project.mp4")
    ap.add_argument("--frame", type=int, default=100)
    ap.add_argument("--url", default=os.environ.get("BADMINTON_INFER_URL", "http://127.0.0.1:8000"))
    args = ap.parse_args()

    print(f"[1/4] health check {args.url} ...")
    h = RemoteYOLO(role="detect", url=args.url).health()
    print(f"      device={h['device']} cuda={h['cuda']} half={h.get('half')}")
    if h.get("gpu"):
        print(f"      gpu={h['gpu']['name']} mem={h['gpu']['mem_total_mb']} MB")

    print(f"[2/4] grabbing frame {args.frame} from {args.video}")
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {args.frame} from {args.video}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, role in zip(axes, ["detect", "pose", "shuttle"]):
        print(f"[3/4] inferring '{role}' ...")
        t0 = time.perf_counter()
        results = RemoteYOLO(role=role, url=args.url)(frame)
        dt = (time.perf_counter() - t0) * 1000

        r = results[0]
        n_boxes = len(r.boxes)
        n_kpts = 0 if r.keypoints is None else (
            r.keypoints.xy.numpy().shape[0] if r.keypoints.xy is not None else 0
        )
        print(f"      {role}: {n_boxes} boxes, {n_kpts} keypoint sets, {dt:.1f} ms round-trip")

        vis = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
        for box in r.boxes.data:
            # detect/shuttle: x1,y1,x2,y2,score,cls   (after RemoteYOLO normalises shape)
            # track or with id: x1,y1,x2,y2,id,score,cls
            x1, y1, x2, y2 = (int(v) for v in box[:4])
            cls = int(box[-1])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, r.names.get(cls, str(cls)), (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        if r.keypoints is not None and r.keypoints.xy is not None:
            for person in r.keypoints.xy.numpy():
                for (kx, ky) in person:
                    if kx > 0 and ky > 0:
                        cv2.circle(vis, (int(kx), int(ky)), 3, (255, 80, 80), -1)
        ax.imshow(vis)
        ax.set_title(f"{role} ({n_boxes} det, {dt:.0f} ms)")
        ax.axis("off")

    out = "remote_inference/smoke_test_result.png"
    print(f"[4/4] saving overlay → {out}")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print("DONE")


if __name__ == "__main__":
    main()
