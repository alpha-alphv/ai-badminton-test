# Notebook patch — switch to RemoteYOLO

Drop this cell in just after the `from ultralytics import YOLO` import in
`Badminton_Analysis.ipynb` (currently cell 1):

```python
# --- Remote inference bootstrap ---------------------------------------------
import sys, os
sys.path.append(os.path.abspath("../remote_inference"))   # adjust if running from /content
from remote_yolo import RemoteYOLO
os.environ.setdefault("BADMINTON_INFER_URL", "http://127.0.0.1:8000")

# Sanity-check the tunnel is up before any inference runs.
_health = RemoteYOLO(role="detect").health()
print(f"Remote inference ready on {_health['device']} (cuda={_health['cuda']})")
```

Then change only the three `YOLO(...)` lines:

```diff
- model = YOLO("yolov8n.pt")
+ model = RemoteYOLO(role="detect")

- pose_model = YOLO("yolov8n-pose.pt")
+ pose_model = RemoteYOLO(role="pose")

- model = YOLO(weights_path)
+ model = RemoteYOLO(role="shuttle")
```

Everything else — `.track`, `.boxes.data.cpu().numpy()`, `.names`,
`.keypoints.xy` — keeps working untouched.
