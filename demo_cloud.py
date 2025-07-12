"""
demo_cloud.py
=============
A minimal Flask‑based HTTP server that lets *any* device (car browser, phone, etc.)
in the same Wi‑Fi/LAN upload a photo, runs YOLOv5 face‑recognition on the Mac,
and returns both the annotated image and the JSON results.

• GET  /             -> HTML page with <video> camera preview + “Capture” button
• POST /infer        -> multipart/form‑data (field “file”) with an image
                        ↳ server saves original, runs inference via YOLOv5 model
                        ↳ writes annotated image to static/result.jpg
                        ↳ returns JSON { "success": true, "img": "/static/result.jpg",
                                         "labels": [ {box, name, conf}, ... ] }

The code *reuses* the preprocessing / model parts from main.py, so main.py remains untouched.
"""

import os
import io
import time
from pathlib import Path
from typing import List, Dict

from flask import Flask, request, jsonify, send_from_directory, render_template_string

import cv2
import numpy as np
import torch

# --- ensure repo root is in PYTHONPATH -------------------------------
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# --------------------------------------------------------------------

import sys
# ---------- reuse YOLOv5 utils (same import style as main.py) ----------
# --- ensure local repo utils shadow any installed ones
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))  # ensure our repo root has priority
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
STATIC_DIR = ROOT / "static"
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# ------------- Load model once ----------------------------------------
device = select_device("mps" if torch.backends.mps.is_available() else "cpu")
weights = "./weights/face04.pt"
model = DetectMultiBackend(str(weights), device=device)
stride = int(model.stride)
img_size = 640
names = model.names
# ----------------------------------------------------------------------


def run_inference(img_bgr: np.ndarray) -> Dict:
    """ Runs YOLOv5 inference on a single BGR image """
    img0 = img_bgr.copy()
    img, ratio, pad = letterbox(img0, new_shape=img_size, stride=stride)
    img = img[:, :, ::-1].copy()  # BGR -> RGB

    img = torch.from_numpy(img).permute(2, 0, 1).to(device)
    img = img.half() if model.fp16 else img.float()
    img /= 255.0
    img = img.unsqueeze(0)

    pred = model(img, augment=False, visualize=False)[0]
    pred = non_max_suppression(pred, 0.25, 0.65, agnostic=True)

    annotator = Annotator(img0, line_width=3, example=str(names))
    detections = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                label_txt = f"{names[int(cls)]} {conf:.2f}"
                annotator.box_label(xyxy, label_txt, color=colors(int(cls), True))
                detections.append({
                    "class": names[int(cls)],
                    "conf": float(conf),
                    "box": [int(x) for x in xyxy]
                })
    result_img = annotator.result()
    return {"annotated": result_img, "detections": detections}


# ------------------- Flask app ----------------------------------------
app = Flask(__name__, static_folder=str(STATIC_DIR))


@app.route("/", methods=["GET"])
def index():
    """Simple camera capture page (works in modern mobile browsers)."""
    html = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>WEB demo</title>
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <link rel="stylesheet" href="/static/src/style.css">
</head>
<body>

<h2>Capture &amp; Upload</h2>

<video id="v" autoplay playsinline></video><br>
<button id="shot">📸 Capture</button>
<p id="msg"></p>
<img id="res" />

<script src="/static/src/script.js"></script>
</body>
</html>
    """
    return render_template_string(html)


@app.route("/infer", methods=["POST"])
def infer():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "no file"}), 400

    f = request.files['file']
    ts = int(time.time() * 1000)
    raw_path = UPLOAD_DIR / f"{ts}.jpg"
    f.save(raw_path)

    img_bgr = cv2.imread(str(raw_path))
    if img_bgr is None:
        return jsonify({"success": False, "error": "decode failed"}), 400

    result = run_inference(img_bgr)
    out_path = STATIC_DIR / "result.jpg"
    cv2.imwrite(str(out_path), result["annotated"])

    return jsonify({
        "success": True,
        "img": "/static/result.jpg",
        "detections": result["detections"]
    })


# ---------------- main --------------------
if __name__ == "__main__":
    host = "0.0.0.0"      # 让同网段设备可访问
    port = 1999
    print(f"Server running: http://{host}:{port}")
    app.run(host=host, port=1999, threaded=True,
            ssl_context=("cert.pem", "key.pem"))