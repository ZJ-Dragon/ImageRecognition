#!/bin/bash
<<<<<<< HEAD
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
=======
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

>>>>>>> ultralytics/master
# Download latest models from https://github.com/ultralytics/yolov5/releases
# Example usage: bash data/scripts/download_weights.sh
# parent
# └── yolov5
#     ├── yolov5s.pt  ← downloads here
#     ├── yolov5m.pt
#     └── ...

<<<<<<< HEAD
python - <<EOF
=======
python - << EOF
>>>>>>> ultralytics/master
from utils.downloads import attempt_download

p5 = list('nsmlx')  # P5 models
p6 = [f'{x}6' for x in p5]  # P6 models
cls = [f'{x}-cls' for x in p5]  # classification models
seg = [f'{x}-seg' for x in p5]  # classification models

for x in p5 + p6 + cls + seg:
    attempt_download(f'weights/yolov5{x}.pt')

EOF
