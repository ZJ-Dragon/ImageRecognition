# 这串代码只能在竞赛方指定的小车上运行，若想要在其他设备（如PC）上运行，请移步 demo_pc.py

# 引入库文件
import cv2
import sys
import torch
import numpy as np

# ---------------- advanced brighten helper -----------------
def enhance_exposure(img, clip=3.5, tile=(8, 8), gamma=1.4):
    """
    1. CLAHE on L-channel (LAB)  -> 提升局部亮度
    2. gamma correction on result -> 再整体提亮
    """
    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # gamma
    inv = 1.0 / gamma
    table = (np.linspace(0, 1, 256) ** inv * 255).astype('uint8')
    return cv2.LUT(img_clahe, table)
# -----------------------------------------------------------


# ------------------ path fix so utils can be imported ------------------
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parent            # .../ImageRecognition
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.append('..')  # allow parent dir as well
# ----------------------------------------------------------------------

from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from utils.dataloaders import letterbox


# os.system('sudo supervisorctl stop uniwiseRosCarTask')  # Disabled for desktop testing

# 打开摄像头并获取一帧图片
cap = cv2.VideoCapture(0)
# ---------- Auto‑exposure fallback ----------
# 0.75 == auto mode on many UVC cams (1 == manual).  If your cam supports
# separate exposure & gain controls, leave auto first; we’ll compensate
# with software brighten if still dark.
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
# ------------------------------------------------
ret,frame = cap.read()
if not ret or frame is None:
    raise RuntimeError("Failed to grab frame from camera.")

# adaptive brightness with CLAHE (more natural colors)
frame = enhance_exposure(frame, clip=4.0, tile=(8, 8), gamma=1.6)

# 设置参数
weights_path = 'weights/face04.pt' # 模型权重路径
device = '' # 默认为空，自动选择设备
augment = False # 是否进行数据增强
line_thickness = 3 # 画框线条的粗细
conf_thres = 0.5 # 置信度阈值，用于过滤低置信度的检测结果
iou_thres = 0.65 # NMS（非最大抑制）的 IOU 阈值
classes = None # 过滤特定类别的目标
agnostic_nms = True # 是否使用类无关的 NMS

#
# 加载模型
device = select_device(device) # 选择设备（CPU 或 GPU）
model = DetectMultiBackend(weights_path, device=device) # 加载预训练模型
names = model.names # 获取模型的类别名称
stride = int(model.stride)          # model stride
img_size = check_img_size(640, s=stride)  # ensure img size is multiple of stride

# -------- 图像预处理 --------
# 1) Letterbox resize to keep aspect and make dims multiple of stride
img0 = frame  # original BGR
img_letter, ratio, pad = letterbox(img0, new_shape=img_size, stride=stride)
# 2) BGR -> RGB -> contiguous
img = img_letter[:, :, ::-1].copy()
# 3) to torch tensor (N,C,H,W)
img = torch.from_numpy(img).permute(2, 0, 1).to(model.device)
img = img.half() if model.fp16 else img.float()  # uint8->fp16/32
img /= 255.0
if img.ndim == 3:
    img = img.unsqueeze(0)  # add batch dim

# 推理
pred = model (img, augment=augment)[0]

# 非最大抑制（NMS）
pred = non_max_suppression (pred, conf_thres, iou_thres, classes=classes,agnostic=agnostic_nms)

#
# 处理检测结果，在原图上绘制预测框
annotator = Annotator(frame, line_width=line_thickness, example=str(names))
for det in pred:
    if len(det):
        # 将检测框坐标从 letterbox 图像尺寸映射回原图尺寸
        det[:, :4] = scale_boxes(img_letter.shape[:2], det[:, :4], frame.shape).round()
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            label = f'{names[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))
result_img = annotator.result()

# 保存结果图像
#cv2.imwrite('/home/lamb/yolov5-master/images/result.jpg',result_img)
#cv2.imwrite('./result/result.jpg',result_img)
cv2.imshow('result', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()