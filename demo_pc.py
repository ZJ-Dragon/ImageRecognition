import cv2
import os
import torch
import numpy as np

from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device
from utils.plots import Annotator, colors

# 自动检测并选择设备
if torch.cuda.is_available():
    device = 'cuda:0'  # 如果有 NVIDIA GPU，则使用 CUDA
elif torch.backends.mps.is_available():
    device = 'mps'  # 如果有苹果芯片 GPU（MPS）则使用 MPS
else:
    device = 'cpu'  # 如果没有 GPU，则使用 CPU

# 模型与参数设置
weights_path = 'weights/yolov5s.pt'  # 模型权重路径
augment = False  # 是否进行数据增强
line_thickness = 3  # 画框线条的粗细
conf_thres = 0.5  # 置信度阈值，用于过滤低置信度的检测结果
iou_thres = 0.65  # NMS（非最大抑制）的 IOU 阈值
classes = None  # 过滤特定类别的目标
agnostic_nms = True  # 是否使用类无关的 NMS

# 加载模型
device = select_device(device)  # 选择设备（CUDA、MPS 或 CPU）
model = DetectMultiBackend(weights_path, device=device)  # 加载预训练模型
names = model.names  # 获取模型的类别名称

# 加载本地图片作为输入
image_path = './materials/img01.jpg'  # 指定图像文件路径
assert os.path.exists(image_path), f"文件 {image_path} 不存在，请检查路径。"
frame = cv2.imread(image_path)
assert frame is not None, f"无法读取图像文件 {image_path}。"

# 将 BGR 图像转换为 RGB（YOLOv5 通常使用 RGB）
# img_size = 640  # YOLOv5 默认使用 640x640
# img = cv2.resize(frame, (img_size, img_size))
img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = np.ascontiguousarray(frame)  # 转为连续内存布局
img = torch.from_numpy(frame).to(model.device)  # 转换为 PyTorch 张量，并移动到设备

# 调整张量维度顺序为 [C, H, W] 并增加批次维度 [1, C, H, W]
img = img.permute(2, 0, 1).unsqueeze(0)
img = img.half() if model.fp16 else img.float()  # 若模型支持半精度则使用
img /= 255.0  # 将像素归一化到 [0,1]

# 打印张量形状以确认
print(f"Image tensor shape before model: {img.shape}")  # 应该输出 [1, 3, H, W]

# 推理
pred = model(frame, augment=augment)[0]  # 进行前向推理，得到预测张量

# 非最大抑制（NMS）
pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

# 处理检测结果，在原图上绘制预测框
annotator = Annotator(frame, line_width=line_thickness, example=str(names))
for det in pred:
    if len(det):
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            label = f'{names[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))

maskImg = annotator.result()

# 保存结果图像
result_path = './results/result.jpg'
cv2.imwrite(result_path, maskImg)