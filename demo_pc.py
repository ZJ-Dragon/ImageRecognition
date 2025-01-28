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
    device = 'cuda'  # 如果有 NVIDIA GPU，则使用 CUDA
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
image_path = './data/images/bus.jpg'  # 指定图像文件路径
assert os.path.exists(image_path), f"文件 {image_path} 不存在，请检查路径。"
frame = cv2.imread(image_path)
assert frame is not None, f"无法读取图像文件 {image_path}。"

# 将 BGR 图像转换为 RGB（YOLOv5 通常使用 RGB）
img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = np.ascontiguousarray(frame)  # 转为连续内存布局
img = torch.from_numpy(frame).to(model.device)  # 转换为 PyTorch 张量，并移动到设备

# 假设 frame 是一个 NumPy 数组
frame_tensor = torch.from_numpy(frame)
frame_tensor = torch.from_numpy(frame).to(torch.float32)  # 显式转换为 float32 类型
frame_tensor = frame_tensor.permute(2, 0, 1)  # 转置为通道优先（CHW）

# 图像预处理
img = frame_tensor.float()
img /= 255.0 # 将图像像素值归一化到 0-1 范围
img = frame_tensor.unsqueeze(0) # 扩展一个维度，作为批次大小维度

# 打印张量形状以确认
print(f"Image tensor shape before model: {img.shape}")  # 我也不知道预期形状是什么

# 推理
pred = model(img, augment=augment)[0]  # 进行前向推理，得到预测张量

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