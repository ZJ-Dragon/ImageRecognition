# 这串代码只能在竞赛方指定的小车上运行，若想要在其他设备（如PC）上运行，请移步 demo_pc.py

# 引入库文件
import cv2
import sys, os
import torch
import numpy as np

from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device
from utils.plots import Annotator, colors


sys.path.append('..') # 将父目录添加到系统路径中，以便可以导入父目录下的模块
os.system('sudo supervisorctl stop uniwiseRosCarTask')

# 打开摄像头并获取一帧图片
cap = cv2.VideoCapture(10)
ret,frame = cap.read()

# 设置参数
weights_path = 'weights/face04.pt' # 模型权重路径
device = '' # 默认为空，自动选择设备
augment = False # 是否进行数据增强
line_thickness = 3 # 画框线条的粗细
conf_thres = 0.5 # 置信度阈值，用于过滤低置信度的检测结果
iou_thres = 0.65 # NMS（非最大抑制）的 IOU 阈值
classes = None # 过滤特定类别的目标
agnostic_nms = True # 是否使用类无关的 NMS

# 加载模型
device = select_device(device) # 选择设备（CPU 或 GPU）
model = DetectMultiBackend(weights_path, device=device) # 加载预训练模型
names = model.names # 获取模型的类别名称

# 图像预处理
img = np.ascontiguousarray (frame)  # 将帧数据转为连续的内存布局
img = torch.from_numpy(img).to(model.device) # 转换为 PyTorch 张量，并移动到模型所在的设备
img = img.half() if model.fp16 else img.float() # 如果模型支持半精度，使用半精度，否则使用单精度
img /= 255.0 # 将图像像素值归一化到 0-1 范围
img = img. unsqueeze(0) # 扩展一个维度，作为批次大小维度

# 推理
pred = model (img, augment=augment)[0]

# 非最大抑制（NMS）
pred = non_max_suppression (pred, conf_thres, iou_thres, classes=classes,agnostic=agnostic_nms)

# 处理检测结果，在原图上绘制预测框
annotator = Annotator(frame, line_width=line_thickness, example=str(names))
for det in pred:
    if len(det):
        # 将检测框坐标从 letterbox 图像尺寸映射回原图尺寸
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            label = f'{names[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))
result_img = annotator.result()

# 保存结果图像
cv2.imwrite('/home/lamb/yolov5-master/images/result.jpg',result_img)