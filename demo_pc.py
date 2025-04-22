import cv2
import os
import torch
import numpy as np
import sys

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.plots import Annotator, colors

if sys.platform != "win32":
    import pathlib
    pathlib.WindowsPath = pathlib.PosixPath

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    调整图像大小并填充，使其符合模型步幅要求，同时保持原始宽高比。
    """
    shape = img.shape[:2]  # 当前尺寸 (height, width)
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    # 缩放图像
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # 计算填充量
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding
    dw /= 2  # 两边各分一半
    dh /= 2
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    # 填充图像
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

# 自动检测并选择设备
if torch.cuda.is_available(): device = 'cuda'
else: device = 'cpu'

# 模型与参数设置
weights_path = 'weights/face04.pt'
augment = False
line_thickness = 3
conf_thres = 0.5
iou_thres = 0.65
classes = None
agnostic_nms = True

# 加载模型
device = select_device(device)
model = DetectMultiBackend(weights_path, device=device)
stride = model.stride  # 获取模型步幅
img_size = check_img_size(640, s=stride)  # 调整目标尺寸，确保是 stride 的倍数
names = model.names

# 加载本地图片
image_path = './data/images/huang.jpg'
assert os.path.exists(image_path), f"文件 {image_path} 不存在，请检查路径。"
frame = cv2.imread(image_path)
assert frame is not None, f"无法读取图像文件 {image_path}。"

# 将 BGR 转换为 RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# 使用 letterbox 调整图像大小并填充，保证输入尺寸符合要求
img_letterbox, ratio, pad = letterbox(frame_rgb, new_shape=(img_size, img_size))

# 转换为 PyTorch 张量，并调整为 (C, H, W)
img = torch.from_numpy(img_letterbox).float().permute(2, 0, 1)
img /= 255.0  # 归一化到 0-1
img = img.unsqueeze(0)  # 增加 batch 维度

print(f"Image tensor shape before model: {img.shape}")

# 推理
pred = model(img, augment=augment)[0]

# 非最大抑制（NMS）
pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

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
result_path = './results/result.jpg'
cv2.imwrite(result_path, result_img)

# 显示结果图像（测试所用，上线时记得打注释）
if result_img is not None:
    cv2.imshow('Result Image', result_img)# 在窗口中显示图像
    cv2.waitKey(0)# 等待用户按键
    cv2.destroyAllWindows()# 关闭所有窗口
else:
    print(f"无法读取图像文件 {result_path}。")

