# Application

# import libraries
import cv2
import sys, os
import torch
import numpy as np

from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device
from utils.plots import Annotator, colors


sys.path.append('..') # add path
os.system('sudo supervisorctl stop uniwiseRosCarTask')

# take photo
cap = cv2.VideoCapture(10)
ret,frame = cap.read()

# set values
weights_path = 'weights/yolov5s.pt'
device = '' # default
augment = False
line_thickness = 3
conf_thres = 0.5 # confidence threshold
iou_thres = 0.65 # iou threshold for NMS
classes = None # filter by class: --class 0, or --class 0 2 3
agnostic_nms = True # class-agnostic NMS

# load the model
device = select_device(device)
model = DetectMultiBackend(weights_path, device=device)
names = model.names

# convert format
img = np.ascontiguousarray (img)
img = torch.from_numpy(img).to(model.device)
img = img.half() if model.fp16 else img.float()
img /= 255.0
img = img. unsqueeze(0)

# reasoning
pred = model (img, augment=augment)[0]

# non-maximum suppression
pred = non_max_suppression (pred, conf_thres, iou_thres, classes=classes,agnostic=agnostic_nms)

# process
for i, det in enumerate (pred) :
annotator = Annotator (frame, line_width=line_thickness, example=str(names))
det[:,:4] = scale_boxes(img.shape[2:], det[:,:4], frame.shape).round()
for *xyxy, conf, cls in reversed(det):
    c = int(cls)
    label = '%s %.2f' % (names[int(cls)], conf)


## Waiting for completion