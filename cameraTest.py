import cv2

cap = cv2.VideoCapture(0)          # 从 0 号开始试
ret, frame = cap.read()
if not ret or frame is None:
    raise RuntimeError("摄像头打不开，检查设备索引或驱动！")