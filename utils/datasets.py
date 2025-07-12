"""
utils/datasets.py

Minimal dataset utilities for YOLOv5-style inference.
Includes the letterbox function to resize and pad images
while maintaining aspect ratio and meeting stride constraints.
"""

import cv2
import numpy as np


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114),
              auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resize and pad image while meeting stride-multiple constraints.

    Args:
        img (np.ndarray): BGR image to resize.
        new_shape (tuple|int): target shape (height, width) or int for square.
        color (tuple): RGB padding color.
        auto (bool): if True, minimum rectangle; padding to multiple of stride.
        scaleFill (bool): if True, stretch image to new_shape without padding.
        scaleup (bool): if False, only scale down; do not scale up.
        stride (int): model stride value.

    Returns:
        img (np.ndarray): resized and padded image.
        ratio (tuple): scaling ratio (height_scale, width_scale).
        pad (tuple): padding applied (dw, dh) in pixels.
    """
    # Original shape
    shape = img.shape[:2]  # (height, width)

    # Ensure new_shape is tuple
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Compute scale ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute unpadded resized dimensions
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (width, height)

    # Compute padding
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding

    if auto:  # adjust padding to be multiple of stride
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # stretch without padding
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    # Divide padding into 2 sides
    dw /= 2
    dh /= 2

    # Resize image
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Compute border (top, bottom, left, right)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    # Add border
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             borderType=cv2.BORDER_CONSTANT,
                             value=color)

    # Return padded image, scaling ratio, and padding
    ratio = (r, r) if not scaleFill else r
    return img, ratio, (dw, dh)
