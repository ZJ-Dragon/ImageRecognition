# YOLOv5 目标检测项目

该项目适用于 2025 青科赛自动驾驶小车的附加任务。
完成 2025 青科赛后，本项目原作者会看心情继续维护，
也欢迎参与本竞赛的后浪继续维护这个项目。  

*Together we Advance_*

## 环境要求

- Python 3.7+
- PyTorch
- TensorFlow
- OpenCV
- NumPy  

`详见 requirements.txt`

## 安装

1. 克隆仓库：
    ```sh
    git clone https://github.com/ZJ-Dragon/ImageRecognition.git
    cd ./ImageRecognition
    ```

2. 安装所需的包：
    ```sh
    pip install -r requirements.txt
    ```

## 使用方法

### 在普通电脑上使用 PyTorch 进行推理

1. 将图像放置在 `data/images` 目录中。
2. 运行 `demo_pc.py` 脚本：
    ```sh
    python demo_pc.py
    ```

### 导出 YOLOv5 模型到 TensorFlow

1. 运行 `models/tf.py` 中的 `run` 函数以导出模型：
    ```sh
    python models/tf.py
    ```

### 命令行选项

您可以使用命令行选项自定义推理：

```sh
python demo_pc.py --weights path/to/weights.pt --imgsz 640 640 --batch-size 1 --dynamic
```

### 特别注意  
 - 在竞赛用的小车上，记得运行`main.py`, `demo_pc.py`是在电脑上调试用的。

# 致谢  
感谢 [ultralytics/yolov5](https://github.com/ultralytics/yolov5) 提供的 YOLOv5 模型。  
这个模型为我们的项目提供了莫大的帮助👍。
