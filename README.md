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
    git clone https://github.com/yourusername/yolov5-object-detection.git
    cd yolov5-object-detection
    ```

2. 安装所需的包：
    ```sh
    pip install -r requirements.txt
    ```

## 使用方法

### 使用 PyTorch 进行推理

1. 将图像放置在 `data/images` 目录中。
2. 运行 `demo_pc.py` 脚本：
    ```sh
    python demo_pc.py
    ```  
   注意到程序现在报错一大堆，您可以自己动手解决，解决后记得 push 一下。  
如果您和原作者一样比较懒，可以等待我们的团队修复这个问题。

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

# 致谢  
感谢 [ultralytics/yolov5](https://github.com/ultralytics/yolov5) 提供的 YOLOv5 模型。  
这个模型为我们的项目提供了莫大的帮助👍。
