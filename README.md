# YOLOv5 目标检测项目

```
该项目适用于 2025 青科赛自动驾驶小车的附加任务。
完成 2025 青科赛后，本项目原作者会看心情继续维护，
也欢迎参与本竞赛的后浪继续维护这个项目。  
```

*Together we Advance_*


## 最基本的环境要求

- Python 3.7+

`其余详见 requirements.txt`

---
指南：若使用 IDE 自动配置环境时出现一堆很 null 的错误，可以试试照着
`requirements.txt`
手动一个个安装包，又或者在 IDE（如 pycharm）中的“运行”框中的错误代码旁边直接点击`安装xxx`  

另外，想必大家都知道 conda 喜欢在电脑里拉屎，因此我们不建议使用 conda 来管理环境，
可根据个人喜好选择 miniconda 或者 pipenv 来管理环境。

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

1. `data/images` 目录中自带比赛所需要的图像，不过是由相机拍摄，清晰度和小车拍摄的不一样。
2. 运行 `demo_pc.py` 脚本：
    ```sh
    python demo_pc.py
    ```

### 命令行选项

您可以使用命令行选项自定义推理：

```sh
python demo_pc.py --weights ./weights/face04.pt --imgsz 640 640 --batch-size 1 --dynamic
```

## 训练模型

*若要自定义数据集进行训练，本文暂不提供指导，请前往其他网站获取支持*

1. 使用`HuggingFace`上的 [ImageRecognition]()

### 特别注意  
 - 本项目保留了应该在小车上运行的主程序，记得运行`main.py`, `demo_pc.py`是在电脑上调试用的。

# 致谢  
感谢 [ultralytics/yolov5](https://github.com/ultralytics/yolov5) 提供的 YOLOv5 模型。  
这个模型为我们的项目提供了莫大的帮助👍。