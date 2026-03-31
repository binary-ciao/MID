# MID
CVPR 2022 论文 "Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion" 的代码

作者：Tianpei Gu*, Guangyi Chen*, Junlong Li, Chunze Lin, Yongming Rao, Jie Zhou, Jiwen Lu

[[论文]](https://arxiv.org/abs/2203.13777) | [[视频演示]](https://www.youtube.com/watch?v=g1vf9wio6VM)

<p align="center">
  <img src="https://user-images.githubusercontent.com/21379120/204936740-65891c87-c4c1-467f-a883-8311af89ba09.gif" alt="animated" />
</p>

> 人类行为具有不确定性 nature，这要求行人轨迹预测系统对未来运动状态的多模态进行建模。与现有的通常使用潜在变量来表示多模态的随机轨迹预测方法不同，我们明确模拟了人类运动从不确定到确定的变化过程。在本文中，我们提出了一个新的框架，将轨迹预测任务表述为**运动不确定扩散（MID）**的逆过程，在该过程中，我们逐步从所有可行区域丢弃不确定性，直到达到期望的轨迹。这个过程通过参数化的马尔可夫链学习，该链以观察到的轨迹为条件。我们可以调整链的长度来控制不确定性的程度，并平衡预测的多样性和确定性。具体来说，我们将历史行为信息和社会交互编码为状态嵌入，并设计了一个基于 Transformer 的扩散模型来捕获轨迹的时间依赖性。

## **2023年4月更新**

我们将 DDIM 集成到 MID 框架中，只需使用**两步**就能达到类似的性能，相比原始的 100 步生成加速 **50 倍**。

该更新是对 ```models/diffusion.py``` 的一行修改。要启用快速采样，可以在 ```main.py``` 中将采样方式设置为 **ddim** 并设置步数。请注意，步数需要是你训练扩散步数（我们的设置中为 100）的因数。快速采样可以直接应用于任何**已训练**模型，无需使用 DDIM 重新训练。

在我们的实验中，使用相同训练模型，仅需两步扩散就能在 ETH 数据集上达到 0.41/0.71（对比原始 100 步的 0.39/0.66）。

## 代码

### 环境依赖
    PyTorch == 1.7.1
    CUDA > 10.1

### 数据准备

ETH/UCY 和 Stanford Drone 数据集的预处理数据分割在 ```raw_data``` 中。我们对数据进行预处理并生成 .pkl 文件用于训练。

运行以下命令：

```
python process_data.py
```

`train/validation/test/` 数据分割与 [Social GAN](https://github.com/agrimgupta92/sgan) 中的相同。详细信息请参阅 ```process_data.py```。

### 训练

#### 第一步：在 ```/configs``` 中修改或创建你自己的配置文件

你可以根据需要调整配置文件中的参数，并在 ```models/diffusion.py``` 中更改扩散模型的网络架构。

确保 ```eval_mode``` 设置为 False。

#### 第二步：训练 MID

```python main.py --config configs/YOUR_CONFIG.yaml --dataset DATASET```

注意 ```$DATASET``` 应为 ["eth", "hotel", "univ", "zara1", "zara2", "sdd"] 中的一个。

日志和检查点将自动保存。

### 评估

要评估已训练模型，请在配置文件中将 ```eval_mode``` 设置为 True，并从 ```eval_at``` 设置你要评估的轮数，然后运行：

```python main.py --config configs/YOUR_CONFIG.yaml --dataset DATASET```

由于扩散模型是一个迭代过程，评估过程可能需要较长时间。我们正在开发更快版本的 MID，或者你可以设置更少的扩散步数（默认 100 步）。

### 引用
```
@inproceedings{gu2022stochastic,
  title={Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion},
  author={Gu, Tianpei and Chen, Guangyi and Li, Junlong and Lin, Chunze and Rao, Yongming and Zhou, Jie and Lu, Jiwen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17113--17122},
  year={2022}
}
```

### 许可证

我们的代码基于 MIT 许可证发布。
