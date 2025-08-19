# 从零实现Transformer模型 

## 简介

本项目旨在使用最基础的 PyTorch 框架，从零开始完整实现经典的 Transformer 模型。项目的核心目标并非追求最先进的性能，而是通过手写每一个核心组件，深入理解其内部架构和工作原理，为本人的学习复习过程做一个记录。

为了聚焦于模型本身，本项目实现了一个经典的 **序列复制任务 (Sequence Copy Task)**。模型将被训练学习复制任意输入的数字序列，这能非常直观地验证模型的学习能力。


## 项目结构

```
transformer/
├── module.py          # Transformer模型所有核心组件的实现
├── model.py           # Transformer模型
├── data.py            # 动态数据生成及批次处理
├── train.py           # 模型训练主脚本
├── inference.py       # 模型推理与测试脚本
└── README.md          # 本说明文件
```

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/dz1104/my_transformer.git
cd my_transformer
```

### 2. 安装依赖

确保你已经安装了Python。本项目仅需要PyTorch。

```bash
pip install torch
```
*如果你的机器支持GPU，请务必安装与你的CUDA版本匹配的PyTorch版本，以获得更快的训练速度。*

## 使用方法

### 1. 模型训练

直接运行训练脚本即可开始训练。脚本会动态生成数据，并在训练结束后自动保存模型权重到 `transformer_copy_model.pth` 文件。

```bash
python train.py
```

训练日志输出：
```
Using device: cuda
--- Epoch 1 ---
Epoch Step: 1 Loss: 2.872746 Tokens per Sec: 1045.362305
Epoch Step: 51 Loss: 1.834612 Tokens per Sec: 1873.578125
...
Epoch 1 Training Loss: 2.1534871234
Model saved.
```

### 2. 模型推理

训练完成后，运行推理脚本来测试模型的效果。脚本会加载已保存的模型，并对一个给定的序列进行预测。

```bash
python inference.py
```

模型的输入和贪心解码后的输出：
```
Input sequence: [[ 1  2  3  4  5  6  7  8  9 10]]
Model output:   [[ 1  2  3  4  5  6  7  8  9 10]]
```


## 实现的核心概念

本项目完整实现了 "Attention Is All You Need" 论文中的以下核心组件：

- [x] **词嵌入 (Embeddings)** 与 **位置编码 (Positional Encoding)**
- [x] **缩放点积注意力 (Scaled Dot-Product Attention)**
- [x] **多头注意力机制 (Multi-Head Attention)**
- [x] **残差连接 (Residual Connections)** 与 **层归一化 (Layer Normalization)**
- [x] **位置前馈网络 (Position-wise Feed-Forward Networks)**
- [x] **编码器 (Encoder)** 与 **解码器 (Decoder)** 的完整堆叠架构
- [x] **序列掩码 (Masking)** 用于处理Padding和防止未来信息泄露

