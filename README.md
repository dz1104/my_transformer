# 从零复现Transformer (Transformer from Scratch)

## 📖 项目简介

本项目旨在使用基础的 PyTorch 框架，从零开始完整复现经典的 Transformer 模型（源自论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)）。项目的核心目的是为了深入理解 Transformer 的底层架构和每一个组件的细节，为个人学习复习做一个记录。

为了验证模型的正确性和有效性，本项目实现了一个完整的端到端训练流程，并在一个经典的**序列复制任务**上进行了测试。

##  项目特性

*   **纯PyTorch实现：** 仅依赖 PyTorch 核心库，不使用任何高级封装（如 `nn.Transformer`），所有关键模块均为手动实现。
*   **模块化设计：** 代码结构清晰，将模型组件（`model.py`）、数据处理（`data.py`）、训练流程（`train.py`）和工具函数（`utils.py`）解耦，易于阅读和扩展。
*   **端到端流程：** 包含了从数据生成、模型构建、训练循环、损失计算到最终推理验证的完整流程。
*   **详细注释：** 关键代码行均附有详细注释，解释了其功能、背后的原理以及张量的维度变化，便于学习和理解。
*   **任务验证：** 通过一个简单的序列复制任务，直观地展示了模型学习序列到序列映射的能力。

##  模型架构

本项目严格按照原论文实现了以下所有关键组件：

*   **输入部分 (Input)**
    *   词嵌入 (Token Embedding)
    *   位置编码 (Positional Encoding)
*   **编码器 (Encoder)**
    *   多头自注意力机制 (Multi-Head Self-Attention)
    *   残差连接与层归一化 (Add & Norm)
    *   逐位置前馈网络 (Position-wise Feed-Forward Network)
*   **解码器 (Decoder)**
    *   带掩码的多头自注意力机制 (Masked Multi-Head Self-Attention)
    *   编码器-解码器注意力机制 (Encoder-Decoder Attention)
*   **输出部分 (Output)**
    *   线性层 (Linear Layer)
    *   Softmax

##  快速开始

### 1. 环境配置

请确保你已经安装了 Python 和 PyTorch。

```bash
# 克隆本项目
git clone  https://github.com/dz1104/my_transformer.git
cd my_transformer

# 安装依赖
pip install torch
```

### 2. 模型训练

运行以下命令开始在动态生成的序列复制任务上训练模型。脚本会进行10个周期的训练，并在结束后保存模型权重到 `transformer_copy_model.pth`。

```bash
python train.py
```


### 3. 模型推理

训练完成后，可以运行以下命令来加载已训练好的模型，并测试其序列复制的能力。

```bash
python inference.py
```

脚本会使用一个预设的序列作为输入，并打印出模型的贪心解码输出：

```
Input sequence: [[ 1  2  3  4  5  6  7  8  9 10]]
Model output:   [[ 1  2  3  4  5  6  7  8  9 10]]
```


## 📂 文件结构

```
my_transformer/
├── model.py           # 所有模型组件 (Encoder, Decoder, Attention等)
├── data.py            # 数据生成器和批次处理逻辑
├── train.py           # 模型训练主脚本
├── inference.py       # 模型推理和测试脚本
└── README.md          # 本说明文件
```


*本项目作为个人学习和面试准备的成果，欢迎交流与讨论。*
