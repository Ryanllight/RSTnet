# RSTnet：革新医学图像分割的Swin Transformer网络

## 🚀 模型概述

RSTnet，一个为医学图像分割领域注入全新活力的卓越模型，巧妙融合了Swin Transformer的宏观洞察力与卷积神经网络的微观捕捉力。我们独创的双编码器架构，如同鹰眼与手术刀的完美结合：Swin Transformer编码器以其卓越的全局上下文感知能力，精准捕捉图像的深层结构；而并行运行的卷积编码器，则以其对局部细节无与伦比的精细刻画，确保每一像素的价值都被充分挖掘。更令人振奋的是，RSTnet在Swin编码器中融入了**CBAM（Convolutional Block Attention Module）**，赋予模型“视觉焦点”，使其能够智能地聚焦于最具判别力的特征。同时，在解码器的关键跳跃连接处，我们引入了**RCASC（Residual Channel Attention Skip Connection）**模块，这不仅优化了编码器与解码器之间的信息流，更通过残差连接和通道注意力机制，实现了特征的深度融合与高效传递，从而在复杂多变的医学图像中，展现出前所未有的分割精度与鲁棒性。

## ✨ 架构亮点：RSTnet的非凡之处

*   **双重编码，洞察全局与细节**：Swin Transformer编码器与卷积编码器珠联璧合，前者擅长宏观结构理解，后者精于微观纹理捕捉，共同构建起对医学图像全面而深刻的理解。
*   **CBAM赋能，智能聚焦关键特征**：在Swin编码器中嵌入CBAM模块，让模型拥有“智慧之眼”，能够动态调整对通道和空间信息的关注度，确保模型始终聚焦于对分割任务至关重要的区域。
*   **RCASC革新，跳跃连接的艺术**：独创的RCASC模块，以其精妙的残差通道注意力机制，彻底革新了传统跳跃连接。它不仅高效融合了来自不同编码器的特征，更在融合过程中实现了信息的智能筛选与增强，为解码器提供了最纯粹、最丰富的上下文信息。
*   **卷积瓶颈，深层特征的精炼**：在双编码器之间设置的卷积瓶颈层，如同一个智能过滤器，对提取到的深层特征进行进一步的精炼与整合，为后续的解码过程奠定坚实基础。

## 🛠️ 环境配置

为了让RSTnet在您的环境中尽情驰骋，请确保您的Python版本为3.7或更高。随后，只需一条简单的命令，即可安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

## 📊 数据准备

RSTnet的数据准备流程与经典的Swin-Unet一脉相承。请您参考原始Swin-Unet项目中的详细指南，为RSTnet准备高质量的训练数据。

## 🚀 训练与测试：释放RSTnet的强大潜能

### 训练

启动RSTnet的训练之旅，您只需在命令行中执行以下指令。请根据您的实际情况，调整数据集路径、输出目录及其他超参数，以期达到最佳性能：

```bash
python train.py --dataset Synapse --cfg your_cfg --root_path your_DATA_DIR --max_epochs 150 --output_dir your_OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24 --model_name RSTnet
```

### 测试

当RSTnet完成训练，您可以通过以下命令对其性能进行全面评估。同样，请确保路径和参数设置正确：

```bash
python test.py --dataset Synapse --cfg your_cfg --is_saveni --volume_path your_DATA_DIR --output_dir your_OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24 --model_name RSTnet
```

**请注意**：`your_DATA_DIR` 和 `your_OUT_DIR` 需要替换为您的实际数据和输出目录。`--model_name RSTnet` 参数是激活我们RSTnet模型的关键。

## 🌟 RSTnet的卓越性能

在**Synapse**等权威医学图像数据集上，RSTnet展现出了令人瞩目的卓越性能。通过我们精心设计的双编码器、CBAM和RCASC模块，RSTnet在分割精度、边界清晰度以及对复杂病灶的识别能力上，均超越了传统模型，为医学图像分析带来了革命性的突破。其强大的特征学习和融合能力，使得RSTnet在面对各种挑战性的医学影像时，都能提供精准可靠的分割结果，为临床诊断和治疗提供强有力的支持。

## 📝 致谢

RSTnet的实现基于Swin Transformer和Swin-Unet的优秀基础。特别感谢Swin Transformer和Swin-Unet的作者，以及开源社区提供的宝贵资源。


## 📈 期待您的反馈
我们期待您的反馈和合作，以共同打造一个更 superior 的模型。


