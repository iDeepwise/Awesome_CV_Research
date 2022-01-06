# Transformers in Vision: A Survey


A curated list of Transformers Paper resources in Deep learning and computer vision.

>To complement or correct it, please send a pull request.

![image](https://github.com/iDeepwise/Awesome_CV_Research/tree/master/Related/iDeepwise.png)

# Transformers

1、[CV] Vision Transformer with Deformable Attention
Z Xia, X Pan, S Song, L E Li, G Huang
[Tsinghua University & AWS AI]

基于可变形注意力的视觉Transformer。Transformer最近在各种视觉任务上表现出了卓越的性能。大的、有时甚至是全局性的感受野赋予了Transformer模型比CNN对应模型更高的表示能力。然而，简单地扩大感受野也引起了一些担忧。一方面，使用稠密的注意力，例如在ViT中，会导致过多的内存和计算成本，而且特征可能会受到超出感兴趣区域的不相关部分的影响。另一方面，在PVT或Swin Transformer中采用的稀疏注意是与数据无关的，可能会限制对长程关系的建模能力。为缓解这些问题，本文提出一种新的可变形自注意力模块，其中自注意力中的键和值对的位置是以一种依赖于数据的方式选择的。这种灵活的方案使自注意力模块能专注于相关区域，并捕获更多的信息特征。在此基础上，提出了Deformable Attention Transformer，一种具有可变形注意力的通用骨干模型，用于图像分类和稠密预测任务。广泛的实验表明，所提出模型在综合基准上取得了持续改进的结果。

2、[CV] PyramidTNT: Improved Transformer-in-Transformer Baselines with Pyramid Architecture
K Han, J Guo, Y Tang, Y Wang
[Huawei Noah’s Ark Lab]

PyramidTNT：用金字塔架构改进Transformer-in-Transformer基线。Transformer网络在计算机视觉任务方面取得了巨大的进展。Transformer-in-Transformer(TNT)架构利用内部Transformer和外部Transformer来提取局部和全局表示。本文通过引入两种先进的设计提出了新的TNT基线。1）金字塔结构，和2）卷积干。新的"PyramidTNT"通过建立分层表示大大改进了原TNT。PyramidTNT实现了比之前最先进的视觉Transformer(如Swin Transformer)更好的性能。




