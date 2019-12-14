# Awesome_CV_Paper


Awesome for Paper Reading

A curated list of awesome Paper resources in Deep learning and computer vision.

>To complement or correct it, please send a pull request.

![image](https://github.com/iDeepwise/Awesome_CV_Research/blob/master/Detection/logo.png)

# Overview

Review

Segmentation

Detection

Reconstruction

Classification

Registration

Others

***

# Detection -- Different NMS Variants 

**1. NMS：Non-Maximum Suppression.**

  Paper: http://arxiv.org/abs/1411.5309
  
  Reference: https://www.coursera.org/lecture/convolutional-neural-networks/non-max-suppression-dvrjH

**2. Soft-NMS：Improving Object Detection With One Line of Code.**

  Paper: https://arxiv.org/abs/1704.04503
  
  Code: https://github.com/bharatsingh430/soft-nms

**3. Softer-NMS: Rethinking Bounding Box Regression for Accurate Object Detection.**

  Paper: https://arxiv.org/abs/1809.08545v1
  
  Code: https://github.com/yihui-he/softer-NMS

**4. IoU guided NMS：Acquisition of Localization Confidence for Accurate Object Detection.**
  
  Paper: https://eccv2018.org/openaccess/content_ECCV_2018/papers/Borui_Jiang_Acquisition_of_Localization_ECCV_2018_paper.pdf
  
  Reference: https://blog.csdn.net/qq_41648043/article/details/82716133
  
  Code: https://github.com/vacancy/PreciseRoIPooling

**5. ConvNMS：A Convnet for Non-maximum Suppression.**

  Paper: https://arxiv.org/abs/1511.06437

**6. Pure NMS Network：Learning non-maximum suppression.**

  Paper: https://arxiv.org/abs/1705.02950
  
  Code: https://github.com/hosang/gossipnet

**7. Yes-Net: An effective Detector Based on Global Information.**

  Paper: https://arxiv.org/abs/1706.09180
  
**8. Pairwise-NMS: Learning Pairwise Relationship for Multi-object Detection in Crowded Scenes**

  Paper:  https://arxiv.org/abs/1901.03796

**9. Relation Module: Relation Networks for Object Detection.**

  Paper: https://arxiv.org/abs/1711.11575
  
  Reference: https://www.zhihu.com/question/263428989
  
  Code: https://github.com/msracver/Relation-Networks-for-Object-Detection
  

# Detection -- Scale Variation & Feature Concat

**1. SNIP：An Analysis of Scale Invariance in Object Detection.**

  Paper: https://arxiv.org/abs/1711.08189
  
  Code: https://github.com/bharatsingh430/snip

**2. SNIPER: Efficient Multi-Scale Training.**

  Paper: https://arxiv.org/abs/1805.09300
  
  Code: https://github.com/mahyarnajibi/SNIPER

**3. HyperNet: Towards Accurate Region Proposal Generation and Joint Object Detection.**

  Paper: https://arxiv.org/abs/1604.00600

**4. PAnet：Path Aggregation Network for Instance Segmentation.**
  
  Paper: https://arxiv.org/abs/1803.01534
  
  Code: https://github.com/ShuLiu1993/PANet

**5. Scale-Aware Face Detection.**

  Paper: https://arxiv.org/abs/1706.09876

**6. Dynamic Zoom-in Network for Fast Object Detection in Large Images.**

  Paper: https://arxiv.org/abs/1711.05187

**7. Zoom Out-and-In Network with Map Attention Decision for Region Proposal and Object Detection.**

  Paper: https://arxiv.org/abs/1709.04347

**9. Scale-Aware Trident Networks for Object Detection.**

  Paper: https://arxiv.org/abs/1901.01892
  
  Code: https://github.com/TuSimple/simpledet/tree/master/models/tridentnet


# Attention Variants -- Detection & Segmentation 

**1. Attention is all you need.**

  Paper: https://arxiv.org/abs/1706.03762
  
  Reference: https://zhuanlan.zhihu.com/p/48508221

**2. Non-local Neural Networks.**

  Paper: https://arxiv.org/abs/1711.07971
  
  Reference: https://hellozhaozheng.github.io/z_post/计算机视觉-NonLocal-CVPR2018/
  
  Code: https://github.com/facebookresearch/video-nonlocal-net

**3. Relation networks for object detection.**

  Paper: https://arxiv.org/abs/1711.11575
  
  Code: https://github.com/msracver/Relation-Networks-for-Object-Detection

**4. Residual attention network for image classification.**
  
  Paper: https://arxiv.org/abs/1704.06904
  
  Reference: https://www.youtube.com/watch?v=Deq1BGTHIPA
  
  Code: https://github.com/fwang91/residual-attention-network

**5. OCNet: Object Context Network for Scene Parsing.**

  Paper: https://arxiv.org/abs/1809.00916
  
  Code: https://github.com/PkuRainBow/OCNet.pytorch

**6. Dual Attention Network for Scene Segmentation.**

  Paper: https://arxiv.org/abs/1809.02983
  
  Code: https://github.com/junfu1115/DANet

**7. Self-Attention Generative Adversarial Networks.**

  Paper: https://arxiv.org/abs/1805.08318
  
  Code: https://github.com/heykeetae/Self-Attention-GAN
  
**8. Context Encoding for Semantic Segmentation**

  Paper:  https://arxiv.org/abs/1803.08904
  
  Reference: https://hangzhang.org/PyTorch-Encoding/experiments/segmentation.html
  
  Code:  https://github.com/zhanghang1989/PyTorch-Encoding

**9. Squeeze-and-Excitation Networks.**

  Paper: https://arxiv.org/abs/1711.11575
  
  Reference: https://zhuanlan.zhihu.com/p/32702350
  
  Code: https://github.com/hujie-frank/SENet
  
  
# Detection -- Anchor Free

**1. DenseBox：Unifying Landmark Localization with End to End Object Detection.**

  Paper: https://arxiv.org/pdf/1509.04874.pdf
  
  Reference: https://blog.csdn.net/App_12062011/article/details/77941343
  
**2. CornerNet: Keypoint Triplets for Object Detection.**

  Paper: https://arxiv.org/pdf/1808.01244.pdf
  
  Reference: https://zhuanlan.zhihu.com/p/41825737
  
  Code:https://github.com/princeton-vl/CornerNet

**3. ExtremeNet: Bottom-up Object Detection by Grouping Extreme and Center Points.**

  Paper: https://arxiv.org/pdf/1901.08043.pdf
  
  Code:https://github.com/xingyizhou/ExtremeNet
  
**4. CenterNet：Objects as Points.**

  Paper: https://arxiv.org/pdf/1904.07850.pdf
  
  Reference: https://www.infoq.cn/article/XUDiNPviWhHhvr6x_oMv
  
  Code:https://github.com/xingyizhou/CenterNet
  
**5. CenterNet: Keypoint Triplets for Object Detection.**

  Paper: https://arxiv.org/pdf/1904.08189.pdf
  
  Reference: https://zhuanlan.zhihu.com/p/66326413 
  
  Code:https://github.com/Duankaiwen/CenterNet

**6. FCOS: Fully Convolutional One-Stage Object Detection.**

  Paper: https://arxiv.org/abs/1904.01355.pdf
  
  Code:https://github.com/tianzhi0549/FCOS
  
  

