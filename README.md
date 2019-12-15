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
  
  
#  Lightweight Network Structure

**1. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size.**

  Paper: https://arxiv.org/abs/1602.07360

  Code: https://github.com/forresti/SqueezeNet

**2. Densely Connected Convolutional Networks.**

  Paper: https://arxiv.org/pdf/1608.06993.pdf
  
  Reference: https://blog.csdn.net/u014380165/article/details/75142664
  
  Code: https://github.com/liuzhuang13/DenseNet

**3. Xception: Deep Learning with Depthwise Separable Convolutions.**

  Paper: https://arxiv.org/abs/1610.02357
  
  Reference: https://blog.csdn.net/u014380165/article/details/75142710
  
  Code: https://github.com/yihui-he/Xception-caffe

**4. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.**

  Paper: https://arxiv.org/abs/1704.04861

  Reference: https://blog.csdn.net/qq_31914683/article/details/79330343
  
  Code: https://github.com/Zehaos/MobileNet
        https://github.com/shicai/MobileNet-Caffe
  
**5. ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices.**

  Paper: https://arxiv.org/abs/1707.01083
  
  Reference: https://blog.csdn.net/u014380165/article/details/75137111
  
  Code: https://github.com/farmingyard/ShuffleNet

**6. NASNet：Learning Transferable Architectures for Scalable Image Recognition.**

  Paper: https://arxiv.org/abs/1707.07012
  
  Reference: https://blog.csdn.net/xjz18298268521/article/details/79079008
             https://zhuanlan.zhihu.com/p/52616166
  
  Code: https://github.com/yeephycho/nasnet-tensorflow

**7. CondenseNet: An Efficient DenseNet using Learned Group Convolutions.**

  Paper: https://arxiv.org/abs/1711.09224
  
  Reference: https://blog.csdn.net/u014380165/article/details/78747711
  
  Code: https://github.com/ShichenLiu/CondenseNet

**8. MobileNetV2: Inverted Residuals and Linear Bottlenecks.**

  Paper: https://arxiv.org/abs/1801.04381
  
  Reference: https://www.cnblogs.com/hejunlin1992/p/9395345.html
  
  Code: https://github.com/xiaochus/MobileNetV2

**9. ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design.**

  Paper: https://arxiv.org/abs/1807.11164
  
  Reference: https://zhuanlan.zhihu.com/p/48261931
  
  Code: https://github.com/farmingyard/ShuffleNet

**10. MnasNet: Platform-Aware Neural Architecture Search for Mobile.**

  Paper: https://arxiv.org/abs/1807.11626
  
  Reference: https://zhuanlan.zhihu.com/p/42474017
  
  Code: https://github.com/AnjieZheng/MnasNet-PyTorch

**11. ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware.**

  Paper: https://arxiv.org/abs/1812.00332
  
  Reference: https://www.cnblogs.com/wangxiaocvpr/p/10559377.html
  
  Code: https://github.com/MIT-HAN-LAB/ProxylessNAS

**12. Searching for MobileNetV3.**

  Paper: https://arxiv.org/abs/1905.02244v2
  
  Reference: https://blog.csdn.net/sinat_37532065/article/details/90813655
  
  Code: https://github.com/xiaolai-sqlai/mobilenetv3
  
**13. MixConv: Mixed Depthwise Convolutional Kernels.**

  Paper: https://arxiv.org/abs/1907.09595
  
  Reference: https://zhuanlan.zhihu.com/p/75242090
  
  Code: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet

**14. MoGA: Searching Beyond MobileNetV3.**

  Paper: https://arxiv.org/pdf/1908.01314.pdf
  
  Reference: https://zhuanlan.zhihu.com/p/76909380
  
  Code: https://github.com/xiaomi-automl/MoGA