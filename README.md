# FPSiam
Feature Aggregation based Siamese Network for visual tracking

针对于目标跟踪任务的：基于特征融合技术的孪生网络

简介：本次课题算法采用基于空间位置特异性卷机块算法，首先是利用经典的Siamese网络框架提取出视频序列中相邻两帧，及序列首帧图像的CNN特征，然后对深层CNN特征进行融合kernel预测操作。用所采集到的融合kernel对目标帧特帧进行空间可变性卷机操作，得到可传播的特征量。利用适应性权重计算，将该传播特征与原帧特征相融合，进而完成特征融合的目标。最后，对融合后的优质特征进行经典的RPN目标检测算法的变形算法，得到target的位移，尺度变化信息输出。网络整体采用端到端批训练，且采用Pytorch框架结合CUDA技术进行实验。


框架：pytorch 1.0.0 with CUDA 9.0/ CUDNN

python：2.7

参考文献：

Low-Latency Video Semantic Segmentation

High Performance Visual Tracking with Siamese Region Proposal Network

Fully-convolutional siamese networks for object tracking

Siamese instance search for tracking

等
