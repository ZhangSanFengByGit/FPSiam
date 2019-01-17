# FPSiam
Feature Aggregation based Siamese Network for visual tracking

针对于目标跟踪任务的：基于特征融合技术的孪生网络

采用基于动态特异性卷机块算法，对深层次CNN特征进行融合kernel预测操作
用所采集到的融合kernel对目标帧特帧进行空间可变性卷机操作，进而完成特征融合
对融合后的优质特征进行经典的RPN目标检测算法的变形算法，得到target的bbox信息输出


框架：pytorch 1.0.0 with CUDA 9.0/ CUDNN
python：2.7
参考文献：
Low-Latency Video Semantic Segmentation
High Performance Visual Tracking with Siamese Region Proposal Network
Fully-convolutional siamese networks for object tracking
Siamese instance search for tracking
等
