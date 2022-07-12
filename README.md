# STDN
A False Alarm Controllable Detection Method Based on CNN for Sea-surface Small Targets

It is a compelling task to detect sea-surface small targets in the background of strong sea clutter. Traditional detection methods usually suffer from poor detection performance and a high probability of false alarm (PFA).

A PFA-controllable and feature-based detection method is desined in this code, The diagram of the whole method is shown in Fig. 1. In summary, the optimal weight of the training set is shared with the test set, and the threshold T is also selected according to the given PFA. The received signal is first converted into a TFM by the STFT, which is also employed as the input of the STDN. Then the STDN outputs a clutter probability P directly in the test stage via the optimal weight and softmax function, and the detection result under the given PFA could be obtained by comparing P with T.

For more details, you could read our paper [1]. In this project, STDN.py is a PyTorch implementation code, and you could easily transfer this network to your research by "from STDN import stdn-18". The required environments include "torch" and "math". This code refers to some parts of ACNet(https://github.com/DingXiaoH/ACNet)

[1] Qizhe Qu, Yong-Liang Wang, Weijian Liu, Binbin Li, "A False Alarm Controllable Detection Method Based on CNN for Sea-surface Small Targets," IEEE Geoscience and Remote Sensing Letters, accepted, 2022.
