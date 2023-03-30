# [ICME 2023] Depth and DOF Cues Make Better Defocus Blur Detection

This repository contains the official Pytorch implementation of our ICME 2023 paper.

## Dataset
We collect a new dataset EBD for testing. Please download from this link if you plan to use it. [EBD dataset](https://pan.baidu.com/s/1mL9gYu-2tnKR4lQoB3jAOA?pwd=cqoz) 

## Pretrained Models
* We provide pretrained models of our DFFNet and D-DFFNet using three different training data.

 Traing Datasets | CUHK-TR-1  | CUHK-TR-1&DUT-TR | CUHK-TR-2&DUT-TR
 ---- | ----- | ------  |  ------
 DFFNet  |  [DFFNet](https://pan.baidu.com/s/1Sd_TDM92-iJ6gZLW15ANaw?pwd=90em)  | [DFFNet](https://pan.baidu.com/s/1AO06sJOiWojS_MFiT58DcA?pwd=ns0z) | [DFFNet]()
 D-DFFNet  | [D-DFFNet](https://pan.baidu.com/s/1coVPb2OtjJ8FnarqkWk7Cw?pwd=r6c5) | [D-DFFNet](https://pan.baidu.com/s/1fw7D6DzNbTNjeLwS-H72UQ?pwd=u7fs)   |[D-DFFNet](https://pan.baidu.com/s/1iKJED5w-obf6kiBBxhI0Ww?pwd=2y3i) 
 * Pretrained model for depth model: [midas_v21-f6b98070.pt](https://pan.baidu.com/s/1VA1yNifcZpy9bpTyygH0EA?pwd=dwo9)


## Results
We provide results on four test datasets. Since we use three different training data for fair comparison with previous works, here we provide all results related to the three different training data.

Traing Datasets | CUHK-TR-1  | CUHK-TR-1&DUT-TR | CUHK-TR-2&DUT-TR
 ---- | ----- | ------  |  ------
 Results  |[Results](https://drive.google.com/file/d/1ncVmYz26pmu_yS_J0Jbe15yvxJDQU3cS/view?usp=sharing)|[Results](https://pan.baidu.com/s/18M5f8oJDYeyy7cWQef05SA?pwd=c3v7)  |[Results](https://pan.baidu.com/s/1rT_rQY9ybOKAVJLfxlmVlQ?pwd=zf26)


## Code
### Dependences
* Pytorch
* OpenCV
* Numpy
* PIL
* glob



