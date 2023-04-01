# [ICME 2023] Depth and DOF Cues Make Better Defocus Blur Detection

This repository contains the official Pytorch implementation of our ICME 2023 paper.

## Dataset
We collect a new dataset EBD for testing. Please download from this link if you plan to use it. [EBD dataset](https://pan.baidu.com/s/1mL9gYu-2tnKR4lQoB3jAOA?pwd=cqoz) 

## Pretrained Models
* We provide pretrained models of our DFFNet and D-DFFNet using three different training data.

 Traing Datasets | CUHK-TR-1  | CUHK-TR-1&DUT-TR | CUHK-TR-2&DUT-TR
 ---- | ----- | ------  |  ------
 DFFNet  |  [DFFNet](https://drive.google.com/file/d/10UhCeEEl7OYjwzHZZByPbe6WpxqnWtbR/view?usp=sharing)  | [DFFNet](https://drive.google.com/file/d/1GN-HZ_lSZg25iX8d0fEx0qFZKXjCEhwc/view?usp=sharing) | [DFFNet](https://drive.google.com/file/d/1qiSoClOHZ9jV6qcaTOr6-QSh0ySwCsaX/view?usp=sharing)
 D-DFFNet  | [D-DFFNet](https://drive.google.com/file/d/1BRWXt8xphFv6AQDZwan4umm9EX3X4E2x/view?usp=sharing) | [D-DFFNet](https://drive.google.com/file/d/1NmrA8amNLkI-QPIq_N0Ti36Su7ungMo1/view?usp=share_link)  |[D-DFFNet](https://drive.google.com/file/d/1hU81jbHG-55HmaSgb0_GXOKhRC-enL8Q/view?usp=sharing) 
 * Pretrained model for depth model: [midas_v21-f6b98070.pt](https://drive.google.com/file/d/1puxWdaUYayZhjf9WGGCwhMfkapl71eeB/view?usp=sharing)


## Results
We provide results on four test datasets. Since we use three different training data for fair comparison with previous works, here we provide all results related to the three different training data.

Traing Datasets | CUHK-TR-1  | CUHK-TR-1&DUT-TR | CUHK-TR-2&DUT-TR
 ---- | ----- | ------  |  ------
 Results  |[Results](https://drive.google.com/file/d/1ncVmYz26pmu_yS_J0Jbe15yvxJDQU3cS/view?usp=sharing) |[Results](https://drive.google.com/file/d/1tLC7zkD2oFu7hsjHA0w615rKqX6bgtAg/view?usp=share_link)  |[Results](https://drive.google.com/file/d/1nfH4l-E2yZuxTd-qh3sRHtYcCZPJW_tR/view?usp=sharing)


## Code

### Dependences
* Pytorch
* OpenCV
* Numpy
* PIL
* glob

### Training
* Set dataset paths in data.py.
* Download depth pretrained model and put it to /D-DFFNet/depth_pretrained/.
* Set save_path in train_single.py and train_kd.py to your saving path.
* Training in stage 1:
`python train_single.py`
* Training in stage 2:
`python train_kd.py`


### Testing
* Set path in test.sh to the pretrained model.
* `bash test.sh`

## Citation

```
@inproceedings{jin2023depth,
  title={Depth and DOF Cues Make Better Defocus Blur Detection},
  author={Jin, Yuxin and Qian, Ming and Xiong, Jincheng and Xue, Nan and Xia, Gui-song},
  booktitle={ICME},
  year={2023},
}
```

## Contact
If you have any question or suggestion, please contact (Yuxin Jin jinyuxin@whu.edu.cn).


