"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn

from .base_model import BaseModel
from .blocks import FeatureFusionBlock, Interpolate, _make_encoder
import matplotlib.pyplot as plt
import numpy as np
import imageio

def show_feature_map(feature_map):
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, 128):
        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index-1], cmap='gray')
        plt.axis('off')
        imageio.imsave('/data/jyx/dbd/MiDaS/testSideFeature/layer1/'+str(index)+".png", feature_map[index-1])
    plt.show()


class MidasNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, non_negative=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet, self).__init__()

        use_pretrained = False if path is None else True

        self.pretrained, self.scratch = _make_encoder(backbone="resnext101_wsl", features=features, use_pretrained=use_pretrained)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        if path:
            self.load(path)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        # encoder
        layer_1 = self.pretrained.layer1(x) #6, 256, 80, 80
        layer_2 = self.pretrained.layer2(layer_1) #6, 512, 40, 40
        layer_3 = self.pretrained.layer3(layer_2) #6, 1024, 20, 20
        layer_4 = self.pretrained.layer4(layer_3) #6, 2048, 10, 10

        # reduce channel to 256
        layer_1_rn = self.scratch.layer1_rn(layer_1) #6, 256, 80, 80
        layer_2_rn = self.scratch.layer2_rn(layer_2) #6, 256, 40, 40
        layer_3_rn = self.scratch.layer3_rn(layer_3) #6, 256, 20, 20
        layer_4_rn = self.scratch.layer4_rn(layer_4) #6, 256, 10, 10

        feat = []
        feat.append(layer_4)
        feat.append(layer_3)
        feat.append(layer_2)
        feat.append(layer_1)

        # aggregation
        path_4 = self.scratch.refinenet4(layer_4_rn) #6, 256, 20, 20
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn) #6, 256, 40, 40
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn) #6, 256, 80, 80
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn) #6, 256, 160, 160

        out = self.scratch.output_conv(path_1)

        return feat,torch.squeeze(out, dim=1)
