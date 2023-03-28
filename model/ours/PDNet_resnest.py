import torch
import torch.nn as nn

from model.ours.RFB import RFB
from model.ours.utils import show_feature_map
from model.ours.blocks import SABlock,SEBlock
from model.ours.aggregation import aggregation,aggregation_4feat,aggregation_5feat,aggregation_4feat_side_loss,aggregation_4feat_side_loss_sa,aggregation_4feat_side_loss_sa_d,aggregation_basic,aggregation_4feat_side_loss_sa_ng

class PDnet_ResNest_3F(nn.Module):
    def __init__(self, channel=32):
        super(PDnet_ResNest_3F, self).__init__()
        print('model:PDnet_ResNest_3F')
        from model.ours.backbone.resnet import ResNest101
        self.resnest = ResNest101([3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False)
        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.agg1 = aggregation(channel)
        self.agg2 = aggregation(channel)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.initialize_weights()

    def forward(self, x):
        x = self.resnest.conv1(x)
        x = self.resnest.bn1(x)
        x = self.resnest.relu(x)

        x = self.resnest.maxpool(x) #[1, 128, 80, 80]
        x1 = self.resnest.layer1(x)  #[1, 256, 80, 80]
        x2 = self.resnest.layer2(x1)  #[1, 512, 40, 40]
        x3 = self.resnest.layer3_1(x2)  #[1, 1024, 20, 20]
        x4 = self.resnest.layer4_1(x3)  #[1, 2048, 10, 10]

        x2 = self.upsample1(self.rfb2_1(x2))  #[1, 32, 80, 80]
        x3 = self.upsample2(self.rfb3_1(x3))  #[1, 32, 40, 40]
        x4 = self.upsample3(self.rfb4_1(x4))  #[1, 32, 20, 20]

        feature_map = []
        feature_map.append(x4)
        feature_map.append(x3)
        feature_map.append(x2)
        out = self.agg1(x4, x3, x2)

        return self.upsample(out),feature_map

    def initialize_weights(self):
        # print('load pretrained for resnest')
        from resnest.torch import resnest101
        resnest101 = resnest101(pretrained=True)
        pretrained_dict = resnest101.state_dict()
        all_params = {}
        for k, v in self.resnest.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnest.state_dict().keys())
        self.resnest.load_state_dict(all_params)

class PDnet_Vgg_4F_SL_LA(nn.Module):
    def __init__(self, channel=32):
        super(PDnet_Vgg_4F_SL_LA, self).__init__()
        print('model:PDnet_Vgg_4F_SL_LA')
        from model.ours.backbone.vgg16 import VGGNet
        self.vgg = VGGNet()
        self.rfb2_1 = RFB(256, channel)
        self.rfb3_1 = RFB(512, channel)
        self.rfb4_1 = RFB(512, channel)
        self.rfb5_1 = RFB(128, channel)

        self.agg1 = aggregation_4feat_side_loss(channel)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        output = self.vgg(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        x2 = self.rfb5_1(x2)
        x3 = self.rfb2_1(x3)  #[1, 512, 80, 80]
        x4 = self.rfb3_1(x4)  #[1, 1024, 40, 40]
        x5 = self.rfb4_1(x5)  #[1, 2048, 20, 20]

        feature_map = []
        feature_map.append(x5)
        feature_map.append(x4)
        feature_map.append(x3)
        feature_map.append(x2)

        out,side = self.agg1(x5, x4, x3, x2)

        return self.upsample(out),feature_map,side

class PDnet_Vgg_3F(nn.Module):
    def __init__(self, channel=32):
        super(PDnet_Vgg_3F, self).__init__()
        print('model:PDnet_Vgg_3F')
        from model.ours.backbone.vgg16 import VGGNet
        self.vgg = VGGNet()
        self.rfb2_1 = RFB(256, channel)
        self.rfb3_1 = RFB(512, channel)
        self.rfb4_1 = RFB(512, channel)
        self.agg1 = aggregation(channel)
        self.agg2 = aggregation(channel)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        output = self.vgg(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        x2 = self.rfb2_1(x3)  #[1, 512, 80, 80]
        x3 = self.rfb3_1(x4)  #[1, 1024, 40, 40]
        x4 = self.rfb4_1(x5)  #[1, 2048, 20, 20]

        feature_map = []
        feature_map.append(x4)
        feature_map.append(x3)
        feature_map.append(x2)
        out = self.agg1(x4, x3, x2)

        return self.upsample(out),feature_map

class PDnet_Vgg_Res_3F(nn.Module):
    def __init__(self, channel=32):
        super(PDnet_Vgg_Res_3F, self).__init__()
        print('model:PDnet_Vgg_Res_3F')
        from model.ours.backbone.vgg16 import VGGNet
        from model.ours.backbone.resnet import ResNest101
        self.resnest = ResNest101([3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False)
        self.vgg = VGGNet()

        self.rfb2_1 = RFB(256+512, channel)
        self.rfb3_1 = RFB(512+1024, channel)
        self.rfb4_1 = RFB(512+2048, channel)
        self.agg1 = aggregation(channel)
        self.agg2 = aggregation(channel)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.initialize_weights()
    

    def forward(self, x):
        output = self.vgg(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        x = self.resnest.conv1(x)
        x = self.resnest.bn1(x)
        x = self.resnest.relu(x)
        x = self.resnest.maxpool(x)
        x11 = self.resnest.layer1(x)  #[1, 256, 80, 80]
        x22 = self.resnest.layer2(x11)  #[1, 512, 40, 40]

        x33 = self.resnest.layer3_1(x22)  #[1, 1024, 20, 20]
        x44 = self.resnest.layer4_1(x33)  #[1, 2048, 10, 10]

        x2 = self.rfb2_1(torch.cat((x3,self.upsample1(x22)),1))  #[1, 512, 80, 80]
        x3 = self.rfb3_1(torch.cat((x4,self.upsample1(x33)),1))  #[1, 1024, 40, 40]
        x4 = self.rfb4_1(torch.cat((x5,self.upsample1(x44)),1))  #[1, 2048, 20, 20]

        feature_map = []
        feature_map.append(x4)
        feature_map.append(x3)
        feature_map.append(x2)
        out = self.agg1(x4, x3, x2)

        return self.upsample(out),feature_map

    def initialize_weights(self):
        print('load pretrained for resnest')
        from resnest.torch import resnest101
        resnest101 = resnest101(pretrained=True)
        pretrained_dict = resnest101.state_dict()
        all_params = {}
        for k, v in self.resnest.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnest.state_dict().keys())
        self.resnest.load_state_dict(all_params)

class PDnet_ResNest_4F(nn.Module):
    def __init__(self, channel=32):
        super(PDnet_ResNest_4F, self).__init__()
        print('model:PDnet_ResNest_4F')
        from model.ours.backbone.resnet import ResNest101
        self.resnest = ResNest101([3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False)
        self.rfb1_1 = RFB(256, channel)
        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.agg1 = aggregation_4feat(channel)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.initialize_weights()

    def forward(self, x):
        x = self.resnest.conv1(x)
        x = self.resnest.bn1(x)
        x = self.resnest.relu(x)

        x = self.resnest.maxpool(x) #[1, 128, 80, 80]
        x1 = self.resnest.layer1(x)  #[1, 256, 80, 80]
        x2 = self.resnest.layer2(x1)  #[1, 512, 40, 40]
        x3 = self.resnest.layer3_1(x2)  #[1, 1024, 20, 20]
        x4 = self.resnest.layer4_1(x3)  #[1, 2048, 10, 10]

        feature_map = []
        feature_map.append(x4)
        feature_map.append(x3)
        feature_map.append(x2)
        feature_map.append(x1)

        x1 = self.rfb1_1(x1)  #[1, 32, 80, 80]
        x2 = self.rfb2_1(x2)  #[1, 32, 40, 40] 
        x3 = self.rfb3_1(x3)  #[1, 32, 20, 20]  
        x4 = self.rfb4_1(x4)  #[1, 32, 10, 10]

        out = self.agg1(x4, x3, x2, x1)
        return self.upsample(out),feature_map


    def initialize_weights(self):
        # print('load pretrained for resnest')
        from resnest.torch import resnest101
        resnest101 = resnest101(pretrained=True)
        pretrained_dict = resnest101.state_dict()
        all_params = {}
        for k, v in self.resnest.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnest.state_dict().keys())
        self.resnest.load_state_dict(all_params)

class PDnet_ResNest_5F(nn.Module):
    def __init__(self, channel=32):
        super(PDnet_ResNest_5F, self).__init__()
        print('model:PDnet_ResNest_5F')
        from model.ours.backbone.resnet import ResNest101
        self.resnest = ResNest101([3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False)
        self.rfb0_1 = RFB(128, channel)
        self.rfb1_1 = RFB(256, channel)
        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.agg1 = aggregation_5feat(channel)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.initialize_weights()

    def forward(self, x):
        x = self.resnest.conv1(x)
        x = self.resnest.bn1(x)
        x = self.resnest.relu(x)
        x = self.resnest.maxpool(x) #[1, 128, 80, 80]
        x1 = self.resnest.layer1(x)  #[1, 256, 80, 80]
        x2 = self.resnest.layer2(x1)  #[1, 512, 40, 40]

        x3 = self.resnest.layer3_1(x2)  #[1, 1024, 20, 20]
        x4 = self.resnest.layer4_1(x3)  #[1, 2048, 10, 10]

        x0 = self.rfb0_1(x)   #[1, 32, 80, 80]
        x1 = self.rfb1_1(x1)  #[1, 32, 80, 80]
        x2 = self.rfb2_1(x2)  #[1, 32, 40, 40] 
        x3 = self.rfb3_1(x3)  #[1, 32, 20, 20]
        x4 = self.rfb4_1(x4)  #[1, 32, 10, 10]

        feature_map = []
        feature_map.append(x4)
        feature_map.append(x3)
        feature_map.append(x2)
        out = self.agg1(x4, x3, x2, x1, x0)

        return self.upsample(out),feature_map

    def initialize_weights(self):
        # print('load pretrained for resnest')
        from resnest.torch import resnest101
        resnest101 = resnest101(pretrained=True)
        pretrained_dict = resnest101.state_dict()
        all_params = {}
        for k, v in self.resnest.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnest.state_dict().keys())
        self.resnest.load_state_dict(all_params)

class PDnet_ResNest_4F_SL(nn.Module):
    def __init__(self, channel=32):
        super(PDnet_ResNest_4F_SL, self).__init__()
        print('model:PDnet_ResNest_4F_SL')
        from model.ours.backbone.resnet import ResNest101
        self.resnest = ResNest101([3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False)
        self.rfb1_1 = RFB(256, channel)
        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.agg1 = aggregation_4feat_side_loss(channel)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.initialize_weights()

    def forward(self, x):
        x = self.resnest.conv1(x)
        x = self.resnest.bn1(x)
        x = self.resnest.relu(x)

        x = self.resnest.maxpool(x) #[1, 128, 80, 80]
        x1 = self.resnest.layer1(x)  #[1, 256, 80, 80]
        x2 = self.resnest.layer2(x1)  #[1, 512, 40, 40]
        x3 = self.resnest.layer3_1(x2)  #[1, 1024, 20, 20]
        x4 = self.resnest.layer4_1(x3)  #[1, 2048, 10, 10]

        feature_map = []
        feature_map.append(x4)
        feature_map.append(x3)
        feature_map.append(x2)
        feature_map.append(x1)

        x1 = self.rfb1_1(x1)  #[1, 32, 80, 80]
        x2 = self.rfb2_1(x2)  #[1, 32, 40, 40] 
        x3 = self.rfb3_1(x3)  #[1, 32, 20, 20]  
        x4 = self.rfb4_1(x4)  #[1, 32, 10, 10]

        out,side = self.agg1(x4, x3, x2, x1)

        return self.upsample(out),feature_map,side


    def initialize_weights(self):
        # print('load pretrained for resnest')
        from resnest.torch import resnest101
        resnest101 = resnest101(pretrained=True)
        pretrained_dict = resnest101.state_dict()
        all_params = {}
        for k, v in self.resnest.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnest.state_dict().keys())
        self.resnest.load_state_dict(all_params)

# DFFNet
class PDnet_ResNest_4F_SL_LA(nn.Module):
    def __init__(self, channel=32):
        super(PDnet_ResNest_4F_SL_LA, self).__init__()
        print('model:PDnet_ResNest_4F_SL_LA')
        from model.ours.backbone.resnet import ResNest101
        self.resnest = ResNest101([3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False)
        self.rfb1_1 = RFB(256, channel)
        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.agg1 = aggregation_4feat_side_loss_sa(channel)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.initialize_weights()

    def forward(self, x):
        x = self.resnest.conv1(x)
        x = self.resnest.bn1(x)
        x = self.resnest.relu(x)

        x = self.resnest.maxpool(x) #[1, 128, 80, 80]
        x1 = self.resnest.layer1(x)  #[1, 256, 80, 80]
        x2 = self.resnest.layer2(x1)  #[1, 512, 40, 40]
        x3 = self.resnest.layer3_1(x2)  #[1, 1024, 20, 20]
        x4 = self.resnest.layer4_1(x3)  #[1, 2048, 10, 10]

        feature_map = []
        feature_map.append(x4)
        feature_map.append(x3)
        feature_map.append(x2)
        feature_map.append(x1)

        x1 = self.rfb1_1(x1)  #[1, 32, 80, 80]
        x2 = self.rfb2_1(x2)  #[1, 32, 40, 40] 
        x3 = self.rfb3_1(x3)  #[1, 32, 20, 20]  
        x4 = self.rfb4_1(x4)  #[1, 32, 10, 10]

        out,side = self.agg1(x4, x3, x2, x1)
        return self.upsample(out),feature_map,side
        # return self.upsample(out),side



    def initialize_weights(self):
        # print('load pretrained for resnest')
        from resnest.torch import resnest101
        resnest101 = resnest101(pretrained=True)
        pretrained_dict = resnest101.state_dict()
        all_params = {}
        for k, v in self.resnest.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnest.state_dict().keys())
        self.resnest.load_state_dict(all_params)

class PDnet_ResNext_4F_SL_LA(nn.Module):
    def __init__(self, channel=32):
        super(PDnet_ResNext_4F_SL_LA, self).__init__()
        print('model:PDnet_ResNext_4F_SL_LA')
        from model.ours.backbone.resnext101.resnext101 import ResNeXt101
        self.resnext = ResNeXt101()
        self.rfb1_1 = RFB(256, channel)
        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.agg1 = aggregation_4feat_side_loss_sa(channel)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.resnext.layer0(x) #[1, 128, 80, 80]
        x1 = self.resnext.layer1(x)  #[1, 256, 80, 80]
        x2 = self.resnext.layer2(x1)  #[1, 512, 40, 40]
        x3 = self.resnext.layer3(x2)  #[1, 1024, 20, 20]
        x4 = self.resnext.layer4(x3)  #[1, 2048, 10, 10]

        feature_map = []
        feature_map.append(x4)
        feature_map.append(x3)
        feature_map.append(x2)
        feature_map.append(x1)

        x1 = self.rfb1_1(x1)  #[1, 32, 80, 80]
        x2 = self.rfb2_1(x2)  #[1, 32, 40, 40] 
        x3 = self.rfb3_1(x3)  #[1, 32, 20, 20]  
        x4 = self.rfb4_1(x4)  #[1, 32, 10, 10]

        out,side = self.agg1(x4, x3, x2, x1)

        return self.upsample(out),feature_map,side

class PDnet_Res2Net_4F_SL_LA(nn.Module):
    def __init__(self, channel=32):
        super(PDnet_Res2Net_4F_SL_LA, self).__init__()
        print('model:PDnet_Res2Net_4F_SL_LA')
        from model.ours.backbone.res2net_v1b import res2net101_v1b
        self.res2net = res2net101_v1b(pretrained=True)
        self.rfb1_1 = RFB(256, channel)
        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.agg1 = aggregation_4feat_side_loss_sa(channel)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.res2net.conv1(x)
        x = self.res2net.bn1(x)
        x = self.res2net.relu(x)
        x = self.res2net.maxpool(x) #[1, 128, 80, 80]

        x1 = self.res2net.layer1(x)  #[1, 256, 80, 80]
        x2 = self.res2net.layer2(x1)  #[1, 512, 40, 40]
        x3 = self.res2net.layer3(x2)  #[1, 1024, 20, 20]
        x4 = self.res2net.layer4(x3)  #[1, 2048, 10, 10]

        feature_map = []
        feature_map.append(x4)
        feature_map.append(x3)
        feature_map.append(x2)
        feature_map.append(x1)

        x1 = self.rfb1_1(x1)  #[1, 32, 80, 80]
        x2 = self.rfb2_1(x2)  #[1, 32, 40, 40] 
        x3 = self.rfb3_1(x3)  #[1, 32, 20, 20]  
        x4 = self.rfb4_1(x4)  #[1, 32, 10, 10]

        out,side = self.agg1(x4, x3, x2, x1)
        return self.upsample(out),feature_map,side

class PDnet_ResNest_4F_SL_LA_D(nn.Module):

    def __init__(self, channel=32):
        super(PDnet_ResNest_4F_SL_LA_D, self).__init__()
        print('model:PDnet_ResNest_4F_SL_LA_D')
        from model.ours.backbone.resnet import ResNest101
        self.resnest = ResNest101([3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False)
        self.rfb1_1 = RFB(256, channel)
        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.agg1 = aggregation_4feat_side_loss_sa_d(channel)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.initialize_weights()

    def forward(self, x):
        x = self.resnest.conv1(x)
        x = self.resnest.bn1(x)
        x = self.resnest.relu(x)

        x = self.resnest.maxpool(x) #[1, 128, 80, 80]
        x1 = self.resnest.layer1(x)  #[1, 256, 80, 80]
        x2 = self.resnest.layer2(x1)  #[1, 512, 40, 40]
        x3 = self.resnest.layer3_1(x2)  #[1, 1024, 20, 20]
        x4 = self.resnest.layer4_1(x3)  #[1, 2048, 10, 10]

        feature_map = []
        feature_map.append(x4)
        feature_map.append(x3)
        feature_map.append(x2)
        feature_map.append(x1)

        x1 = self.rfb1_1(x1)  #[1, 32, 80, 80]
        x2 = self.rfb2_1(x2)  #[1, 32, 40, 40] 
        x3 = self.rfb3_1(x3)  #[1, 32, 20, 20]  
        x4 = self.rfb4_1(x4)  #[1, 32, 10, 10]

        out,side,depth = self.agg1(x4, x3, x2, x1)
        return self.upsample(out),feature_map,side,depth


    def initialize_weights(self):
        # print('load pretrained for resnest')
        from resnest.torch import resnest101
        resnest101 = resnest101(pretrained=True)
        pretrained_dict = resnest101.state_dict()
        all_params = {}
        for k, v in self.resnest.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnest.state_dict().keys())
        self.resnest.load_state_dict(all_params)

class PDnet_ResNest_4F_SL_LA_NG(nn.Module):
    def __init__(self, channel=32):
        super(PDnet_ResNest_4F_SL_LA_NG, self).__init__()
        print('model:PDnet_ResNest_4F_SL_LA_NG')
        from model.ours.backbone.resnet import ResNest101
        self.resnest = ResNest101([3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False)
        self.rfb1_1 = RFB(256, channel)
        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.agg1 = aggregation_4feat_side_loss_sa_ng(channel)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.initialize_weights()

    def forward(self, x):
        x = self.resnest.conv1(x)
        x = self.resnest.bn1(x)
        x = self.resnest.relu(x)

        x = self.resnest.maxpool(x) #[1, 128, 80, 80]
        x1 = self.resnest.layer1(x)  #[1, 256, 80, 80]
        x2 = self.resnest.layer2(x1)  #[1, 512, 40, 40]
        x3 = self.resnest.layer3_1(x2)  #[1, 1024, 20, 20]
        x4 = self.resnest.layer4_1(x3)  #[1, 2048, 10, 10]

        feature_map = []
        feature_map.append(x4)
        feature_map.append(x3)
        feature_map.append(x2)
        feature_map.append(x1)

        x1 = self.rfb1_1(x1)  #[1, 32, 80, 80]
        x2 = self.rfb2_1(x2)  #[1, 32, 40, 40] 
        x3 = self.rfb3_1(x3)  #[1, 32, 20, 20]  
        x4 = self.rfb4_1(x4)  #[1, 32, 10, 10]

        out,side = self.agg1(x4, x3, x2, x1)
        return self.upsample(out),feature_map,side


    def initialize_weights(self):
        # print('load pretrained for resnest')
        from resnest.torch import resnest101
        resnest101 = resnest101(pretrained=True)
        pretrained_dict = resnest101.state_dict()
        all_params = {}
        for k, v in self.resnest.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnest.state_dict().keys())
        self.resnest.load_state_dict(all_params)

class PDnet_ResNest_4F_SL_LA1(nn.Module):
    def __init__(self, channel=32):
        super(PDnet_ResNest_4F_SL_LA1, self).__init__()
        print('model:PDnet_ResNest_4F_SL_LA')
        from model.ours.backbone.resnet import ResNest101_new
        self.resnest = ResNest101_new([3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False)
        self.rfb1_1 = RFB(256, channel)
        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.agg1 = aggregation_4feat_side_loss_sa(channel)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.initialize_weights()

    def forward(self, x,y):
        x = self.resnest.conv1(torch.concat((x,y),1))
        x = self.resnest.bn1(x)
        x = self.resnest.relu(x)

        x = self.resnest.maxpool(x) #[1, 128, 80, 80]
        x1 = self.resnest.layer1(x)  #[1, 256, 80, 80]
        x2 = self.resnest.layer2(x1)  #[1, 512, 40, 40]
        x3 = self.resnest.layer3(x2)  #[1, 1024, 20, 20]
        x4 = self.resnest.layer4(x3)  #[1, 2048, 10, 10]

        feature_map = []
        feature_map.append(x4)
        feature_map.append(x3)
        feature_map.append(x2)
        feature_map.append(x1)

        x1 = self.rfb1_1(x1)  #[1, 32, 80, 80]
        x2 = self.rfb2_1(x2)  #[1, 32, 40, 40] 
        x3 = self.rfb3_1(x3)  #[1, 32, 20, 20]  
        x4 = self.rfb4_1(x4)  #[1, 32, 10, 10]

        out,side = self.agg1(x4, x3, x2, x1)
        return self.upsample(out),feature_map,side


    def initialize_weights(self):
        # print('load pretrained for resnest')
        from resnest.torch import resnest101
        resnest101 = resnest101(pretrained=True)
        pretrained_dict = resnest101.layer1.state_dict()
        all_params = {}
        for k, v in self.resnest.layer1.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
        self.resnest.layer1.load_state_dict(all_params)

        pretrained_dict = resnest101.layer2.state_dict()
        all_params = {}
        for k, v in self.resnest.layer2.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
        self.resnest.layer2.load_state_dict(all_params)

        pretrained_dict = resnest101.layer3.state_dict()
        all_params = {}
        for k, v in self.resnest.layer3.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
        self.resnest.layer3.load_state_dict(all_params)

        pretrained_dict = resnest101.layer4.state_dict()
        all_params = {}
        for k, v in self.resnest.layer4.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
        self.resnest.layer4.load_state_dict(all_params)

# most basic structure
class PDnet(nn.Module):
    def __init__(self, channel=32):
        super(PDnet, self).__init__()
        print('model:PDnet')
        from model.ours.backbone.resnet import ResNest101
        self.resnest = ResNest101([3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False)
        self.rfb1_1 = RFB(256, channel)
        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.agg1 = aggregation_basic(channel)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.initialize_weights()

    def forward(self, x):
        x = self.resnest.conv1(x)
        x = self.resnest.bn1(x)
        x = self.resnest.relu(x)

        x = self.resnest.maxpool(x) #[1, 128, 80, 80]
        x1 = self.resnest.layer1(x)  #[1, 256, 80, 80]
        x2 = self.resnest.layer2(x1)  #[1, 512, 40, 40]
        x3 = self.resnest.layer3_1(x2)  #[1, 1024, 20, 20]
        x4 = self.resnest.layer4_1(x3)  #[1, 2048, 10, 10]

        feature_map = []
        feature_map.append(x4)
        feature_map.append(x3)
        feature_map.append(x2)
        feature_map.append(x1)

        out = self.agg1(x4, x3, x2, x1)
        return self.upsample(out),feature_map


    def initialize_weights(self):
        # print('load pretrained for resnest')
        from resnest.torch import resnest101
        resnest101 = resnest101(pretrained=True)
        pretrained_dict = resnest101.state_dict()
        all_params = {}
        for k, v in self.resnest.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnest.state_dict().keys())
        self.resnest.load_state_dict(all_params)