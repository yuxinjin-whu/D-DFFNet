import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class aggregation_basic(nn.Module):
    
    def __init__(self, channel):
        super(aggregation_basic, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)

        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4*channel, 1, 1)

        self.conv11 = nn.Conv2d(2048, channel, 1)
        self.conv22 = nn.Conv2d(1024, channel, 1)
        self.conv33 = nn.Conv2d(512, channel, 1)
        self.conv44 = nn.Conv2d(256, channel, 1)


    def forward(self, x1, x2, x3, x4):

        x1 = self.conv11(x1)
        x2 = self.conv22(x2)
        x3 = self.conv33(x3)
        x4 = self.conv44(x4)

        x2_2 = torch.cat((x2, self.conv_upsample4(self.upsample(x1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        x4_2 = torch.cat((x4, self.conv_upsample7(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat4(x4_2)
        x = self.conv4(x4_2)
        x = self.conv5(x)

        return x

class aggregation(nn.Module):
    # dense aggregation
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)
        self.conv6 = nn.Conv2d(3*channel, 1, 1)

        self.attention1 = nn.Sequential(nn.Conv2d(1, 1, 7, padding=3, bias=False), nn.Sigmoid())
    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3
        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

class aggregation_4feat(nn.Module):

    def __init__(self, channel):
        super(aggregation_4feat, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)

        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4*channel, 1, 1)

    def forward(self, x1, x2, x3, x4):

        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3
        x4_1 = self.conv_upsample2(self.upsample(self.upsample(self.upsample(x1)))) \
               * self.conv_upsample3(self.upsample(self.upsample(x2))) * self.conv_upsample6(self.upsample(x3)) * x4

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        x4_2 = torch.cat((x4_1, self.conv_upsample7(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat4(x4_2)

        x = self.conv4(x4_2)
        x = self.conv5(x)

        return x

class aggregation_5feat(nn.Module):
    
    def __init__(self, channel):
        super(aggregation_5feat, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_upsample8 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample9 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        
        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv_concat5 = BasicConv2d(5*channel, 5*channel, 3, padding=1)

        self.conv4 = BasicConv2d(5*channel, 5*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(5*channel, 1, 1)

    def forward(self, x1, x2, x3, x4, x5):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3
        x4_1 = self.conv_upsample2(self.upsample(self.upsample(self.upsample(x1)))) \
               * self.conv_upsample3(self.upsample(self.upsample(x2))) * self.conv_upsample6(self.upsample(x3)) * x4
        x5_1 = self.conv_upsample2(self.upsample(self.upsample(self.upsample(x1)))) \
               * self.conv_upsample3(self.upsample(self.upsample(x2))) * self.conv_upsample6(self.upsample(x3)) * self.conv_upsample8(x4) * x5

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        x4_2 = torch.cat((x4_1, self.conv_upsample7(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat4(x4_2)
        x5_2 = torch.cat((x5_1, self.conv_upsample9(x4_2)), 1)
        x5_2 = self.conv_concat5(x5_2)

        x = self.conv4(x5_2)
        x = self.conv5(x)

        return x

class aggregation_4feat_side_loss(nn.Module):
    
    def __init__(self, channel):
        super(aggregation_4feat_side_loss, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)

        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4*channel, 1, 1)

        self.side1 = nn.Conv2d(32, 1, 1)
        self.side2 = nn.Conv2d(32, 1, 1)
        self.side3 = nn.Conv2d(32, 1, 1)
        self.side4 = nn.Conv2d(32, 1, 1)


    def forward(self, x1, x2, x3, x4):

        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3
        x4_1 = self.conv_upsample2(self.upsample(self.upsample(self.upsample(x1)))) \
               * self.conv_upsample3(self.upsample(self.upsample(x2))) \
               * self.conv_upsample6(self.upsample(x3)) * x4
        
        side1 = self.conv_upsample4(self.upsample(x1_1))

        side = []
        side.append(self.side1(side1))
        side.append(self.side2(x2_1))
        side.append(self.side3(x3_1))
        side.append(self.side4(x4_1))

        x2_2 = torch.cat((x2_1, side1), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        x4_2 = torch.cat((x4_1, self.conv_upsample7(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat4(x4_2)

        x = self.conv4(x4_2)
        x = self.conv5(x)


        return x,side

class aggregation_4feat_side_loss_sa(nn.Module):
    
    def __init__(self, channel):
        super(aggregation_4feat_side_loss_sa, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)

        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4*channel, 1, 1)

        self.side1 = nn.Conv2d(32, 1, 1)
        self.side2 = nn.Conv2d(32, 1, 1)
        self.side3 = nn.Conv2d(32, 1, 1)
        self.side4 = nn.Conv2d(32, 1, 1)

        self.spatial_attention1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False),
                                               nn.Sigmoid())
        self.spatial_attention2 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False),
                                               nn.Sigmoid())
        self.spatial_attention3 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False),
                                               nn.Sigmoid())
        self.spatial_attention4 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False),
                                               nn.Sigmoid())
    def forward(self, x1, x2, x3, x4):

        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3
        x4_1 = self.conv_upsample2(self.upsample(self.upsample(self.upsample(x1)))) \
               * self.conv_upsample3(self.upsample(self.upsample(x2))) \
               * self.conv_upsample6(self.upsample(x3)) * x4
        
        x1_1 = self.conv_upsample4(self.upsample(x1_1))

        side = []
        side1 = self.side1(x1_1)
        side2 = self.side2(x2_1)
        side3 = self.side3(x3_1)
        side4 = self.side4(x4_1)
        side.append(side1)
        side.append(side2)
        side.append(side3)
        side.append(side4)

        x1_1 = torch.mul(self.spatial_attention1(side1),x1_1)
        x2_1 = torch.mul(self.spatial_attention2(side2),x2_1)
        x3_1 = torch.mul(self.spatial_attention3(side3),x3_1)
        x4_1 = torch.mul(self.spatial_attention4(side4),x4_1)

        x2_2 = torch.cat((x2_1, x1_1), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        x4_2 = torch.cat((x4_1, self.conv_upsample7(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat4(x4_2)

        x = self.conv4(x4_2)
        x = self.conv5(x)

        return x,side

class aggregation_4feat_side_loss_sa_ng(nn.Module):
    
    def __init__(self, channel):
        super(aggregation_4feat_side_loss_sa_ng, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)

        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4*channel, 1, 1)

        self.side1 = nn.Conv2d(32, 1, 1)
        self.side2 = nn.Conv2d(32, 1, 1)
        self.side3 = nn.Conv2d(32, 1, 1)
        self.side4 = nn.Conv2d(32, 1, 1)

        self.spatial_attention1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False),
                                               nn.Sigmoid())
        self.spatial_attention2 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False),
                                               nn.Sigmoid())
        self.spatial_attention3 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False),
                                               nn.Sigmoid())
        self.spatial_attention4 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False),
                                               nn.Sigmoid())
    def forward(self, x1, x2, x3, x4):

        x1_1 = x1
        x2_1 = x2
        x3_1 = x3
        x4_1 = x4

        side = []
        side1 = self.side1(x1_1)
        side2 = self.side2(x2_1)
        side3 = self.side3(x3_1)
        side4 = self.side4(x4_1)
        side.append(side1)
        side.append(side2)
        side.append(side3)
        side.append(side4)

        x1_1 = torch.mul(self.spatial_attention1(side1),x1_1)
        x2_1 = torch.mul(self.spatial_attention2(side2),x2_1)
        x3_1 = torch.mul(self.spatial_attention3(side3),x3_1)
        x4_1 = torch.mul(self.spatial_attention4(side4),x4_1)

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        x4_2 = torch.cat((x4_1, self.conv_upsample7(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat4(x4_2)

        x = self.conv4(x4_2)
        x = self.conv5(x)

        return x,side

class aggregation_4feat_side_loss_sa_d(nn.Module):
    
    def __init__(self, channel):
        super(aggregation_4feat_side_loss_sa_d, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)

        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4*channel, 1, 1)

        self.side1 = nn.Conv2d(32, 1, 1)
        self.side2 = nn.Conv2d(32, 1, 1)
        self.side3 = nn.Conv2d(32, 1, 1)
        self.side4 = nn.Conv2d(32, 1, 1)

        self.side5 = nn.Conv2d(32, 1, 1)
        self.side6 = nn.Conv2d(32, 1, 1)
        self.side7 = nn.Conv2d(32, 1, 1)
        self.side8 = nn.Conv2d(32, 1, 1)

        self.spatial_attention1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False),
                                               nn.Sigmoid())
        self.spatial_attention2 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False),
                                               nn.Sigmoid())
        self.spatial_attention3 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False),
                                               nn.Sigmoid())
        self.spatial_attention4 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False),
                                               nn.Sigmoid())
    def forward(self, x1, x2, x3, x4):

        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3
        x4_1 = self.conv_upsample2(self.upsample(self.upsample(self.upsample(x1)))) \
               * self.conv_upsample3(self.upsample(self.upsample(x2))) \
               * self.conv_upsample6(self.upsample(x3)) * x4
        
        x1_1 = self.conv_upsample4(self.upsample(x1_1))

        side = []
        depth = []

        side1 = self.side1(x1_1)
        side2 = self.side2(x2_1)
        side3 = self.side3(x3_1)
        side4 = self.side4(x4_1)

        side5 = self.side5(x1_1)
        side6 = self.side6(x2_1)
        side7 = self.side7(x3_1)
        side8 = self.side8(x4_1)

        side.append(side1)
        side.append(side2)
        side.append(side3)
        side.append(side4)

        depth.append(side5)
        depth.append(side6)
        depth.append(side7)
        depth.append(side8)

        x1_1 = torch.mul(self.spatial_attention1(side1),x1_1)
        x2_1 = torch.mul(self.spatial_attention2(side2),x2_1)
        x3_1 = torch.mul(self.spatial_attention3(side3),x3_1)
        x4_1 = torch.mul(self.spatial_attention4(side4),x4_1)

        x2_2 = torch.cat((x2_1, x1_1), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        x4_2 = torch.cat((x4_1, self.conv_upsample7(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat4(x4_2)

        x = self.conv4(x4_2)
        x = self.conv5(x)

        return x,side,depth