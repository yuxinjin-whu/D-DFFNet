import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os, argparse
from datetime import datetime
from loss import ECELoss,AutomaticWeightedLoss 
from model.ours.depth_model.midas_net import MidasNet
from model.ours.depth_model.dpt_depth import DPTDepthModel1
from model.ours.PDNet_resnest import PDnet_ResNest_3F,PDnet_Vgg_3F,PDnet_Vgg_Res_3F,\
        PDnet_ResNest_4F,PDnet_ResNest_5F,PDnet_ResNest_4F_SL,PDnet_ResNest_4F_SL_LA,\
            PDnet_ResNext_4F_SL_LA,PDnet_Vgg_4F_SL_LA,PDnet_Res2Net_4F_SL_LA,PDnet,PDnet_ResNest_4F_SL_LA_NG
from data import get_loader
from utils import clip_gradient,eval_mae,adjust_learning_rate_poly

class Adapter_DP2DP_vgg(torch.nn.Module):
    def __init__(self):
        super(Adapter_DP2DP_vgg, self).__init__()
        self.conv1 = torch.nn.Conv2d(512, 512, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(512, 512, 1, bias=False)
        self.conv3 = torch.nn.Conv2d(256, 256, 1, bias=False)
        self.conv4 = torch.nn.Conv2d(128, 128, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, inputs):
        results = []
        results.append(self.conv1(inputs[0]))
        results.append(self.conv2(inputs[1]))
        results.append(self.conv3(inputs[2]))
        results.append(self.conv4(inputs[3]))

        return results

class Adapter_PD2Depth(torch.nn.Module):
    def __init__(self):
        super(Adapter_PD2Depth, self).__init__()
        self.conv1 = torch.nn.Conv2d(2048, 2048, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(1024, 1024, 1, bias=False)
        self.conv3 = torch.nn.Conv2d(512, 512, 1, bias=False)
        self.conv4 = torch.nn.Conv2d(256, 256, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, inputs):
        results = []
        results.append(self.conv1(inputs[0]))
        results.append(self.conv2(inputs[1]))
        results.append(self.conv3(inputs[2]))
        results.append(self.conv4(inputs[3]))

        return results

class Adapter_DP2DP(torch.nn.Module):
    def __init__(self):
        super(Adapter_DP2DP, self).__init__()
        self.conv1 = torch.nn.Conv2d(2048, 2048, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(1024, 1024, 1, bias=False)
        self.conv3 = torch.nn.Conv2d(512, 512, 1, bias=False)
        self.conv4 = torch.nn.Conv2d(256, 256, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, inputs):
        results = []
        results.append(self.conv1(inputs[0]))
        results.append(self.conv2(inputs[1]))
        results.append(self.conv3(inputs[2]))
        results.append(self.conv4(inputs[3]))

        return results

class Adapter_PD2Depth_vgg(torch.nn.Module):
    def __init__(self):
        super(Adapter_PD2Depth_vgg, self).__init__()
        self.conv1 = torch.nn.Conv2d(512, 2048, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(512, 1024, 1, bias=False)
        self.conv3 = torch.nn.Conv2d(256, 512, 1, bias=False)
        self.conv4 = torch.nn.Conv2d(128, 256, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, inputs):
        results = []
        results.append(self.conv1(inputs[0]))
        results.append(self.conv2(inputs[1]))
        results.append(self.conv3(inputs[2]))
        results.append(self.conv4(inputs[3]))

        return results
        
parser = argparse.ArgumentParser()
parser.add_argument('--backbone',  type=str, default='Resnest')
parser.add_argument('--decoder_num',  type=int, default=4)
parser.add_argument('--sl',  type=str, default='yes')
parser.add_argument('--la',  type=str, default='yes')

parser.add_argument('--epoch', type=int, default=75, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_strategy', type=str, default='poly', help='learning rate strategy')

parser.add_argument('--batchsize', type=int, default=6, help='training batch size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--loss',  type=str, default='ea')
parser.add_argument('--beta',  type=float, default=0.5)

parser.add_argument('--gamma',  type=float, default=0.9)  #2.0 0.8
parser.add_argument('--train_dataset', type=str, default='Shi')
parser.add_argument('--test_dataset', type=str, default='Shi')
parser.add_argument('--image_path',  type=str, default='./data/shidatatset/image')
parser.add_argument('--gt_path',  type=str, default='./data/shidatatset/gt')
opt = parser.parse_args()

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

model_depth = MidasNet()
model_depth.cuda()
model_depth.load_state_dict(torch.load("./depth_pretrained/midas_v21-f6b98070.pt"))

model_dbd = PDnet_ResNest_4F_SL_LA()
model_dbd.cuda()
# training checkpoint from stage 1.
model_dbd.load_state_dict(torch.load('./checkpoint/DFFNet.pth'))

model = PDnet_ResNest_4F_SL_LA()
model.cuda()

adapters = {}
if opt.backbone == 'vgg':
    adapters[0] = Adapter_DP2DP_vgg().cuda()
    adapters[1] = Adapter_PD2Depth_vgg().cuda()
else:
    adapters[0] = Adapter_DP2DP().cuda()
    adapters[1] = Adapter_PD2Depth().cuda()
params1 = []
for i in range(2):
    params1 += adapters[i].parameters()
transformer_optimizer = torch.optim.Adam(params1, lr=1e-1, weight_decay=5e-4)

print(20*'*','training',20*'*')
train_loader = get_loader(opt.image_path, opt.gt_path, opt.batchsize, dataset=opt.train_dataset, mode='train', num_thread=8)

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

if opt.loss == 'bce':
    criteria = torch.nn.BCEWithLogitsLoss()
elif opt.loss == 'ea':
    criteria = ECELoss(n_classes=1,beta=opt.beta).cuda()
else:
    raise NotImplementedError

weight =AutomaticWeightedLoss(3)

def train(train_loader, model, optimizer, epoch):
    model.train()
    sum_mae = 0
    loss_sum = 0

    if opt.lr_strategy == 'poly':
        adjust_learning_rate_poly(optimizer,epoch,opt.epoch,opt.lr,opt.gamma)

    for i, pack in enumerate(train_loader, start=1):

        optimizer.zero_grad()
        transformer_optimizer.zero_grad()
        
        images, gts = pack
        images, gts = images.to('cuda:0'), gts.to('cuda:0')

        if 'Shi' == opt.train_dataset:
            gts = 1-gts

        if opt.sl == 'yes':
            preds,feat0,side = model(images)
        else:
            preds,feat0 = model(images)

        with torch.no_grad():
            feat_depth, _ = model_depth(images)
            if opt.sl == 'yes':
                _, feat_dbd, _ = model_dbd(images)
            else:
                _, feat_dbd = model_dbd(images)

        dist_loss = []
        feat_dbd1 = feat_dbd[0].detach()
        feat_dbd1 = feat_dbd1 / (feat_dbd1.pow(2).sum(1) + 1e-6).sqrt().view(feat_dbd1.size(0), 1, feat_dbd1.size(2), feat_dbd1.size(3))
        feat_dbd2 = feat_dbd[1].detach()
        feat_dbd2 = feat_dbd2 / (feat_dbd2.pow(2).sum(1) + 1e-6).sqrt().view(feat_dbd2.size(0), 1, feat_dbd2.size(2), feat_dbd2.size(3))
        feat_dbd3 = feat_dbd[2].detach()
        feat_dbd3 = feat_dbd3 / (feat_dbd3.pow(2).sum(1) + 1e-6).sqrt().view(feat_dbd3.size(0), 1, feat_dbd3.size(2), feat_dbd3.size(3))
        feat_dbd4 = feat_dbd[3].detach()
        feat_dbd4 = feat_dbd4 / (feat_dbd4.pow(2).sum(1) + 1e-6).sqrt().view(feat_dbd4.size(0), 1, feat_dbd4.size(2), feat_dbd4.size(3))

        feat_depth1 = feat_depth[0].detach()
        feat_depth1 = feat_depth1 / (feat_depth1.pow(2).sum(1) + 1e-6).sqrt().view(feat_depth1.size(0), 1, feat_depth1.size(2), feat_depth1.size(3))
        feat_depth2 = feat_depth[1].detach()
        feat_depth2 = feat_depth2 / (feat_depth2.pow(2).sum(1) + 1e-6).sqrt().view(feat_depth2.size(0), 1, feat_depth2.size(2), feat_depth2.size(3))
        feat_depth3 = feat_depth[2].detach()
        feat_depth3 = feat_depth3 / (feat_depth3.pow(2).sum(1) + 1e-6).sqrt().view(feat_depth3.size(0), 1, feat_depth3.size(2), feat_depth3.size(3))
        feat_depth4 = feat_depth[3].detach()
        feat_depth4 = feat_depth4 / (feat_depth4.pow(2).sum(1) + 1e-6).sqrt().view(feat_depth4.size(0), 1, feat_depth4.size(2), feat_depth4.size(3))

        feat = adapters[0](feat0)
        feat_si1 = feat[0] / (feat[0].pow(2).sum(1) + 1e-6).sqrt().view(feat[0].size(0), 1, feat[0].size(2), feat[0].size(3))
        feat_si2 = feat[1] / (feat[1].pow(2).sum(1) + 1e-6).sqrt().view(feat[1].size(0), 1, feat[1].size(2), feat[1].size(3))
        feat_si3 = feat[2] / (feat[2].pow(2).sum(1) + 1e-6).sqrt().view(feat[2].size(0), 1, feat[2].size(2), feat[2].size(3))
        feat_si4 = feat[3] / (feat[3].pow(2).sum(1) + 1e-6).sqrt().view(feat[3].size(0), 1, feat[3].size(2), feat[3].size(3))

        feat1 = adapters[1](feat0)
        feat_di1 = feat1[0] / (feat1[0].pow(2).sum(1) + 1e-6).sqrt().view(feat1[0].size(0), 1, feat1[0].size(2), feat1[0].size(3))
        feat_di2 = feat1[1] / (feat1[1].pow(2).sum(1) + 1e-6).sqrt().view(feat1[1].size(0), 1, feat1[1].size(2), feat1[1].size(3))
        feat_di3 = feat1[2] / (feat1[2].pow(2).sum(1) + 1e-6).sqrt().view(feat1[2].size(0), 1, feat1[2].size(2), feat1[2].size(3))
        feat_di4 = feat1[3] / (feat1[3].pow(2).sum(1) + 1e-6).sqrt().view(feat1[3].size(0), 1, feat1[3].size(2), feat1[3].size(3))

        dist_1 = (feat_si1 - feat_dbd1).pow(2).sum(1).mean()
        dist_2 = (feat_si2 - feat_dbd2).pow(2).sum(1).mean()
        dist_3 = (feat_si3 - feat_dbd3).pow(2).sum(1).mean()
        dist_7 = (feat_si4 - feat_dbd4).pow(2).sum(1).mean()
        dist_loss.append(dist_1 + dist_2 + dist_3 + dist_7)

        if opt.backbone == 'vgg':    
            dist_4 = (feat_di1 - F.interpolate(feat_depth1, size=20, mode='bilinear')).pow(2).sum(1).mean()
            dist_5 = (feat_di2 - F.interpolate(feat_depth2, size=40, mode='bilinear')).pow(2).sum(1).mean()
            dist_6 = (feat_di3 - F.interpolate(feat_depth3, size=80, mode='bilinear')).pow(2).sum(1).mean()
            dist_8 = (feat_di4 - F.interpolate(feat_depth4, size=160, mode='bilinear')).pow(2).sum(1).mean()
        else:
            dist_4 = (feat_di1 - feat_depth1).pow(2).sum(1).mean()
            dist_5 = (feat_di2 - feat_depth2).pow(2).sum(1).mean()
            dist_6 = (feat_di3 - feat_depth3).pow(2).sum(1).mean()
            dist_8 = (feat_di4 - feat_depth4).pow(2).sum(1).mean()     

        dist_loss.append(dist_4 + dist_5 + dist_6 + dist_8)

        if opt.sl == 'yes':
            loss_out = 1.1*criteria(preds, gts)
            for i in side:
                loss_out += criteria(F.interpolate(i, size=320, mode='bilinear'),gts)
        else:
            loss_out = criteria(preds, gts)

        if epoch < 15:
            alpha = 3
        else:
            alpha = 3*((epoch-15)/(opt.epoch-15))

        if opt.backbone == 'Resnext':
            alpha = 1
        loss = loss_out + alpha*(dist_loss[0] + dist_loss[1])

        loss_sum+=loss
        loss.backward()

        preds = torch.sigmoid(preds)
        mae = eval_mae(preds, gts).item()
        sum_mae+=mae
        if epoch < 10:
            print((dist_loss[0] + dist_loss[1]).item())

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        transformer_optimizer.step()
    
    print('{} Epoch [{:03d}/{:03d}] loss: {:0.4f} mae: {:0.5f}'.
            format(datetime.now(), epoch, opt.epoch, loss_sum/len(train_loader), sum_mae/len(train_loader)))

print("Let's go!")
save_path = './checkpoint/'
for epoch in range(1, opt.epoch):
    train(train_loader, model, optimizer, epoch)

torch.save(model.state_dict(), save_path + 'D-DFFNet.pth')

print(datetime.now())
