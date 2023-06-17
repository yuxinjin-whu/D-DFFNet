import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os, argparse
from datetime import datetime

from model.ours.PDNet_resnest import PDnet_ResNest_3F,PDnet_Vgg_3F,PDnet_Vgg_Res_3F,\
        PDnet_ResNest_4F,PDnet_ResNest_5F,PDnet_ResNest_4F_SL,PDnet_ResNest_4F_SL_LA,\
            PDnet_ResNext_4F_SL_LA,PDnet_Vgg_4F_SL_LA,PDnet_Res2Net_4F_SL_LA,PDnet,PDnet_ResNest_4F_SL_LA_NG

from data import get_loader
from utils import adjust_learning_rate_poly,clip_gradient,eval_mae
from loss import ECELoss

parser = argparse.ArgumentParser()
parser.add_argument('--backbone',  type=str, default='Resnest')
parser.add_argument('--model',  type=str, default='PDnet_ResNest_4F_SL_LA') # DFFNet
parser.add_argument('--decoder_num',  type=int, default=4)
parser.add_argument('--sl',  type=str, default='yes')
parser.add_argument('--la',  type=str, default='yes')

parser.add_argument('--epoch', type=int, default=75, help='epoch number')
parser.add_argument('--optim', type=str, default='adam')

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_strategy', type=str, default='poly', help='learning rate strategy')
parser.add_argument('--gamma',  type=float, default=0.9)  #2.0 0.8

parser.add_argument('--batchsize', type=int, default=6, help='training batch size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')

parser.add_argument('--loss',  type=str, default='ea') # bce + DOF-edge loss
parser.add_argument('--beta',  type=float, default=0.5)

parser.add_argument('--save_name', type=str, default='DFFNet.pth')
parser.add_argument('--train_dataset', type=str, default='Shi')
parser.add_argument('--test_dataset', type=str, default='Shi') #CTCUG Shi EBD DUT
parser.add_argument('--image_path',  type=str, default='./data/shidatatset/image')
parser.add_argument('--gt_path',  type=str, default='./data/shidatatset/gt')
opt = parser.parse_args()

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)  
torch.cuda.manual_seed(seed) 

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

print(20*'*','training',20*'*')
train_loader = get_loader(opt.image_path, opt.gt_path, opt.batchsize, dataset=opt.train_dataset, mode='train', num_thread=8)
print(len(train_loader))

# model
if opt.backbone == 'Resnest':
    if opt.decoder_num == 3:
        model = PDnet_ResNest_3F()
    elif opt.decoder_num == 5:
        model = PDnet_ResNest_5F()
    elif opt.decoder_num == 4:
        if opt.sl == 'yes':
            if opt.la == 'yes':
                model = PDnet_ResNest_4F_SL_LA() # DFFNet
            else:
                model = PDnet_ResNest_4F_SL()
        else:
            model = PDnet_ResNest_4F()
elif opt.backbone == 'Vgg':
    model = PDnet_Vgg_4F_SL_LA()
elif opt.backbone == 'Resnext':
    model = PDnet_ResNext_4F_SL_LA()
elif opt.backbone == 'Res2net':
    model = PDnet_Res2Net_4F_SL_LA()

if opt.model == 'PDNet':
    model = PDnet()
if opt.model == 'NG':
    model = PDnet_ResNest_4F_SL_LA_NG()

model.cuda()
params = model.parameters()
if opt.optim == 'adam':
    optimizer = torch.optim.Adam(params, opt.lr)
elif opt.optim == 'sgd':
    optimizer = torch.optim.SGD(params,momentum=0.9,lr=0.005)

# loss
if opt.loss == 'bce':
    # bce loss
    criteria = torch.nn.BCEWithLogitsLoss()
elif opt.loss == 'ea':
    # bce + DOF-edge loss
    criteria = ECELoss(n_classes=1,beta=opt.beta).cuda()
    print(opt.beta)
else:
    raise NotImplementedError

def train(train_loader, model, optimizer, epoch):
    model.train()
    sum_mae = 0
    loss_sum = 0
    
    if opt.optim == 'adam' and opt.lr_strategy == 'poly':
        adjust_learning_rate_poly(optimizer,epoch,opt.epoch,opt.lr,opt.gamma)

    for i, pack in enumerate(train_loader, start=1):

        optimizer.zero_grad()
        images, gts = pack
        images, gts = images.to('cuda:0'), gts.to('cuda:0')

        if 'Shi' == opt.train_dataset:
            gts = 1-gts

        if opt.sl == 'yes':
            preds,_,side = model(images)
        else:
            preds,_ = model(images)

        loss= criteria(preds, gts)
        if opt.sl == 'yes':
            for i in side:
                loss += criteria(F.interpolate(i, size=320, mode='bilinear'),gts)
        loss_sum+=loss
        loss.backward()

        # if epoch == 1:
        #     print(loss.item())

        preds = torch.sigmoid(preds)
        mae = eval_mae(preds, gts).item()
        sum_mae+=mae

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

    print('{} Epoch [{:03d}/{:03d}] loss: {:0.4f} mae: {:0.4f}'.
            format(datetime.now(), epoch, opt.epoch, loss_sum/len(train_loader), sum_mae/len(train_loader)))

print("Let's go!")
save_path = './checkpoint/'
for epoch in range(1, opt.epoch):
    train(train_loader, model, optimizer, epoch)

torch.save(model.state_dict(), save_path + opt.save_name)

