import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os, argparse
from sklearn import metrics

from model.ours.PDNet_resnest import PDnet_ResNest_3F,PDnet_Vgg_3F,PDnet_Vgg_Res_3F,\
        PDnet_ResNest_4F,PDnet_ResNest_5F,PDnet_ResNest_4F_SL,PDnet_ResNest_4F_SL_LA,\
            PDnet_ResNext_4F_SL_LA,PDnet_Vgg_4F_SL_LA,PDnet_Res2Net_4F_SL_LA,PDnet_ResNest_4F_SL_LA_D

from data import get_loader
from utils import eval_mae, F_score1, F_score

parser = argparse.ArgumentParser()
parser.add_argument('--test_dataset', type=str, default='OnePic') #CTCUG Shi EBD DUT
parser.add_argument('--pretrained', type=str) 
parser.add_argument('--model', type=str, default='res') #vgg 
parser.add_argument('--image_path',  type=str, default='/data/jyx/dbd/data/DBDdataset/shidatatset/image')
parser.add_argument('--gt_path',  type=str, default='/data/jyx/dbd/data/DBDdataset/shidatatset/gt')
opt = parser.parse_args()

val_loader = get_loader(opt.image_path, opt.gt_path, 1, dataset=opt.test_dataset, mode='val', num_thread=8)

if opt.model == 'res':
    model = PDnet_ResNest_4F_SL_LA()
elif opt.model == 'vgg':
    model = PDnet_Vgg_4F_SL_LA()

model.cuda()
model.load_state_dict(torch.load(opt.pretrained))

def test(val_loader, model):
    model.eval()
    sum_f_score = 0
    sum_mae = 0
    number = 0
    with torch.no_grad():
        for i, pack in enumerate(val_loader, start=1):
            images, gts, size, name = pack
            images, gts = images.to('cuda:0'), gts.to('cuda:0')

            if 'Shi' == opt.test_dataset or 'CTCUG' == opt.test_dataset or 'Shi1' == opt.test_dataset:
                gts = 1-gts

            preds,_,_ = model(images)
            preds = torch.sigmoid(preds)
            preds = F.interpolate(preds, size=size, mode='nearest')

            # # save images
            # dets = preds.cpu().numpy()[0][0]
            # dets = dets*255  
            # image_path = os.path.join('/data/jyx/dbd/D-DFFNet/results/EBD/D-DFFNet(vgg)/',name[0].split('.')[0]+'.png')
            # print(image_path)
            # cv2.imwrite(image_path, dets)

            if preds.shape == gts.shape:
                mae = eval_mae(preds, gts).item()
                sum_mae += mae
                number += 1

                f_score = F_score1(preds,gts)
                sum_f_score += f_score

    return sum_mae/number,sum_f_score/number

mae,f = test(val_loader,model)
print(opt.test_dataset,mae,f)


