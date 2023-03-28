import torch
import torch.nn.functional as F
import numpy as np
import os, argparse

from model.ours.PDNet_resnest import PDnet_ResNest_3F,PDnet_Vgg_3F,PDnet_Vgg_Res_3F,\
        PDnet_ResNest_4F,PDnet_ResNest_5F,PDnet_ResNest_4F_SL,PDnet_ResNest_4F_SL_LA,\
            PDnet_ResNext_4F_SL_LA,PDnet_Vgg_4F_SL_LA,PDnet_Res2Net_4F_SL_LA

from data import get_loader

parser = argparse.ArgumentParser()
parser.add_argument('--test_dataset', type=str, default='Shi') #CTCUG Shi EBD DUT
parser.add_argument('--pretrained', type=str) 
parser.add_argument('--model', type=str, default='res') #vgg 
parser.add_argument('--image_path',  type=str, default='/data/jyx/dbd/data/DBDdataset/shidatatset/image')
parser.add_argument('--gt_path',  type=str, default='/data/jyx/dbd/data/DBDdataset/shidatatset/gt')
opt = parser.parse_args()

val_loader = get_loader(opt.image_path, opt.gt_path, 1, dataset=opt.test_dataset, mode='val', num_thread=8)

if opt.model == 'res':
    model = PDnet_ResNest_4F_SL()
elif opt.model == 'vgg':
    model = PDnet_Vgg_4F_SL_LA()
elif opt.model == 'res1':
    model = PDnet_ResNest_4F()
model.cuda()
model.load_state_dict(torch.load(opt.pretrained))

def test(val_loader, model):
    model.eval()
    number = 0
    GT = np.array([0])
    MASK = np.array([0])
    with torch.no_grad():
        for i, pack in enumerate(val_loader, start=1):
            images, gts, size, _ = pack
            images, gts = images.to('cuda:0'), gts.to('cuda:0')

            if 'Shi' == opt.test_dataset or 'CTCUG' == opt.test_dataset or 'Shi1' == opt.test_dataset:
                gts = 1-gts

            preds,_,_ = model(images)
            # preds,_ = model(images)

            preds = torch.sigmoid(preds)
            preds = F.interpolate(preds, size=size, mode='nearest')

            if preds.shape == gts.shape:

                gts[gts>0.5] = 1.0
                gts[gts<=0.5] = 0.0
                preds[preds>0.5] = 1.0
                preds[preds<=0.5] = 0.0

                if 'Mydata'in opt.test_dataset:
                    gts = F.interpolate(gts, size=320, mode='nearest')
                    preds = F.interpolate(preds, size=320, mode='nearest')

                gts = 1-gts
                preds = 1-preds

                GT = np.concatenate((GT,gts.detach().cpu().numpy().reshape(-1)),0)
                MASK = np.concatenate((MASK,preds.detach().cpu().numpy().reshape(-1)),0)

        pred = MASK[1:]
        target = GT[1:]

        pred_inds = pred == 1
        target_inds = target == 1
        intersection = (pred_inds[target_inds]).sum() 
        union = pred_inds.sum() + target_inds.sum() - intersection
        ious = float(intersection) / float(union)

    return ious

iou = test(val_loader,model)
print(opt.test_dataset,iou)


