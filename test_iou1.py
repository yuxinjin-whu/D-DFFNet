import argparse
import os
import numpy as np
from PIL import Image

def iou(mask_path,gt_path):
    files=os.listdir(gt_path)
    num = 0
    GT = np.array([0])
    MASK = np.array([0])

    for file in files:
        mask1=mask_path+'/'+file.split('.')[0]+'.jpg'
        gt1=gt_path+'/'+file

        mask=np.array(Image.open(mask1).convert('L'))/255.0
        gt=np.array(Image.open(gt1).convert('L'))/255.0

        # mask = 1-mask

        if gt.shape == mask.shape:
            num+=1
            if num%10 == 0:
                print(num)
        
            gt[gt>0.5] = 1.0
            gt[gt<=0.5] = 0.0

            mask[mask>0.5] = 1.0
            mask[mask<=0.5] = 0.0

            gt = 1-gt
            mask = 1-mask

            GT = np.concatenate((GT,gt.reshape(-1)),0)
            MASK = np.concatenate((MASK,mask.reshape(-1)),0)

    pred = MASK[1:]
    target = GT[1:]

    pred_inds = pred == 1
    target_inds = target == 1
    intersection = (pred_inds[target_inds]).sum()  # Cast to long to prevent overflows
    union = pred_inds.sum() + target_inds.sum() - intersection
    ious = float(intersection) / float(union)
    print(ious)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--mask_path', type=str)
    parser.add_argument('--gt_path', type=str)
    args=parser.parse_args()

    iou(args.mask_path,args.gt_path)
    print(args.mask_path.split('/')[-1])
    

