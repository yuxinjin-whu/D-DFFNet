import argparse
import os
import numpy as np
import torch
from PIL import Image
from sklearn import metrics

def eval1(mask_path,gt_path,m):
    files=os.listdir(gt_path)
    maes=0
    num = 0
    precesions=0
    recalls=0
    fmeasures=0
    for file in files:
        mask1=mask_path+'/'+file.split('.')[0]+'.png'
        # mask1=mask_path+'/'+file.split('.')[0]+'.bmp'

        gt1=gt_path+'/'+file

        mask1 = Image.open(mask1)
        mask1 = mask1.convert('L')
        # mask1 = mask1.resize((320, 320))
        mask = np.array(mask1)
        mask = mask.astype(float)/255.0

        # mask = 1-mask
        mask_1 = mask

        (w,h)=mask.shape
        zeros = np.zeros((w, h))

        for i in range(w):
            for j in range(h):
                if mask_1[i,j]>=m:
                    zeros[i,j]=1.0
                else:
                    zeros[i,j]=0.0

        gt=(np.array(Image.open(gt1).convert('L')))/255.0
   
        if gt.shape == mask.shape:
            num+=1
            if num%10 == 0:
                print(num)
        
            for i in range(w):
                for j in range(h):
                    if gt[i,j]>=0.5:
                        gt[i,j]=1.0
                    else:
                        gt[i,j]=0.0
            mae=np.mean(np.abs((gt-mask)))
            maes+=mae

            gt = 1-gt
            zeros = 1-zeros

            precesion=metrics.precision_score(gt.reshape(-1), zeros.reshape(-1))
         
            precesions+=precesion
            recall=metrics.recall_score(gt.reshape(-1), zeros.reshape(-1))
            
            recalls+=recall
            if precesion==0 and recall==0:
                fmeasure=0.0
            else:
                fmeasure=((1+0.3)*precesion*recall)/(0.3*precesion+recall)
            
            fmeasures+=fmeasure
    mae1=maes/num
    fmeasure1=fmeasures/num
    recall1=recalls/num
    precesion1=precesions/num
    return mae1,fmeasure1,recall1,precesion1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--mask_path', type=str)
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--m', default=0.5, type=int)
    args=parser.parse_args()

    mae, a, r0, p0 = eval1(args.mask_path, args.gt_path, 0.5)
    print(mae,a,r0,p0)

    
    

