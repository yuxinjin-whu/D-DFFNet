import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import os,random
from torchvision.transforms import InterpolationMode
import torch
import numpy as np
from PIL import Image
from torch.utils import data
import glob
import torchvision.transforms.functional as F

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)

def h_v_filp(image, mask,mask1):
    if random.random() > 0.5:
        image = F.hflip(image)
        mask = F.hflip(mask)
        mask1 = F.hflip(mask1)

    return image, mask,mask1

def h_v_filp1(image, mask):
    if random.random() > 0.5:
        image = F.hflip(image)
        mask = F.hflip(mask)

    return image, mask

class ImageData(data.Dataset):
    def __init__(self, img_root, label_root, transform, t_transform, dataset, istrain=True,factor=8,preloadimg=False):
        self.istrain = istrain
        self.preloadimg = preloadimg
        self.factor = factor
        # the same split with BTBNet, we borrow it from http://ice.dlut.edu.cn/ZhaoWenda/DBD.html 
        train_names_file = open('split.txt', mode='r')
        image_path = []
        label_path = []

        # CUHK
        if dataset == 'Shi':
            for line in train_names_file:
                line = line.split('--')[1]
                line1 = line.split('\n')[0]
                line2 = line.split('.')[0]
                image = os.path.join('./data/shidatatset/image',line1)
                gt = os.path.join('./data/shidatatset/gt',line2+'.png')
                image_path.append(image)
                label_path.append(gt) 
            
            if not self.istrain:
                self.image_path = image_path
                self.label_path =  label_path
            else:
                image_path_all = sorted( glob.glob(img_root+'/*'))
                self.image_path = [i for i in image_path_all if i not in image_path]
                label_path_all =  sorted( glob.glob(label_root+'/*'))
                self.label_path = [i for i in label_path_all if i not in label_path]

        # EBD dataset with 1605 images
        elif dataset == 'EBD':
            self.image_path = sorted( glob.glob('./data/EBD/image'+'/*'))
            self.label_path =  sorted( glob.glob('./data/EBD/gt'+'/*'))   

        elif dataset == 'CTCUG':
            self.image_path = sorted( glob.glob('./data/CTCUG/CTCUG_images'+'/*'))
            self.label_path =  sorted( glob.glob('./data/CTCUG/CTCUG_gt'+'/*')) 

        elif dataset == 'DUT':
            if not self.istrain:
                self.image_path = sorted( glob.glob('./data/DUT/DUT-DBD_Dataset/image'+'/*'))
                self.label_path =  sorted( glob.glob('./data/DUT/DUT-DBD_Dataset/gt'+'/*'))
            else:
                self.image_path = sorted( glob.glob('./data/DUT/DUT-DBD_Dataset/DUT600S_Training'+'/*'))
                self.label_path =  sorted( glob.glob('./data/DUT/DUT-DBD_Dataset/DUT600GT_Training'+'/*'))                

        self.transform = transform
        self.t_transform = t_transform
        if self.preloadimg:
            self.image_preload = []
            self.label_preload = []
            for i in range(len(self.image_path)):
                self.image_preload.append(Image.open(self.image_path[i]).convert("RGB"))
                self.label_preload.append(Image.open(self.label_path[i]).convert('L'))

    def __getitem__(self, item):
        image_name = []
        name = self.image_path[item].split('/')[-1]
        for i in range (len(self.image_path)):
            image_name.append(self.image_path[i].split('/')[-1])
        if self.preloadimg:
            image = self.image_preload[item]
            label = self.label_preload[item]
        else:
            image = Image.open(self.image_path[item]).convert("RGB")
            label = Image.open(self.label_path[item]).convert('L')
            image_name = image_name[item]
        if self.istrain:
            image,label = h_v_filp1(image,label)
            image,label = F.resize(image,(320,320)),F.resize(label,(320,320),interpolation=InterpolationMode.NEAREST)
        else:
            w,h = image.size
            image = F.resize(image,(320,320))

        if self.transform is not None:
            image = self.transform(image)
        if self.t_transform is not None:
            label = self.t_transform(label)
        
        if self.istrain:
            return image, label
        else:
            return image, label, (h,w), image_name

    def __len__(self):
        return len(self.image_path)

def get_loader(img_root, label_root, batch_size, dataset='Shi',mode='train', num_thread=4,preload=False):

    if mode == 'train':
        transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        t_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))
        ])

        dataset = ImageData(img_root, label_root, transform, t_transform, dataset, istrain=True,preloadimg=preload)
        data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_thread,drop_last=True)
        return data_loader
    
    elif mode=='val':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        t_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))
        ])

        dataset = ImageData(img_root, label_root, transform, t_transform, dataset, istrain=False, preloadimg=preload)
        data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_thread)
        return data_loader

