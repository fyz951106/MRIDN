import numpy as np
import os
import torch
import random
import cv2
def padding(img_lq, img_gt, gt_size):
    h, w, _ = img_lq.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)

    if h_pad == 0 and w_pad == 0:
        return img_lq, img_gt

    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)

    if img_lq.ndim == 2:
        img_lq = np.expand_dims(img_lq, axis=2)
    if img_gt.ndim == 2:
        img_gt = np.expand_dims(img_gt, axis=2)
    return img_lq, img_gt
def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    return img

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,img_dir,gt_size,transform=None):
        super(MyDataset, self).__init__()
        self.img_dir = img_dir
        self.gt = sorted(os.listdir(os.path.join(img_dir, 'groundtruth')))
        self.img = sorted(os.listdir(os.path.join(img_dir, 'input')))
        self.transform = transform



        self.gt_size = gt_size

    def __len__(self):
        return len(self.img)
    def __getitem__(self,index):

        img = load_img(os.path.join(self.img_dir,'input',self.img[index]))
        gt = load_img(os.path.join(self.img_dir,'groundtruth',self.gt[index]))

        img,gt = padding(img,gt,self.gt_size)

        h_lq, w_lq, _ = img.shape
        h_gt, w_gt, _ = gt.shape

        top = random.randint(0, h_lq - self.gt_size)
        left = random.randint(0, w_lq - self.gt_size)

        img = img[top : top+self.gt_size, left:left + self.gt_size, ...]
        gt = gt[top : top+self.gt_size, left:left + self.gt_size, ...]
        if self.transform is not None:
            img = self.transform(img)
            gt = self.transform(gt)
        return img,gt


class MyDatasetval(torch.utils.data.Dataset):
    def __init__(self, img_dir,transform=None):
        super(MyDatasetval, self).__init__()
        self.img_dir = img_dir
        self.gt = sorted(os.listdir(os.path.join(img_dir, 'groundtruth')))
        self.img = sorted(os.listdir(os.path.join(img_dir, 'input')))
        self.transform = transform
        self.gt_size = 256

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):

        img = load_img(os.path.join(self.img_dir,'input',self.img[index]))
        gt = load_img(os.path.join(self.img_dir,'groundtruth',self.gt[index]))

        if self.transform is not None:
            img = self.transform(img)
            gt = self.transform(gt)



        return img, gt
def collate_fn(batch):

    img,gt = tuple(zip(*batch))
    img = torch.stack(img)
    gt = torch.stack(gt)

    return img,gt
