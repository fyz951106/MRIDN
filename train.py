import argparse
import datetime
import logging
import math
import random
import time
from os import path as osp
import numpy as np
import sys

from MRIDN import UnetSE
import os
from dataset import MyDataset,MyDatasetval,collate_fn
from tqdm import tqdm
from utils import batch_PSNR,batch_SSIM
import torch
from torch import nn
from torchvision import transforms
from tensorboardX import SummaryWriter

# 解析命令行参数
# parser = argparse.ArgumentParser(description="MIRNet")  # 创建解析器，创建对象
parser = argparse.ArgumentParser(description="MRIDN")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")  # 添加参数
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=5e-5, help="Initial learning rate")
parser.add_argument("--data_dir", type=str, default="./SIDD_patches/train/", help="path of train dataset")
parser.add_argument("--val_dir", type=str, default="./SIDD_patches/val/", help="path of val dataset")
parser.add_argument("--log_dir", type=str, default="output", help="path of save results")
parser.add_argument("--patch_size", type=int, default=128, help="Training patch size")
parser.add_argument("--model", type=str, default="MRIDN", help='model for train')

opt = parser.parse_args()   # 解析添加的参数

# gpus = '0,1'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpus
def main():
    seed = 2020  # 2019
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = 'MRIDN'
    result_dir = os.path.join(opt.log_dir, 'results')
    model_dir = os.path.join(opt.log_dir, 'models')
    log_dir = os.path.join(opt.log_dir, 'log')
    model = UnetSE(inchannel=3,outchannel=32)
    new_lr = 5e-5
    warmup = True

    Train_ps = [128, 160, 192, 256]
    Train_bs = [8, 6, 1, 2]
    epoches = [60, 40, 20, 10]

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),lr=new_lr)

    pro_step = 0
    start_epoch = 1
    pro_epoch = epoches[pro_step]

    data_transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    # 加载数据
    train_dataset = MyDataset(opt.data_dir, gt_size=Train_ps[pro_step],transform=data_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=8,
                                             collate_fn=collate_fn)


    val_dataset = MyDatasetval(opt.val_dir,transform=data_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=8,
                                               collate_fn=collate_fn)

    psnr_writer = SummaryWriter('./tensorboard/'+model_name+'./psnr')
    ssim_writer = SummaryWriter('./tensorboard/'+model_name+'./ssim')
    train_loss_writer = SummaryWriter('./tensorboard/'+model_name+'./train_loss')
    val_loss_writer = SummaryWriter('./tensorboard/' + model_name + './val_loss')

    print('------------------------------------------------------------------------------')
    print("==> Start Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, sum(epoches) + 1))
    print('===> Loading datasets')

    num = 0
    val_num = 0
    for epoch in range(start_epoch, sum(epoches) + 1):
        epoch_start_time = time.time()


        epoch_loss = 0
        batch_loss = 0.
        model.train()
        train_loader = tqdm(train_loader,file=sys.stdout)
        eval_now = len(train_loader) // 4 - 1
        print(f"\nEvaluation after every {eval_now} Iterations !!!\n")
        for step,data in enumerate(train_loader):
            target, img = (data[1],data[0])
            target = target.to(device)
            img = img.to(device)


            output = model(img)
            output = torch.clip(output,0,1)
            loss = criterion(output,target)
            batch_loss += loss.item() / 200.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_loader.desc = "[train epoch {}] loss: {:.6f}".format(epoch, epoch_loss / (step + 1))
            if step % 200 == 0 and step > 0:
                print("Epoch: {}\tBatch: {}/{}\tTime: {:.4f}\tLoss: {:.4f}".format(epoch, step, len(train_loader),
                                                                                   time.time() - epoch_start_time,
                                                                                   batch_loss))
                num = num + 1
                train_loss_writer.add_scalar(model_name,batch_loss,num)
                batch_loss = 0

            if step % eval_now == 0 and step > 0:
                model.eval()
                val_num = val_num+1
                with torch.no_grad():
                    best_psnr = 0
                    best_epoch = 0
                    val_loader = tqdm(val_loader,file=sys.stdout)
                    psnr_val_rgb = []
                    ssim_val_rgb = []
                    for step,data_val in enumerate(val_loader):
                        img = data_val[0].to(device)
                        target = data_val[1].to(device)
                        output = model(img)
                        output = torch.clip(output, 0, 1)
                        psnr = batch_PSNR(output,target,1.)
                        ssim = batch_SSIM(output,target)
                        psnr_val_rgb.append(psnr)
                        ssim_val_rgb.append(ssim)
                        val_loader.desc = "[val epoch {}] psnr: {:.3f}".format(epoch, psnr)
                    psnr_val_rgb = sum(psnr_val_rgb) / len(psnr_val_rgb)
                    ssim_val_rgb = sum(ssim_val_rgb) / len(ssim_val_rgb)
                    psnr_writer.add_scalar(model_name,psnr_val_rgb,val_num)
                    ssim_writer.add_scalar(model_name,ssim_val_rgb,val_num)
                    if psnr_val_rgb > best_psnr:
                        best_psnr = psnr_val_rgb
                        best_ssim = ssim_val_rgb
                        best_epoch = epoch
                        best_iter = step
                        model_state = {
                            'state': model.state_dict(),
                            'epoch': epoch
                        }
                        ckpt_path = './ckpt/' + model_name
                        if not os.path.isdir('./ckpt'):
                            os.mkdir('./ckpt')
                        if not os.path.isdir(ckpt_path):
                            os.mkdir(ckpt_path)
                        torch.save(model_state, os.path.join(ckpt_path, 'model_%04d.pth' % epoch))
                    print(
                        "[Ep %d it %d\t PSNR: %.4f\t SSIM: %.4f\t] ----  [best_Ep %d best_it %d Best_PSNR %.4f Best_SSIM %.4f] " % (
                            epoch, step, psnr_val_rgb, ssim_val_rgb, best_epoch, best_iter, best_psnr, best_ssim))

            model.train()


        if epoch == pro_epoch and epoch < sum(epoches):
            pro_step += 1
            pro_epoch += epoches[pro_step]
            train_dataset = MyDataset(opt.data_dir, gt_size=Train_ps[pro_step])
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=1,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       num_workers=8,
                                                       collate_fn=collate_fn)

if __name__ == '__main__':
    main()