import torch
import os
import math
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision import transforms
from torch.nn import init
import skimage.color as color
from PIL import Image
import itertools
from torchvision import transforms
import imageio
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from ImageDataset import ImageDataset
import argparse
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from metrics import PSNR_weight,SSIM_weight,MAE_weight,dists_weight,lipis_weight
from DISTS_pytorch import DISTS
import pytorch_ssim
import time
import lpips

class Exposure(nn.Module):
    def __init__(self,batch_size):
        super(Exposure, self).__init__()
        self.b = 1 / 128
        ek = torch.randn((batch_size,1,1,1), requires_grad=True)
        self.ek = torch.nn.Parameter(ek)
        self.register_parameter("ek", self.ek)

    def forward(self, img):
        ekk = 2 ** self.ek
        img1 = (img/(ekk+0.00000001) - self.b) / (1 - self.b)
        imgClamp = torch.clamp(img1,0.000000000001,1)
        imgP = (imgClamp) ** (1 / 2.2)
        return imgP

class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        self.train_data = ImageDataset(csv_file=os.path.join('./','samples.txt'),
                                       img_dir=r'./image/test/',
                                       ref_dir=r'./image/ref/',
                                       test=False)

        self.train_loader = DataLoader(self.train_data,
                                       batch_size=1,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=4)
        self.metric = config.metric
        self.device = torch.device("cuda")
        if self.metric=="MAE":
            self.loss_fn = torch.nn.L1Loss()
            self.assess = MAE_weight()
        elif self.metric=="SSIM":
            self.loss_fn = pytorch_ssim.SSIM1(window_size=11)
            self.assess = SSIM_weight()
        elif self.metric == "PSNR":
            self.loss_fn = torch.nn.MSELoss()
            self.assess = PSNR_weight()
        elif self.metric == "DISTS":
            self.loss_fn = DISTS()
            self.assess = dists_weight()
        elif self.metric == "LPIPS":
            self.loss_fn = lpips.LPIPS1(net='vgg')
            self.assess = lipis_weight()

        self.loss_fn.to(device)
        self.assess.to(device)

        self.start_epoch = 0
        self.start_step = 0
        self.max_epochs = config.max_epochs
        self.model_name = "weight"

    def fit(self):
        for step, sample_batched in enumerate(self.train_loader, 0):
            img, imgref,weight1,self.img_name = sample_batched['img'], sample_batched['imgref'], sample_batched['weight'], sample_batched['img_name']
            imgrefSq = imgref.squeeze()
            length = len(imgrefSq)
            weight2 = torch.cat(weight1, dim=0)
            weight_all = torch.sum(weight2, dim=0)
            weight=weight2/weight_all
            self.model = Exposure(length)
            self.model.to(device)

            self.initial_lr = config.lr
            self.decay_ratio = config.decay_ratio
            self.decay_interval = config.decay_interval
            self.max_epochs = config.max_epochs

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_lr, weight_decay=5e-4)
            self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                                 last_epoch=self.start_epoch - 1,
                                                 step_size=self.decay_interval,
                                                 gamma=self.decay_ratio)

            assessmentV = self.train_single_image(imgrefSq,img,weight.squeeze(),length)

            # with open('./result.txt', '+a') as f:
            #     f.write("{}\t{}\n".format(self.img_name[0], '%.5f' % assessmentV))

    def train_single_image(self,img_Ref,img,weight,length):
        # initialize logging system
        std_loss_corrected = 0.0
        self.min_loss = 100000
        running_loss = 0
        # start training
        print('Adam learning rate: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
        self.model.train()
        #self.scheduler.step()
        starttime = time.time()
        for epoch in range(self.start_epoch, self.max_epochs):
            print('Adam learning rate: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
            img1 = img.to(self.device)
            img_Ref1 = img_Ref.to(self.device).float()
            weight1 = weight.to(self.device)
            img_gen = self.model(img1).float()
            self.optimizer.zero_grad()
            self.loss = self.loss_fn(img_Ref1,img_gen)*1000
            self.loss.backward()
            self.optimizer.step()

            delta_loss = abs(self.loss.data.item() - running_loss)
            if delta_loss < 10 ** (-80):
                break
            else:
                running_loss = self.loss.data.item()
                # loss_corrected = running_loss / (1 - beta ** local_counter)
                format_str = ('(E:%d,) [Loss = %.4f] [Std Loss = %.4f]')
                print(format_str % (epoch, running_loss, std_loss_corrected))
                self.scheduler.step()

        assementV = self.assess(img_gen, img_Ref1, weight1, length)
        endtime = time.time()
        print(endtime - starttime)

        return assementV.data.item()


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--metric", type=str, default="MAE")#choose from MAE,PSNR,SSIM,LPIPS,DISTS
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--decay_interval", type=int, default=2000)
    parser.add_argument("--decay_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=19901116)

    return parser.parse_args()

def main(cfg):
    t = Trainer(cfg)
    if cfg.train:
        t.fit()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = parse_config()
    main(config)




