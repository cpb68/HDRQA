import os
import sys
import math
import torch
import numpy as np
import cv2
import pytorch_ssim
import imageio
import lpips
from DISTS_pt import DISTS

class lipis_weight(torch.nn.Module):
    def __init__(self):
        super(lipis_weight, self).__init__()
        self.name = "lipis_weight"
        self.lipis = lpips.LPIPS(net='vgg')

    def __call__(self, img2,ref,weight,length):
        weight[weight < 1 / length] = 0
        weight1 = weight.unsqueeze(1)
        with torch.no_grad():
            score = torch.mean(self.lipis(ref, img2,weight1))
        return score

class dists_weight(torch.nn.Module):
    def __init__(self):
        super(dists_weight, self).__init__()
        self.name = "dists_weight"
        self.dists = DISTS()

    def __call__(self, img2,ref,weight,length):
        with torch.no_grad():
            score = torch.mean(self.dists(ref, img2))
        return score

class SSIM(torch.nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.name = "SSIM"
        self.ssim = pytorch_ssim.SSIM(window_size = 11)

    @staticmethod
    def __call__(img,ref):
        ssim,ssim_map = self.ssim(ref,img)
        return ssim

class SSIM_weight(torch.nn.Module):
    def __init__(self):
        super(SSIM_weight, self).__init__()
        self.name = "SSIM_weight"
        self.ssim1 = pytorch_ssim.SSIM(window_size = 11)

    # @staticmethod
    def __call__(self,img,ref, weight,length):
        ss, ssim_map = self.ssim1(ref, img)
        weight[weight < 1 / length] = 0
        weight1 = weight.unsqueeze(1)
        ssim_map1 = ssim_map * weight1
        ssim = torch.sum(torch.sum(torch.sum(ssim_map1,dim=1),dim=1),dim=1)/  (3*torch.sum(torch.sum(torch.sum(weight1,dim=1),dim=1),dim=1))
        return torch.mean(ssim)

class PSNR(torch.nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
        self.name = "PSNR"

    @staticmethod
    def __call__(img2,ref):
        max_value = ref.max()
        mse = torch.mean((ref - img2) ** 2)
        return 20 * torch.log10(max_value / torch.sqrt(mse))

class MAE(torch.nn.Module):
    def __init__(self):
        super(MAE, self).__init__()
        self.name = "MAE"

    @staticmethod
    def __call__(img2,ref):
        mae = torch.mean(torch.abs((ref - img2)))
        return mae

class MAE_weight(torch.nn.Module):
    def __init__(self):
        super(MAE_weight, self).__init__()
        self.name = "MAE_weight"

    @staticmethod
    def __call__(img2, ref, weight,length):
        mae_map = torch.abs((ref - img2))
        weight[weight < 1 / length] = 0
        weight1 = weight.unsqueeze(1)
        mae_map1 = mae_map * weight1
        mae = torch.sum(torch.sum(torch.sum(mae_map1,dim=1),dim=1),dim=1)/(3*torch.sum(torch.sum(torch.sum(weight1,dim=1),dim=1),dim=1))
        return torch.mean(mae)

class PSNR_weight(torch.nn.Module):
    def __init__(self):
        super(PSNR_weight, self).__init__()
        self.name = "PSNR_weight"

    @staticmethod
    def __call__(img2,ref, weight,length):
        weight[weight < 1 / length] = 0
        error = (ref.float() - img2.float()) ** 2
        weight1 = weight.unsqueeze(1)
        error_weight = error * weight1
        mse=torch.sum(torch.sum(torch.sum(error_weight,dim=1),dim=1),dim=1)/(3*torch.sum(torch.sum(torch.sum(weight1,dim=1),dim=1),dim=1))
        max_value = torch.max(torch.max(torch.max(ref,dim=1)[0],dim=1)[0],dim=1)[0]
        return torch.mean(20 * torch.log10(max_value / torch.sqrt(mse)))