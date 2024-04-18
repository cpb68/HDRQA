import torch
import torch.nn as nn
import torch.nn.functional as F
from percentile import Percentile

class LDR_Seq(torch.nn.Module):
    def __init__(self):
        super(LDR_Seq, self).__init__()

    def get_luminance(self,img):
        if (img.shape[1] == 3):
            R = img[:, 2, :, :]
            G = img[:, 1, :, :]
            B = img[:, 0, :, :]
            Y = R * 0.212656 + G * 0.715158 + B * 0.072186
        elif (img.shape[1] == 1):
            Y = img
        else:
            print('Error: get_luminance: wrong matrix dimension')
        return Y

    def get_weight(self, img):
        v_channel = self.get_luminance(img)
        gamma = 1 - 0.1
        gamma1 = 0.1
        v_channel[v_channel < gamma1] = 10**(-5)
        v_channel[v_channel > gamma] = 10**(-5)
        v_channel[v_channel > gamma1] = 1
        return v_channel

    def generation(self, img):
        img_q = img[img > 0]
        b = 1 / 128
        min_v = torch.min(img_q)
        img[img <= 0] = min_v
        L = self.get_luminance(img)
        img_l = torch.log2(L)
        l_img = Percentile()(img_l[:].reshape(1, -1).squeeze(), [0, 100])
        l_min = l_img[0]
        l_max = l_img[1]
        l_min = l_min
        f8_stops = torch.ceil((l_max - l_min) / 8)
        l_start = l_min  # + (l_max - l_min - f8_stops * 8) / 2
        number = 8 * 3 * f8_stops / 8
        number = torch.tensor((number), dtype=torch.int64)

        result = []
        ek_value = []
        weight = []
        for i in range(number):
            k = (l_start + (8 / 3) * (i + 1))
            ek = 2 ** k
            img1 = (img / (ek + 0.00000001) - b) / (1 - b)
            imgClamp = img1.clamp(0.00000001, 1)
            imgP = (imgClamp) ** (1 / 2.2)
            all_len = len(imgP[imgP >= 0])
            white_len = len(imgP[imgP == 1])
            black_len = len(imgP[imgP <= 0.000232])
            pecent1 = white_len / all_len
            pecent2 = black_len / all_len
            if pecent1 < (3.5) / 4 and pecent2 <= 3 / 4:
                weight_temp = self.get_weight(imgP)
                result.append(imgP)
                weight.append(weight_temp)
                ek_value.append(ek)
        return result,weight,ek_value


class LDR_Seq_out(torch.nn.Module):
    def __init__(self):
        super(LDR_Seq_out, self).__init__()

    def generation(self, img, ek_value):
        b = 1 / 128
        number = len(ek_value)
        result = []

        for i in range(number):
            ek = ek_value[i]
            img1 = (img / (ek+1e-15) - b) / (1 - b)
            imgClamp = img1.clamp(1e-30, 1)
            imgP = (imgClamp) ** (1 / 2.2)
            result.append(imgP)
        return result

def MAE_weight(img2, ref, weight,length):
    mae_map = torch.abs((ref - img2))
    weight[weight < 1 / length] = 0
    mae_map1 = mae_map * weight.unsqueeze(1)
    mae = torch.sum(torch.sum(torch.sum(mae_map1,dim=1),dim=1),dim=1)/(3*torch.sum(torch.sum(torch.sum(weight1,dim=1),dim=1),dim=1))
    return torch.mean(mae)

class hdrLoss(torch.nn.Module):
    def __init__(self):
        super(hdrLoss, self).__init__()
        self.generate_GT = LDR_Seq()
        self.generate_out = LDR_Seq_out()

    def MAE_weight(self,img, ref, weight, length):
        mae_map = torch.abs((ref - img))
        weight[weight < 1 / length] = 0
        mae_map1 = mae_map * weight.unsqueeze(1)
        mae = torch.sum(torch.sum(torch.sum(mae_map1, dim=1), dim=1), dim=1) / (3 * torch.sum(torch.sum(torch.sum(weight.unsqueeze(1), dim=1), dim=1), dim=1))
        return torch.mean(mae)

    def forward(self, output, gt):
        gt_seq, weight, ek = self.generate_GT.generation(gt)
        output_seq = self.generate_out.generation(output, ek)
        length = len(gt_seq)
        loss = self.MAE_weight(torch.cat(output_seq,dim=0),torch.cat(gt_seq,dim=0),torch.cat(weight,dim=0),length)
        return loss

