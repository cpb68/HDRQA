import os
import torch
import functools
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import imageio
import re
from torchpercentile import Percentile


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    return I

def image_seq_loader(img_seq_dir):
    img_seq_dir = os.path.expanduser(img_seq_dir)
    img_seq = []
    for root, _, fnames in sorted(os.walk(img_seq_dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                image_name = os.path.join(root, fname)
                im = np.array(Image.open(image_name).convert("RGB"))
                img_seq.append(torch.from_numpy(im/255).permute(2,0,1))#imageio.imread(image_name)/255
    return img_seq

def get_default_img_loader():
    return functools.partial(image_seq_loader)

class LDR_Seq(torch.nn.Module):
    def __init__(self):
        super(LDR_Seq, self).__init__()

    def get_luminance(self,img):
        if (img.shape[1] == 3):
            Y = img[:, 0, :, :] * 0.212656 + img[:, 1, :, :] * 0.715158 + img[:, 2, :, :] * 0.072186
        elif (img.shape[1] == 1):
            Y = img
        else:
            error('get_luminance: wrong matrix dimension')
        return Y

    def get_weight(self, img):
        v_channel = self.get_luminance(img)
        gamma = 1 - 0.1
        gamma1 = 0.1
        v_channel[v_channel < gamma1] = 10**(-5)
        v_channel[v_channel > gamma] = 10**(-5)
        v_channel[v_channel > gamma1] = 1
        return v_channel

    def generation(self,img):
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
        l_start = l_min #+ (l_max - l_min - f8_stops * 8) / 2
        number = 8 * 3 * f8_stops / 8
        number = torch.tensor((number), dtype=torch.int64)

        result = []
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
                result.append(imgP.squeeze())
                weight.append(weight_temp)
        return result,weight

class ImageDataset(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 ref_dir,
                 test=False,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        print('start loading csv data...')
        self.data = pd.read_csv(csv_file, sep='\t', header=None)
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.ref_dir = ref_dir
        self.test = test
        self.loader = get_loader()
        self.generate = LDR_Seq()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = os.path.join(self.img_dir, self.data.iloc[index, 0] + '.hdr')
        s = self.data.iloc[index, 0]
        a = r'(.*?)-'
        slotList = re.findall(a, s)
        image_ref_dir = os.path.join(self.ref_dir, slotList[1] + '.hdr')
        img = torch.from_numpy(imageio.imread(image_ref_dir)).permute(2, 0, 1).unsqueeze(0)
        imgref1, weight = self.generate.generation(img)
        imgref = torch.stack(imgref1, dim=0)
        imgh = imageio.imread(image_name)
        img = torch.from_numpy(imgh / (imgh.max())).permute(2, 0, 1)
        sample = {'img': img,'weight': weight,'imgref': imgref, 'img_name':self.data.iloc[index, 0]}
        return sample

    def __len__(self):
        return len(self.data.index)





