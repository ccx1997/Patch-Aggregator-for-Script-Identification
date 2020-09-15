# -*- coding: utf-8 -*-
# # By Changxu Cheng, HUST

import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision as tv
from torchvision import transforms
import random
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class MyData(Dataset):
    """A dataset to feed dataloader
    Args:

    """
    def __init__(self, root_dir, list_txt, tolength, transf, aug=False):
        super(MyData, self).__init__()
        self.imglabels = pd.read_table(os.path.join(root_dir, list_txt), delim_whitespace=True, header=None)
        self.rootdir = root_dir
        self.tolength = tolength
        self.transf = transf
        self.aug = aug

    def __len__(self):
        return self.imglabels.shape[0]

    def image_process(self, image):
        if self.aug:
            image = random_augmentation(image)
        image = cv2.resize(image, (self.tolength, 32))
        image = image / 255
        image = torch.from_numpy(image)
        image = image.unsqueeze(0).to(torch.float32)
        return self.transf(image)

    def __getitem__(self, idx):    # Get through here every iteration when refreshing the dataloader
        img_name, label = self.imglabels.iloc[idx, :]
        img_dir = os.path.join(self.rootdir, img_name)
        image = cv2.imread(img_dir, 0)
        image = self.image_process(image)
        return image, label - 1


def random_augmentation(image, allow_crop=True):
    f = ImageTransfer(image)
    seed = random.randint(0, 4)     # 0: original image used
    switcher = random.random() if allow_crop else 1.0
    if seed == 1:
        image = f.add_noise()
    elif seed == 2:
        image = f.change_contrast()
    elif seed >= 3:
        f1 = ImageTransfer(f.add_noise())
        image = f1.change_contrast()
    if switcher < 0.5:
        fn = ImageTransfer(image)
        image = fn.slight_crop()
    return image


class ImageTransfer(object):
    """crop, add noise, change contrast, color jittering"""
    def __init__(self, image):
        """image: a ndarray with size [h, w, 3]"""
        self.image = image

    def slight_crop(self):
        h, w = self.image.shape[:2]
        k0 = 5 if w / h > 3 else 7
        k = random.randint(k0, 10) / 10
        ch, cw = int(h * 0.95), int(w * k)     # cropped h and w
        hs = random.randint(0, h - ch)      # started loc
        ws = random.randint(0, w - cw)
        return self.image[hs:hs+ch, ws:ws+cw]

    def seq_exchange(self):
        # Change the sequence of patches
        h, w = self.image.shape[:2]
        interval = random.choice([1.5, 1.7, 2.0, 2.2])
        if w / h > interval * 2:
            patches = []
            for i in range(w // int(interval * h)):
                patches.append(self.image[:, int(i*interval*h):int((i+1)*interval*h)])
            random.shuffle(patches)
            return np.concatenate(patches, axis=1)
        else:
            return self.image

    def add_noise(self):
        img = self.image * (np.random.rand(*self.image.shape) * 0.3 + 0.7)
        img = img.astype(np.uint8)
        return img

    def change_contrast(self):
        if random.random() < 0.5:
            k = random.randint(6, 9) / 10.0
        else:
            k = random.randint(11, 14) / 10.0
        b = 128 * (k - 1)
        img = self.image.astype(np.float)
        img = k * img - b
        img = np.maximum(img, 0)
        img = np.minimum(img, 255)
        img = img.astype(np.uint8)
        return img


def getloaders(root_dir, mid_dir, widths, txts, transf, shuffle=True, bs=32, aug=False):
    mid_dir = os.path.join(root_dir, mid_dir)
    loaders = []
    for i in range(len(widths)):
        list_txt = os.path.join(mid_dir, txts[i])
        mydataset = MyData(root_dir, list_txt, widths[i], transf, aug=aug)
        loaders.append(DataLoader(mydataset, batch_size=bs, shuffle=shuffle, num_workers=3))
    return loaders


if __name__ == "__main__":
    root_dir = '/workspace/datasets/script_id/CVSI2015/TrainDataset_CVSI2015'
    mid_dir = "z_grp_ccx"
    widths = [32, 64, 128]
    txts = ["grp32.txt", "grp64.txt", "grp128.txt"]
    transf = transforms.Normalize((0.5,), (0.5,))
    myloaders = getloaders(root_dir, mid_dir, widths, txts, transf, shuffle=True, bs=16, aug=False)
    print(len(myloaders))
    print([len(myloader) for myloader in myloaders])
    dataiter = iter(myloaders[1])
    images, label = dataiter.next()
    print(images[2].size(), label)
    # Save some batches
    images = tv.utils.make_grid(images)
    images = (images * 0.5 + 0.5) * 255
    cv2.imwrite("batch64.jpg", images.numpy().transpose(1, 2, 0))
    print("Note: batch display saved in file batch.jpg")
