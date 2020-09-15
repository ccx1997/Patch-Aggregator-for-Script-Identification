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
    def __init__(self, root_dir, list_txt, tolength, transf, aug=False, pad=True, keep=False,
                 fusion=False, multilabel=False, triplet=False):
        super(MyData, self).__init__()
        self.imglabels = pd.read_table(os.path.join(root_dir, list_txt),
                                       delim_whitespace=True, header=None)
        self.rootdir = root_dir
        self.tolength = tolength
        self.transf = transf
        self.aug = aug
        self.pad = pad
        self.keep = keep
        self.fusion = fusion
        self.multilabel = multilabel
        self.triplet = triplet

    def __len__(self):
        return self.imglabels.shape[0]

    def random_select(self, interval=None):
        if interval is None:
            interval = (0, self.__len__() - 1)
        id_seed = random.randint(*interval)
        return self.imglabels.iloc[id_seed, :]

    def image_process(self, image):
        if self.aug:
            image = random_augmentation(image)
        if not self.keep:
            if self.pad:
                image, _ = resize_h(image, newh=32)
                image = add_padding(image, self.tolength)
            else:
                image = cv2.resize(image, (self.tolength, 32))
        else:
            image, _ = resize_h(image, 32)
        return self.transf(image)

    def __getitem__(self, idx):    # Get through here every iteration when refreshing the dataloader
        img_name, label = self.imglabels.iloc[idx, :]
        img_dir = os.path.join(self.rootdir, img_name)
        image = cv2.imread(img_dir, 1)
        image = self.image_process(image)
        if self.fusion:
            label = label_fusion(label)
        if self.triplet:
            image_pos, image_neg = None, None
            interval = None
            while (image_pos is None) or (image_neg is None):
                random_imgname, random_label = self.random_select(interval)
                if random_label != label:
                    image_neg = cv2.imread(os.path.join(self.rootdir, random_imgname), 1)
                    label_neg = random_label
                    interval = (max(0, idx-150), min(self.__len__()-1, idx+150))
                else:
                    image_pos = cv2.imread(os.path.join(self.rootdir, random_imgname), 1)
                    label_pos = random_label
            return image, label-1, self.image_process(image_pos), label_pos-1, \
                   self.image_process(image_neg), label_neg-1

        if self.multilabel:
            label_list = [(0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 1, 0),
                          (3, 2, 0), (4, 0, 0), (2, 0, 2), (5, 0, 0),
                          (6, 0, 0), (7, 0, 0), (3, 3, 0), (8, 0, 0),
                          (9, 0, 0)]
            return image, torch.tensor((*label_list[label-1], int(label-1)))

        return image, label - 1

def label_fusion(label):
    """(Chi, Jap) => Chi; (Eng, Gre, Rus) => Eng     label: 1-13"""
    if label == 7:
        return 3
    elif label in [5, 11]:
        return 4
    else:
        return label

def resize_h(img, newh=32):
    h, w = img.shape[:2]
    neww = int(newh * w / h)
    img = cv2.resize(img, (neww, newh))
    return img, neww

def add_padding(img, towidth):
    h, w, c = img.shape
    img1 = np.zeros((h, towidth, c))
    img1[:, towidth-w:towidth] = img
    return img1

def random_augmentation(image, allow_crop=True):
    f = ImageTransfer(image)
    seed = random.randint(0, 5)     # 0: original image used
    switcher = random.random() if allow_crop else 1.0
    if seed == 1:
        image = f.add_noise()
    elif seed == 2:
        image = f.change_contrast()
    elif seed == 3:
        image = f.change_hsv()
    elif seed >= 4:
        f1 = ImageTransfer(f.add_noise())
        f2 = ImageTransfer(f1.change_hsv())
        image = f2.change_contrast()
    if switcher < 0.4:
        fn = ImageTransfer(image)
        image = fn.slight_crop()
    elif switcher < 0.8:
        fn = ImageTransfer(image)
        image = fn.perspective_transform()
    # if random.random() < 0.9:
    #     fn2 = ImageTransfer(image)
    #     image = fn2.seq_exchange()
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

    def perspective_transform(self):
        h, w = self.image.shape[:2]
        gate = int(h * 0.3)
        mrg = []
        for _ in range(8):
            mrg.append(random.randint(0, gate))
        pts1 = np.float32([[mrg[0], mrg[1]], [w-1-mrg[2], mrg[3]], [mrg[4], h-1-mrg[5]], [w-1-mrg[6], h-1-mrg[7]]])
        pts2 = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(self.image, M, (w, h))

    def change_hsv(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        s = random.random()
        def ch_h():
            dh = random.randint(2, 11) * random.randrange(-1, 2, 2)
            img[:, :, 0] = (img[:,:,0] + dh) % 180
        def ch_s():
            ds = random.random() * 0.25 + 0.7
            img[:, :, 1] = ds * img[:, :, 1]
        def ch_v():
            dv = random.random() * 0.35 + 0.6
            img[:, :, 2] = dv * img[:, :, 2]
        if s < 0.25:
            ch_h()
        elif s < 0.50:
            ch_s()
        elif s < 0.75:
            ch_v()
        else:
            ch_h()
            ch_s()
            ch_v()
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

def getloaders(root_dir, mid_dir, widths, txts, transf, shuffle=True, bs=32, aug=False, pad=False,
               multilabel=False, fusion=False, triplet=False):
    mid_dir = os.path.join(root_dir, mid_dir)
    keep = (bs == 1)
    loaders = []
    for i in range(len(widths)):
        list_txt = os.path.join(mid_dir, txts[i])
        mydataset = MyData(root_dir, list_txt, widths[i], transf, aug=aug, pad=pad, keep=keep,
                           multilabel=multilabel, fusion=fusion, triplet=triplet)
        loaders.append(DataLoader(mydataset, batch_size=bs, shuffle=shuffle, num_workers=4))
    return loaders

if __name__ == "__main__":
    root_dir = '/workspace/datasets/script_id/SIW-13/'
    mid_dir = "z_grp_ccx/testset/"
    # list_txts =['train-list.txt', 'test-list.txt']
    # train_distribution = [502, 583, 798, 721, 518, 742, 715, 529, 1061, 692, 531, 1722, 677]
    widths = [64, 128, 256, 512]
    txts = ["grp64.txt", "grp128.txt", "grp256.txt", "grp512.txt"]
    transf = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    myloaders = getloaders(root_dir, mid_dir, widths, txts, transf, shuffle=True, bs=16, aug=True,
                           pad=False, multilabel=False, triplet=False)
    print(len(myloaders))
    print([len(myloader) for myloader in myloaders])
    dataiter = iter(myloaders[1])
    images, label = dataiter.next()
    # images, label, _, label_pos, _, label_neg = dataiter.next()
    print(images[2].size(), label)
    # print(label_pos, label_neg)
    # Save some batches
    images = tv.utils.make_grid(images)
    images = (images * 0.5 + 0.5) * 255
    cv2.imwrite("batch128.jpg", images.numpy().transpose(1, 2, 0))
    print("Note: batch display saved in file batch.jpg")
