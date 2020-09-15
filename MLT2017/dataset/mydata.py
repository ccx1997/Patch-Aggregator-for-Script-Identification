# By Changxu Cheng, HUST

import os
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from torchvision import transforms
import random
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class MyData(Dataset):
    """A dataset to feed dataloader
    Example of args: root_dir='/dataset/ccx/ICDAR2017'
    list_txt='z_grp_ccx/trainset/grp81.txt'
    prefix='TrainingSet/ch8_training_word_images_gt_part_'
    """
    def __init__(self, root_dir, list_txt, prefix, classes, aspect_ratio, transf, aug=False):
        super(MyData, self).__init__()
        self.imglabels = pd.read_table(os.path.join(root_dir, list_txt), header=None)
        self.rootdir = root_dir
        self.prefix = os.path.join(root_dir, prefix)
        self.classes = classes
        self.ar = aspect_ratio
        self.short = 64
        self.transf = transf
        self.aug = aug

    def __len__(self):
        return self.imglabels.shape[0]

    def read_from_line(self, line):
        line = line.strip().split(',')
        img_name, label = line[:2]
        label = self.classes.index(label)
        return img_name, label

    def image_process(self, image):
        if self.aug:
            image = random_augmentation(image)
        if self.ar >= 1:
            newSize = (int(self.short * self.ar), self.short)
        else:
            newSize = (self.short, int(self.short/self.ar))
        image = cv2.resize(image, newSize)
        return self.transf(image)

    def the_image_path(self, img_name):
        if 'TrainingSet' in self.prefix:
            id_img = int(img_name.split('.')[0].split('_')[1])
            tail = str((id_img - 1) // 23000 + 1)
            return os.path.join(self.prefix + tail, img_name)
        else:
            return os.path.join(self.prefix, img_name)

    def __getitem__(self, idx):    # Get through here every iteration when refreshing the dataloader
        imglb = self.imglabels.iloc[idx, 0]
        img_name, label = self.read_from_line(imglb)
        image = cv2.imread(self.the_image_path(img_name), 1)
        image = self.image_process(image)
        return image, label


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
    return image


class ImageTransfer(object):
    """crop, add noise, change contrast, color jittering"""
    def __init__(self, image):
        """image: a ndarray with size [h, w, 3]"""
        self.image = image

    def slight_crop(self):
        h, w = self.image.shape[:2]
        k0 = 6 if w / h > 3 else 8
        k = random.randint(k0, 10) / 10
        ch, cw = int(h * 0.9), int(w * k)     # cropped h and w
        hs = random.randint(0, h - ch)      # started loc
        ws = random.randint(0, w - cw)
        return self.image[hs:hs+ch, ws:ws+cw]

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
        short = min(h, w)
        gate = int(short * 0.3)
        mrg = []
        for _ in range(8):
            mrg.append(random.randint(0, gate))
        pts1 = np.float32(
            [[mrg[0], mrg[1]], [w - 1 - mrg[2], mrg[3]], [mrg[4], h - 1 - mrg[5]], [w - 1 - mrg[6], h - 1 - mrg[7]]])
        pts2 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(self.image, M, (w, h))

    def change_hsv(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        s = random.random()
        def ch_h():
            dh = random.randint(2, 11) * random.randrange(-1, 2, 2)
            img[:, :, 0] = (img[:, :, 0] + dh) % 180
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


def getloaders(root_dir, list_dir, prefix, classes, aspect_ratios, txts, transf, shuffle=True, bs=32, aug=False):

    list_dir = os.path.join(root_dir, list_dir)
    loaders = []
    for i in range(len(aspect_ratios)):
        if aspect_ratios[i] > 1:
            bs = int(bs / (aspect_ratios[i] // 16 + 1))
        list_txt = os.path.join(list_dir, txts[i])
        mydataset = MyData(root_dir, list_txt, prefix, classes, aspect_ratios[i], transf, aug=aug)
        loaders.append(DataLoader(mydataset, batch_size=bs, shuffle=shuffle, num_workers=2))
    return loaders


if __name__ == "__main__":
    root_dir = '/workspace/datasets/script_id/ICDAR17'
    trainset = True
    list_dir = "z_grp_ccx/trainsetSplit/" if trainset else "z_grp_ccx/valset/"
    prefix = "TrainingSet/ch8_training_word_images_gt_part_" if trainset else "ValidationSet"
    classes = ["Arabic", "Latin", "Chinese", "Japanese", "Korean", "Bangla", "Symbols"]
    aspect_ratios = [1/8, 1/4, 1/2, 1, 2, 4, 8, 16]
    txts = ["grp81.txt", "grp41.txt", "grp21.txt", "grp1.txt", "grp2.txt", "grp4.txt", "grp8.txt", "grp16.txt"]
    txts_Latin = ["grpLatin81.txt", "grpLatin41.txt", "grpLatin21.txt", "grpLatin1.txt",
                  "grpLatin2.txt", "grpLatin4.txt", "grpLatin8.txt", "grpLatin16.txt"]
    transf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    myloaders = getloaders(root_dir, list_dir, prefix, classes, aspect_ratios, txts, transf, shuffle=True, bs=16,
                           aug=True)
    myloaders_Latin = getloaders(root_dir, list_dir, prefix, classes, aspect_ratios, txts_Latin, transf,
                                 shuffle=True, bs=16, aug=True)
    print(len(myloaders), len(myloaders_Latin))
    print([len(myloader) for myloader in myloaders])
    dataiter = iter(myloaders_Latin[4])
    # images, label = dataiter.next()
    images, label = dataiter.next()
    print(images[2].size(), label)
    # Save some batches
    images = tv.utils.make_grid(images)
    images = (images * 0.5 + 0.5) * 255
    cv2.imwrite("batch2.jpg", images.numpy().transpose(1, 2, 0))
    print("Note: batch display saved in file batch.jpg")
