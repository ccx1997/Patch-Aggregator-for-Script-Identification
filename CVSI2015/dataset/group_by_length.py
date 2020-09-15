# -*- coding: utf-8 -*-
"""Group images according to their w (with the same h)
"""
import cv2
import os
import math


def grp_cvsi(grp_dir='z_grp_ccx', trainset=True):
    """4 Groups is proposed where the width is 32, 64, 128, 256 respectively
    aspect ratio: 1,  2,  4
    belongings:    3/2, 3
    """
    ars = [1, 2, 4]
    root_dir = '/workspace/datasets/script_id/CVSI2015/'
    sub_dir = 'TrainDataset_CVSI2015' if trainset else 'ValidationDataset_CVSI2015'
    dir_main = os.path.join(root_dir, sub_dir)   # '/workspace/datasets/script_id/CVSI2015/TrainDataset_CVSI2015'
    grp_dir = os.path.join(dir_main, grp_dir)   # '....../CVSI2015/TrainDataset_CVSI2015/z_grp_ccx'
    if not os.path.exists(grp_dir):
        os.makedirs(grp_dir)
    class_names = ('Arabic', 'Bengali', 'English', 'Gujrathi', 'Hindi', 'Kannada', 'Oriya',
                   'Punjabi', 'Tamil', 'Telegu')
    savings = []
    for ar in iter(ars):
        savings.append(open(os.path.join(grp_dir, 'grp' + str(ar*32) + '.txt'), 'w'))
    statistics = dict()
    for label, cn in enumerate(class_names):
        dir_now = os.path.join(dir_main, cn)    # '....../CVSI2015/TrainDataset_CVSI2015/Arabic'
        img_list = os.listdir(dir_now)  # ['ARB_001.jpg', ...]
        for img_name in iter(img_list):
            image = cv2.imread(os.path.join(dir_now, img_name), 1)
            h, w = image.shape[:2]
            aspect_ratio = w / h
            tmp = math.ceil(math.log2(aspect_ratio/3.)) + 1
            ar_belong = 2 ** tmp
            if ar_belong not in statistics.keys():
                statistics[ar_belong] = 1
            else:
                statistics[ar_belong] += 1
            tmp = max(min(tmp, 2), 0)
            savings[tmp].writelines(os.path.join(cn, img_name) + ' ' + str(label+1) + '\n')
    for fi in savings:
        fi.close()
    return statistics


if __name__ == "__main__":
    trainset = True
    statistics = grp_cvsi(trainset=trainset)
    dataset = 'Trainset' if trainset else "ValidationSet"
    print("There are {0} images in the {1}".format(sum(statistics.values()), dataset))
    print("Format: aspect ratio: num_imgs")
    print(statistics)
    print("---finished---")
