# -*- coding: utf-8 -*-
"""Group images according to their w (with the same h)
"""
import cv2
import os

def grp_siw(grp_dir='z_grp_ccx', trainset=True):
    """4 Groups is proposed where the width is 64, 128, 256, 512 respectively"""
    ws = [64, 128, 256, 512]
    root_dir = '/workspace/datasets/script_id/SIW-13/'
    grp_dir = os.path.join(root_dir, grp_dir)
    if not os.path.exists(grp_dir):
        os.makedirs(grp_dir)
    list_txt, state = ("train-list.txt", 'trainset') if trainset else ("test-list.txt", 'testset')
    state_dir = os.path.join(grp_dir, state)
    if not os.path.exists(state_dir):
        os.makedirs(state_dir)
    g = []
    for wg in ws:
        g.append(open(os.path.join(state_dir, 'grp' + str(wg) + '.txt'), 'w'))
    mapping = {0: 0, 1: 1, 2: 2, 3: 2}
    with open(os.path.join(root_dir, list_txt), 'r') as full:
        while True:
            line = full.readline()
            if not line:
                break
            img_name = line.split()[0]
            image = cv2.imread(os.path.join(root_dir, img_name), 1)
            w = w_resize(image)
            tmp = w // 96
            if tmp in mapping.keys():
                idx = mapping[tmp]
            else:
                idx = 3
            g[idx].writelines(line)
    for gi in g:
        gi.close()
    print("---finished---")

def w_resize(img, newh=32):
    h, w = img.shape[:2]
    neww = int(newh * w / h)
    return neww


if __name__ == "__main__":
    grp_siw(trainset=False)
