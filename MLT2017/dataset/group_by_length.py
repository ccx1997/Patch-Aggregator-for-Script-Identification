"""Group images according to their w (with the same h)
"""
import cv2
import os
import math

def grp_mlt(grp_dir='z_grp_ccx', trainset=True):
    """8 Groups is proposed where the aspect ratio are 1/8, 1/4, 1/2, 1, 2, 4, 8, 16 respectively"""
    ws = [81, 41, 21, 1, 2, 4, 8, 16]    # represent [1/8, 1/4, 1/2, 1, 2, 4, 8, 16] respectively
    root_dir = '/workspace/datasets/script_id/ICDAR17/'
    grp_dir = os.path.join(root_dir, grp_dir)
    if not os.path.exists(grp_dir):
        os.makedirs(grp_dir)
    list_txt, state = ("TrainingSet/ch8_training_word_gt_v2/gt.txt", 'trainsetSplit') if trainset else\
        ("ValidationSet/ch8_validation_word_gt_v2/gt.txt", 'valset')
    state_dir = os.path.join(grp_dir, state)
    if not os.path.exists(state_dir):
        os.makedirs(state_dir)
    g = []
    g_Latin = []
    for wg in ws:
        g.append(open(os.path.join(state_dir, 'grp' + str(wg) + '.txt'), 'w', encoding='utf-8'))
        if trainset:
            g_Latin.append(open(os.path.join(state_dir, 'grpLatin' + str(wg) + '.txt'), 'w', encoding='utf-8'))
    prefix = os.path.join(root_dir, 'TrainingSet/ch8_training_word_images_gt_part_') if trainset else \
        os.path.join(root_dir, 'ValidationSet')
    with open(os.path.join(root_dir, list_txt), 'r', encoding='utf-8') as full:
        while True:
            line0 = full.readline()
            if not line0:
                break
            line = line0.strip().split(',')  # e.g. i=100, ['word_100.png', 'Arabic', '?????', '...']
            img_name = line[0]
            if trainset:
                id_img = int(img_name.split('.')[0].split('_')[1])
                tail = str((id_img - 1) // 23000 + 1)
                prefix1 = prefix + tail
            else:
                prefix1 = prefix
            image = cv2.imread(os.path.join(prefix1, img_name), 1)
            h, w = image.shape[:2]
            aspect_ratio = w / h
            tmp = max(min(math.ceil(math.log(aspect_ratio / 3, 2)), 3), -4) + 4
            if line[1] == 'Latin' and trainset:
                g_Latin[tmp].writelines(line0)
            else:
                g[tmp].writelines(line0)
    for gi in g:
        gi.close()
    print("---finished---")

def unite(file_name):
    # To make the same class entry together
    file = open(file_name, 'r', encoding='utf-8')
    classes = ["Arabic", "Latin", "Chinese", "Japanese", "Korean", "Bangla", "Symbols"]
    nc = len(classes)
    entry = [[] for _ in range(nc)]
    for line in file:
        if line == '\n':
            continue
        line1 = line.strip().split(',')
        label = classes.index(line1[1])
        entry[label].append(line[:-1])
    file.close()
    with open(file_name, 'w', encoding='utf-8') as file:
        for i in range(nc):
            if len(entry[i]) == 0:
                continue
            file.write('\n'.join(entry[i]))
            file.write('\n')


if __name__ == "__main__":
    grp_mlt(trainset=True)
    txts = ["grp81.txt", "grp41.txt", "grp21.txt", "grp1.txt", "grp2.txt", "grp4.txt", "grp8.txt", "grp16.txt"]
    for txt in iter(txts):
        unite(os.path.join("/workspace/datasets/script_id/ICDAR17/z_grp_ccx/trainsetSplit", txt))
