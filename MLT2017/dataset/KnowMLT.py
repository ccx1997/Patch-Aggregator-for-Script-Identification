"""Have a look at the distribution about the data in MLT Task2
"""
import cv2
import os
import linecache
import math
import warnings
warnings.filterwarnings('ignore')

class KnowU(object):
    """To Know the distribution about the data across classes and aspect ratio respectively
    """
    def __init__(self, root, list_dir, part_dir, state):
        self.root = root
        self.listdir = os.path.join(root, list_dir)
        self.prefix = os.path.join(root, part_dir)
        self.istrainset = (state == 0)
        with open(self.listdir, 'r', encoding='utf-8') as file:
            self.len = len(file.readlines())
        self.scripts = ["Arabic", "Latin", "Chinese", "Japanese", "Korean", "Bangla", "Symbols"]
        self.num_classes = len(self.scripts)

    def __call__(self):
        num_per_classes = list(0 for _ in range(self.num_classes))
        num_ar = dict()
        for i in range(self.len):
            line = linecache.getline(self.listdir, i+1)
            line = line.strip().split(',')  # e.g. i=100, ['word_100.png', 'Arabic', 'فنادق']
            num_per_classes[self.scripts.index(line[1])] += 1
            if self.istrainset:
                id_img = int(line[0].split('.')[0].split('_')[1])
                tail = str((id_img-1) // 23000 + 1)
                img_path = os.path.join(self.prefix+tail, line[0])
            else:
                img_path = os.path.join(self.prefix, line[0])
            img = cv2.imread(img_path, 1)
            if img is None:
                print(line)
            h, w = img.shape[:2]
            aspect_ratio = w / h
            # fixed aspect ratios: 1/8,  1/4,  1/2,  1,  2,  4, 8, 16
            # intervals:              3/16, 3/8, 3/4, 3/2, 3,  6, 12, 24
            tmp = math.pow(2, math.ceil(math.log(aspect_ratio / 3, 2))) * 3
            if tmp not in num_ar.keys():
                num_ar[tmp] = 1
            else:
                num_ar[tmp] += 1
        return num_per_classes, num_ar


if __name__ == "__main__":
    root_dir = '/workspace/datasets/script_id/ICDAR17'
    set_names = ['TrainingSet', 'ValidationSet']
    state = 1   # 0: trainset; 1: validationset
    DataSet = os.path.join(root_dir, set_names[state])
    s = 'training' if state == 0 else 'validation'
    list_dir = 'ch8_'+s+'_word_gt_v2/gt.txt'
    part_dir = 'ch8_'+s+'_word_images_gt_part_' if state == 0 else ''

    f = KnowU(DataSet, list_dir=list_dir, part_dir=part_dir, state=state)
    print(f())

