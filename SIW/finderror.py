# -*- coding: utf-8 -*-
# By Changxu Cheng, HUST

import torch
import os
import numpy as np
import argparse
import time
import demo
from model import vgg
from draw.drawConfmat import plotCM

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='gspa', help='choose the network')
parser.add_argument('--model', type=str, default="./params/tmp04.pkl",
                    help="Choose the pretrained model parameters to drive")
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
root_dir = '/workspace/datasets/script_id/SIW-13/'
mid_dir = os.path.join(root_dir, "z_grp_ccx/testset/")
classes = ('Arabic', 'Cambodian', 'Chinese', 'English', 'Greek', 'Hebrew', 'Japanese', 'Kannada',
           'Korean', 'Mongolian', 'Russian', 'Thai', 'Tibetan')
num_classes = len(classes)
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

if opt.net == 'gs':
    net = vgg.CNN(num_classes, pretrain_use=opt.model)
elif opt.net == 'gspa':
    net = vgg.GSPA(num_classes, pretrain_use=opt.model)
net.to(device)
net.eval()
# Find error samples and predict
total = 0
class_correct = list(0 for _ in range(num_classes))
class_total = list(0 for _ in range(num_classes))
conf_matrix = np.zeros([num_classes, num_classes])  # (i, j): i-Gt; j-Pr
error_num = 0
error_file = open(opt.net + '_error_samples.txt', 'w')
time1 = time.clock()
for txt_file in os.listdir(mid_dir):
    txt_file = os.path.join(mid_dir, txt_file)
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            img_name0, label0 = line.strip().split(' ')
            img_name = os.path.join(root_dir, img_name0)
            label = int(label0) - 1
            predict = demo.predict_me(img_name, net, device)
            if predict != label:
                error_file.writelines(img_name0 + ' is predicted to ' + classes[predict] + '\n')
                error_num += 1
            else:
                class_correct[label] += 1
            total += 1
            class_total[label] += 1
            conf_matrix[label, predict] += 1
time2 = time.clock()
error_file.close()
print('Accuracy in testset is %.2f %%' % ((total - error_num) / total * 100))
for i in range(num_classes):
    print('Accuracy of %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
plotCM(list(classes), conf_matrix, "./params/Confusion_Matrix.jpeg")
print("Processing 1 image spend around %.1f ms." % ((time2 - time1) * 1000 / total))
