# -*- coding: utf-8 -*-
# By Changxu Cheng, HUST

import torch
import os
import demo
from model import vgg
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="./params/siw_param/vgg001.pkl",
                    help="Choose the pretrained model parameters to drive")
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
root_dir = '/workspace/datasets/script_id/CVSI2015/'
mid_dir = os.path.join(root_dir, "z_grp_ccx")
classes = ('Arabic', 'Cambodian', 'Chinese', 'English', 'Greek', 'Hebrew', 'Japanese', 'Kannada',
           'Korean', 'Mongolian', 'Russian', 'Thai', 'Tibetan')
num_classes = len(classes)
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

net = vgg.CNN_Att4(num_classes, pretrain_use=opt.model)
# net = vgg.CNN_Att2(num_classes, pretrain_use=opt.model)
net.to(device)
net.eval()
error_num, total = 0, 0
error_file = open('error_samples.txt', 'w')
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
            total += 1
error_file.close()
print('Accuracy in testset is %.2f %%' % ((total - error_num) / total * 100))


