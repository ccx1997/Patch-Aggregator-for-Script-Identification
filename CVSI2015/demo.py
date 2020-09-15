# -*- coding: utf-8 -*-
# By Changxu Cheng, HUST

import torch
from torchvision import transforms
import cv2
import argparse
from model import vgg
import os
import time


def predict_me(img_name, net, device):
    """Predict the label of an given image by net"""
    image = cv2.imread(img_name, 1)
    h, w = image.shape[:2]
    image = cv2.resize(image, (int(32*w/h), 32))
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = tfm(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = net(image)
    _, output = torch.max(output, 1)
    return output.item()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help='Choose the textline image')
    opt = parser.parse_args()

    classes = ('Arabic', 'Bengali', 'English', 'Gujrathi', 'Hindi', 'Kannada', 'Oriya', 'Punjabi', 'Tamil', 'Telegu')
    num_classes = len(classes)
    # img_name = opt.file
    img_name = './test/russian_000427_2.jpg'
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    time1 = time.clock()
    net = vgg.CNN_Att4(num_classes, pretrain_use='./params/siw_param/vgg427.pkl')
    net.to(device)
    net.eval()
    time2 = time.clock()
    output = predict_me(img_name, net, device)
    time3 = time.clock()
    print("I think the text in the image {} is: {}".format(img_name, classes[output]))
    print('Time: {0:6.3f}s for net-load, {1:6.3f}s for prediction'.format(time2 - time1, time3 - time2))


