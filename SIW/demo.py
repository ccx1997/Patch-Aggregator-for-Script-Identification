# -*- coding: utf-8 -*-
# By Changxu Cheng, HUST

import torch
from torchvision import transforms
import cv2
import argparse
from model import vgg
import os
import time

def predict_me(img_name, net, device, show_p=False):
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
    if show_p:
        output = torch.nn.functional.softmax(output, dim=1)
    p_max, output = torch.max(output, 1)
    if show_p:
        return p_max.item(), output.item()
    else:
        return output.item()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default='gspa', help='Choose the network')
    parser.add_argument("--file", type=str, required=True, help='Choose the textline image')
    opt = parser.parse_args()

    classes = ('Arabic', 'Cambodian', 'Chinese', 'English', 'Greek', 'Hebrew', 'Japanese', 'Kannada',
               'Korean', 'Mongolian', 'Russian', 'Thai', 'Tibetan')
    num_classes = len(classes)
    img_name = opt.file
    # img_name = './test/greek_000050_1.jpg'
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    time1 = time.clock()
    if opt.net == "gspa":
        net = vgg.GSPA(num_classes, pretrain_use='./params/tmp04.pkl')
    elif opt.net == 'gs':
        net = vgg.CNN(num_classes, pretrain_use='./params/vgg04.pkl')
    else:
        print('Please specify a correct network name! ([gs or gspa supported])')
    net.to(device)
    net.eval()
    time2 = time.clock()
    p_max, output = predict_me(img_name, net, device, show_p=True)
    time3 = time.clock()
    print("I think the text in the image {} is: {} with confidence={}".format(img_name, classes[output], p_max))
    print('Time: {0:6.3f}s for net-load, {1:6.3f}s for prediction'.format(time2 - time1, time3 - time2))


