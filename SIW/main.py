# -*- coding: utf-8 -*-
# By Changxu Cheng, HUST

import torch
from model import vgg
import os
import numpy as np
import argparse
from draw.drawConfmat import plotCM
import time
import utils


def train_model(net, num_classes, device, loaders_val, file_saving='tmp.pkl', fusion=False,
                multlabel=False, triplet=False):

    print(net)
    train_loaders = utils.loaders_default(istrainset=True, batchsize=16, pad=False, fusion=fusion,
                                          multilabel=multlabel, triplet=triplet)  # [loader1, loader2, ...]
    print('*** Successfully Got Trainset ***')

    param_list = []
    lr_list = [1e-1]

    # Statistics about the number of each classes to use weighted loss
    train_distribution = [502, 583, 798, 721, 518, 742, 715, 529, 1061, 692, 531, 1722, 677]
    tr0 = [502, 583, 1513, 1770, 742, 529, 1061, 692, 1722, 677]
    tr1 = [8021, 721, 518, 531]
    tr2 = [8278, 798, 715]
    tr = (tr0, tr1, tr2)
    weight = list(800 / np.array(train_distribution))
    weight[-2] = 0.6
    weights = []
    for i in range(3):
        weights.append(list(800 / np.array(tr[i])))
    weights.append(weight)

    train_me = utils.TrainMe(net, device, train_loaders, loaders_val,
                             num_classes, param_list, lr_list, howto_optim="SGD")
    loss_name = 'CrossEntropyLoss'
    train_me.training(500, file_saving, weight=weights, lr_min=1e-5, loss_name=loss_name,
                      multilabel=multlabel, triplet=triplet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--idc', type=int, default=1, help="Specify a gpu to run")
    parser.add_argument('--f0', type=str, default="", help="Choose the pretrained model parameters to finetune or test")
    parser.add_argument('--f1', type=str, default="tmp.pkl", help="Set the name of file to save model parameters")
    parser.add_argument('--test', action='store_true', help='To test, not train')
    parser.add_argument('--net', type=str, default='vgg', help='choose the network')
    parser.add_argument('--noise', action='store_true', help='Make specified noise to labels')
    parser.add_argument('--fuse', action='store_true', help='Make label fusion')
    parser.add_argument('--multilabel', action='store_true', help='Use multiLabel')
    parser.add_argument('--triplet', action='store_true', help='Use triplet loss in metric learning')
    opt = parser.parse_args()
    print(opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.idc)
    time.sleep(4)
    classes = ('Arabic', 'Cambodian', 'Chinese', 'English', 'Greek', 'Hebrew', 'Japanese', 'Kannada',
               'Korean', 'Mongolian', 'Russian', 'Thai', 'Tibetan')
    num_classes = len(classes)
    device = torch.device("cuda: 0")
    # Prepare CNN
    loaders_val = utils.loaders_default(istrainset=False, batchsize=128, pad=False, fusion=opt.fuse)
    param_file = opt.f0
    if param_file == "" and opt.test:
        param_file = opt.f1
    if opt.net == 'vgg':
        net = vgg.CNN(num_classes, pretrain_use=param_file)
    elif opt.net == 'gspa':
        net = vgg.GSPA(num_classes, pretrain_use=param_file)
    else:
        raise ModuleNotFoundError("Please specify the correct net using --net")

    if opt.test:
        if not os.path.exists(param_file):
            raise FileNotFoundError("Please train the model first!")
        accr, class_correct, class_total, conf_matrix = utils.accuracy(loaders_val, net, num_classes, device, disp=True)
        print('Accuracy on the testset: %.2f %%' % (100 * accr))
        for i in range(num_classes):
            print('Accuracy of %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        plotCM(list(classes), conf_matrix, "./params/siw_param/Confusion_Matrix_cnn.jpeg")
    else:
        param_file = opt.f1
        train_model(net, num_classes, device, loaders_val, file_saving=param_file,
                    fusion=opt.fuse, multlabel=opt.multilabel, triplet=opt.triplet)
