# By Changxu Cheng, HUST

import torch
from model import vgg
import os
import numpy as np
import argparse
from draw.drawConfmat import plotCM
import time
import utils
import warnings
warnings.filterwarnings('ignore')


def train_model(net, classes, device, loaders_val, file_saving='tmp.pkl'):

    print(net)
    num_classes = len(classes)
    bs = 32
    train_loaders = utils.loaders_default(istrainset=True, classes=classes, batchsize=bs)  # [loader1, loader2, ...]

    print('*** Successfully Got Trainset ***')

    param_list = []
    lr_list = [1e-2]

    # Statistics about the number of each classes to use weighted loss
    train_distribution = [3711, 47452, 2702, 4633, 5631, 3237, 1247]
    weight = list(4000 / np.array(train_distribution))
    weight[1] = 0.7
    weight[3] = 0.9
    weight[-1] = 1.8

    train_me = utils.TrainMe(net, device, train_loaders, loaders_val,
                             num_classes, param_list, lr_list, howto_optim="SGD")
    loss_name = 'CrossEntropyLoss'
    # loss_name = 'CompromiseLoss'
    train_me.training(120, file_saving, weight, lr_min=1e-5, loss_name=loss_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--idc', type=int, default="1", help="Specify a gpu to run")
    parser.add_argument('--f0', type=str, default="", help="Choose the pretrained model parameters to finetune or test")
    parser.add_argument('--f1', type=str, default="./params/gspa2.pkl", help="Set the name of file to save model parameters")
    parser.add_argument('--valid', action='store_true', help='To valid, not to train')
    parser.add_argument('--net', type=str, default='gs', help='choose the network')
    opt = parser.parse_args()
    print(opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.idc)
    time.sleep(4)
    classes = ("Arabic", "Latin", "Chinese", "Japanese", "Korean", "Bangla", "Symbols")
    num_classes = len(classes)
    device = torch.device("cuda: 0")
    # Prepare CNN
    loaders_val = utils.loaders_default(istrainset=False, classes=classes, batchsize=32)
    param_file = opt.f0
    if param_file == "" and opt.valid:
        param_file = opt.f1
    if opt.net == 'gs':
        net = vgg.VGG16(num_classes, pretrain_use=param_file)
    elif opt.net == 'gspa':
        net = vgg.GSPA(num_classes, pretrain_use=param_file)
    else:
        raise ModuleNotFoundError("Please specify the correct net using --net")

    if opt.valid:
        if not os.path.exists(param_file):
            raise FileNotFoundError("Please train the model first!")

        # Predict and calculate accuracy
        accr, class_correct, class_total, conf_matrix = utils.accuracy(loaders_val, net, num_classes, device, disp=True)
        print('Accuracy on the validation set: %.2f %%' % (100 * accr))
        for i in range(num_classes):
            print('Accuracy of %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        plotCM(list(classes), conf_matrix, "./params/Confusion_Matrix.jpeg")
    else:
        param_file = opt.f1
        train_model(net, classes, device, loaders_val, file_saving=param_file)
