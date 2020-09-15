# -*- coding: utf:8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from dataset.mydata import getloaders
import random
import matplotlib.pyplot as plt
import os
import visdom
import math


def loaders_default(istrainset, batchsize=16):
    # Get loaders
    root_dir = '/workspace/datasets/script_id/CVSI2015/'
    tmp = "TrainDataset_CVSI2015" if istrainset else "ValidationDataset_CVSI2015"
    root_dir = os.path.join(root_dir, tmp)
    mid_dir = "z_grp_ccx"
    widths = [32, 64, 128]
    txts = ["grp32.txt", "grp64.txt", "grp128.txt"]
    transf = transforms.Normalize((0.5,), (0.5,))
    return getloaders(root_dir, mid_dir, widths, txts, transf, shuffle=istrainset, bs=batchsize, aug=istrainset)


def softermaxloss(x, k, labels=None, eps=None, is_p=False):
    """x: [b, c]  k: top-k"""
    if not is_p:
        x = F.softmax(x, dim=1)

    if labels is None:
        x_top, _ = torch.topk(x, k, dim=1)
    else:
        x_top0, idtop = torch.topk(x, k, dim=1)
        b, c = x.size()
        lbd = 1 - eps * torch.sum(x_top0, dim=1) / x[torch.arange(b), labels]
        lbd.detach_()
        tmp = torch.ones(b, c).cuda()
        tmp[torch.arange(b), labels] = lbd
        x = x * tmp
        x_top = x.gather(1, idtop)

    x_sum = torch.sum(x_top, dim=1)
    losses = -torch.log(x_sum)
    return losses.mean()


def topkloss(x, k, labels, eps=0.1):
    """L = -eps * log(py) - (1 - eps)log(lbd * py + pj + pk + ...)
    x: Probability distributions of a batch with size [b, c];
    k: top-k, k can be int(num_classes * 0.2);       eps: the weight of CELoss
    """
    # jky = k - 1     # jky: (pj+pk+...)/py, by default we have a hypothesis that Epj=Epk=Epy
    # lbd = 1 - eps * (1 + jky)   # To ensure pj,pk,py changes the same scalar for every update(grad)
    func = nn.NLLLoss().cuda()
    loss = eps * func(x.log(), labels) + (1 - eps) * softermaxloss(x, k, labels, eps, is_p=True)
    return loss


class CompromiseLoss(nn.Module):
    """The content of softness is between softmax and direct weight"""
    def __init__(self, k=3, size_average=True):
        super(CompromiseLoss, self).__init__()
        self.k = torch.tensor(k, dtype=torch.float32, requires_grad=False).cuda()
        self.size_average = size_average

    def slower_exp(self, x):
        # x is a tensor, y = Linear(x) if x > k else exp(x)
        pos = (x >= self.k.item())
        neg = (x < self.k.item())
        pos, neg = pos.to(torch.float32), neg.to(torch.float32)
        return (torch.exp(self.k) * (x - self.k + 1)) * pos + torch.exp(x) * neg

    def forward(self, x, target):
        # x: [b, c]
        x = self.slower_exp(x)
        x = x / torch.sum(x, 1, keepdim=True)
        x = torch.log(x)
        reduction = 'elementwise_mean' if self.size_average else 'sum'
        return F.nll_loss(x, target, reduction=reduction)


def accuracy(dataloaders, net, num_classes, device, disp=False):

    net = net.to(device)
    net.eval()
    correct = 0
    total = 0
    class_correct = list(0 for _ in range(num_classes))
    class_total = list(0 for _ in range(num_classes))
    conf_matrix = np.zeros([num_classes, num_classes])  # (i, j): i-Gt; j-Pr
    for loader in iter(dataloaders):
        for data in iter(loader):
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = net(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                conf_matrix[label, predicted[i].item()] += 1
    accr = correct / total
    if disp:
        print('Total number of images={}'.format(total))
        print('Total number of correct images={}'.format(correct))

    return accr, class_correct, class_total, conf_matrix


class TrainMe(object):
    """A basic training strategy that can be used in many circumstances"""
    def __init__(self, net, device, dataloaders, loader_val, num_classes,
                 param_list, lr_list, howto_optim="Adam", weight_decay=1e-4):
        assert howto_optim.lower() in {"adam", "sgd"}, "Only support SGD and Adam!"
        if len(param_list) > 0:
            assert len(param_list) == len(lr_list), "len(param_list) should be equal to len(lr_list)!"
        self.net = net
        self.device = device
        self.dataloaders = dataloaders
        self.loader_val = loader_val
        self.nc = num_classes
        self.param_list = param_list
        self.lr_list = lr_list
        self.howto_optim = howto_optim
        self.wd = weight_decay

    def get_optimizer(self):

        n_seg = len(self.param_list)
        if n_seg > 0:
            L_in = [{"params": getattr(self.net, self.param_list[i]).parameters(),
                     "lr": self.lr_list[i]} for i in range(n_seg)]
        else:
            L_in = [{"params": self.net.parameters(), "lr": self.lr_list[0]}]
        if self.howto_optim.lower() == "adam":
            return torch.optim.Adam(L_in, weight_decay=self.wd)
        else:
            return torch.optim.SGD(L_in, weight_decay=self.wd, momentum=0.9)

    def calculate_loss(self, data, i, vis_im, f_loss):
        images, labels = data
        # Show images in visdom
        if i % 20 == 19:
            vis_im.images(images / 2 + 0.5, win='pic')
        images, labels = images.to(self.device), labels.to(self.device)
        # adaptive to the output of the net
        y = self.net(images)
        loss_ce = f_loss(y, labels)
        return loss_ce

    def calculate_multiloss(self, data, i, vis_im, f_loss, weight):
        # In case that multiple result needed penalized is returned from net
        images, labels = data
        # Show images in visdom
        if i % 20 == 19:
            vis_im.images(images / 2 + 0.5, win='pic')
        images, labels = images.to(self.device), labels.to(self.device)
        ys = self.net(images)
        loss_ce = 0.0
        f_loss0 = nn.NLLLoss().to(self.device)
        for j, y in enumerate(ys):
            if j == 0:
                width = y.size(0) // images.size(0)
                labels_all = labels.repeat(width, 1).t().contiguous().view(1, -1).squeeze(0)
                loss_ce = loss_ce + topkloss(y, 3, labels_all, eps=0.1) * weight[j]
                # loss_ce = loss_ce + (1.0 * softermaxloss(y, 3, is_p=True) + 0.1 * f_loss0(y.log(), labels_all)) * weight[j]
            else:
                loss_ce = loss_ce + weight[j] * f_loss(y, labels)
            # loss_ce = loss_ce + f_loss(y, labels) * weight[j]
        return loss_ce

    def training(self, num_epoch, model_param, loss_name="CrossEntropyLoss", lr_min=1e-5):
        # Prepare for visdom
        vis_im = visdom.Visdom(env='CVSI', port=8098)
        vis_l = visdom.Visdom(env='CVSI', port=8098)
        legend_names = ['GAP', 'Inference']
        legend_name = legend_names[1]
        assert vis_l.check_connection()

        self.net = self.net.to(self.device)
        f_loss = getattr(torch.nn, loss_name)(reduction='elementwise_mean').to(self.device)
        optimizer = self.get_optimizer()
        i_list = []
        avg_loss_list = []
        accr_list = []
        accr_best = 0.0
        dc = 25
        usebest = 1
        print("***Start training!***")
        for epoch in range(num_epoch):
            running_loss = 0.0
            random.shuffle(self.dataloaders)
            i0 = 0
            for loader in iter(self.dataloaders):
                for i2, data in enumerate(loader):
                    self.net.train()
                    i = i0 + i2
                    loss = self.calculate_multiloss(data, i, vis_im, f_loss, [0.1, 0.15, 0.15, 0.8])
                    # loss = self.calculate_loss(data, i, vis_im, f_loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                i0 = i + 1
            avg_loss = running_loss / i
            test_net = self.net
            accr_cur, _, _, _ = accuracy(self.loader_val, test_net, self.nc, self.device)
            accr_list.append(accr_cur)
            if accr_best < accr_cur:
                net_best = self.net
                accr_best = accr_cur
                loss_best = avg_loss
                # Save the trained Net
                print("saving...", end=', ')
                torch.save(net_best.state_dict(), model_param)

            print('[%2d,%5d] lr=%.5f loss: %.4f accr(val): %.3f %%' %
                  (epoch + 1, i + 1, self.lr_list[0], avg_loss, accr_cur * 100))
            i_list.append(epoch)
            avg_loss_list.append(avg_loss)
            vis_l.line(np.array([avg_loss]), np.array([epoch]), update='append',
                       win='losses', name=legend_name, opts=dict(showlegend=True))
            vis_l.line(np.array([accr_cur]), np.array([epoch]), update='append',
                       win='accr', name=legend_name, opts=dict(showlegend=True))
            vis_l.line(np.array([-math.log(self.lr_list[0], 10)]), np.array([epoch]), update='append',
                       win='lr_nl', name=legend_name, opts=dict(showlegend=True))
            # Update lr according to the process of training
            if len(avg_loss_list) > 1 and ((1 - avg_loss_list[-1] / (avg_loss_list[-2] + 1e-10) < 0.005) or
                                           (avg_loss < 0.005 and avg_loss_list[-2] - avg_loss_list[-1] < 0.0003)):
                dc -= 1
                if avg_loss > 0.1 and avg_loss_list[-1] / (avg_loss_list[-2] + 1e-10) - 1 > 0.6:
                    dc += 1
                if dc == 0:
                    self.lr_list = [0.3 * i for i in self.lr_list]
                    dc = 16 if self.lr_list[0] > 1e-3 else 10
                    if max(self.lr_list) < lr_min:
                        self.lr_list = [1e-2 for i in self.lr_list]
                        if epoch > 350 and usebest > 0:
                            print("\nNow we use the best model to train as initialization")
                            self.net.load_state_dict(torch.load(model_param))
                            usebest -= 1

                    optimizer = self.get_optimizer()
        # plot loss-iter figure
        print("best_loss=%.3f, best_accr=%.3f%%, model params %s saved " % (loss_best, accr_best * 100, model_param))
        # plot loss-i
        plt.switch_backend('agg')
        plt.plot(i_list, avg_loss_list)
        plt.ylabel('loss')
        plt.savefig('loss.png')
        print('*** Finished Training! ***')

