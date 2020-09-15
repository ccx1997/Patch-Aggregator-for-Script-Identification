# -*- encoding: utf-8 -*-
"""Fully connected layers VS Global average pooling
Comment about map size is based on images with w > h"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import copy


def param_init(net, param_file):
    # use pretrained model to the new model initialization
    param_pre = torch.load(param_file)
    param_new = net.state_dict()
    for key in param_new.keys():
        if key in param_pre.keys() and param_new[key].size() == param_pre[key].size():
            param_new[key] = param_pre[key]
    return param_new


class VGG16(nn.Module):
    """Pretrained vgg16_bn"""
    def __init__(self, num_classes, pretrain_use=''):
        super(VGG16, self).__init__()
        imagenet_pretrain = False if pretrain_use else True
        self.features = tv.models.vgg16_bn(pretrained=imagenet_pretrain).features
        self.classifier = nn.Sequential(nn.Linear(512, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(512, num_classes))
        self.pretrain_use = pretrain_use
        self._initialize()

    def forward(self, x):
        x = self.features(x)
        b, c = x.size()[:2]
        # Global average pooling
        x = x.contiguous().view(b, c, -1)
        x = torch.mean(x, dim=2)  # [b, c]
        y = self.classifier(x)
        if self.training:
            return x, y
        else:
            return y

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d and nn.BatchNorm2d and nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                try:
                    m.bias.data.zero_()
                except:
                    pass
        if self.pretrain_use:
            net_pre_params = param_init(self, self.pretrain_use)
            print("\nUse pretrained net to start! --%s" % self.pretrain_use)
            self.load_state_dict(net_pre_params)


class GSPA(VGG16):
    def __init__(self, num_classes, pretrain_use=''):
        super(GSPA, self).__init__(num_classes)
        imagenet_pretrain = False if pretrain_use else True
        features = tv.models.vgg16_bn(pretrained=imagenet_pretrain).features
        self.features = features[:34]   # [512, 4, w/16], shared part
        # GS
        self.embedding = features[34:]   # [512, 2, w/32]
        # PA
        self.special = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )   # [512, 2, w']
        self.cls1 = nn.Sequential(
             nn.Conv2d(512, 128, kernel_size=1, bias=False),
             nn.BatchNorm2d(128),
             nn.ReLU(),
             nn.Conv2d(128, num_classes, kernel_size=1))
        self.cls2 = nn.Sequential(nn.Linear(num_classes, 32),
                                  nn.ReLU(),
                                  nn.Linear(32, num_classes)
                                  )
        self.fusion = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 1, kernel_size=2),
                                    nn.BatchNorm2d(1),
                                    nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Sigmoid(),
                                    )
        self.pretrain_use = pretrain_use
        self._initialize()

    def forward(self, x):
        x = self.features(x)
        # G
        x_g = self.embedding(x)
        b = x_g.size(0)
        out1 = self.classifier(F.adaptive_avg_pool2d(x_g, (1, 1)).contiguous().view(b, -1))   # [b, nc]
        # L
        x_l = self.special(x)
        y_l = self.cls1(x_l)    # [b, 7, 2, w]
        y_l0 = F.softmax(y_l, dim=1)
        y_l = F.adaptive_max_pool2d(y_l0, (1, 1)).contiguous().view(b, -1)   # [b, 7]
        out2 = self.cls2(y_l)
        # Dynamic weighting
        weight = self.fusion(x).contiguous().view(b, -1)    # [b, 1]
        weight = torch.cat((weight, 1-weight), dim=1)    # [b, 2]
        y_all = torch.stack((out1, out2), dim=2)
        out = y_all.matmul(weight.unsqueeze(2)).squeeze(2)

        if self.training:
            nc = out2.size(1)
            ally = y_l0.permute(0, 2, 3, 1).contiguous().view(-1, nc)  # [b*w*2, 7], it is p.
            return ally, out1, out2, out
        else:
            return out


if __name__ == "__main__":
    im = torch.rand(2, 3, 64, 128)
    net = GSPA(7)
    net.eval()
    print(net)
    out = net(im)
    print(out)
    print(out.size())
