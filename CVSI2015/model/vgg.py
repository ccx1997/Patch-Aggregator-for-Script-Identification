# -*- encoding: utf-8 -*-
"""Networks based on vgg"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def param_init(net, param_file):
    # use pretrained model to the new model initialization
    param_pre = torch.load(param_file)
    param_new = net.state_dict()
    for key in param_new.keys():
        if key in param_pre.keys() and param_new[key].size() == param_pre[key].size():
            param_new[key] = param_pre[key]
    return param_new


class CNN(nn.Module):
    def __init__(self, num_classes, pretrain_use=''):
        super(CNN, self).__init__()
        ks = [3, 3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1, 1]
        ss = [1, 1, 1, 1, 1, 1]
        nm = [64, 64, 128, 128, 256, 256]  # Number of feature maps

        cnn = nn.Sequential()

        def convRelu(i, relu=True):
            nIn = 1 if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i], bias=False))
            cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if relu:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0, True)
        convRelu(1, True)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x(Width/2)
        convRelu(2, True)
        convRelu(3, True)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x(Width/4)
        convRelu(4, True)
        convRelu(5, True)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))  # 256x4x(Width/8)

        self.cnn = cnn
        self.embedding = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=(0, 1), bias=False),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU(),
                                       nn.Conv2d(512, 512, kernel_size=(2, 3), stride=(1, 2),
                                                 padding=(0, 1), bias=False),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU()
                                       )    # 512x1x(Width/16)
        self.fc = nn.Sequential(nn.Linear(512, 512),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(512, num_classes))
        self.pretrain_use = pretrain_use
        self._initialize()

    def forward(self, x):
        x = self.cnn(x)     # [b,256,4,w]
        x = self.embedding(x)   # [b, c, 1, w], c=512
        b, c = x.size()[:2]
        x = x.contiguous().view(b, c, -1)
        out1 = torch.mean(x, dim=2)     # GAP
        out = self.fc(out1)
        return out

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


class CNN_Att4(CNN):
    def __init__(self, num_classes, pretrain_use=''):
        super(CNN_Att4, self).__init__(num_classes)
        self.special = copy.deepcopy(self.embedding)
        self.cls1 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes+1, kernel_size=1),
        )
        self.cls2 = nn.Sequential(nn.Linear(num_classes+1, 32),
                                  nn.ReLU(),
                                  nn.Linear(32, num_classes)
                                  )
        self.fusion = nn.Sequential(nn.Linear(num_classes, 1),
                                    nn.Sigmoid(),
                                    )
        self.pretrain_use = pretrain_use
        self._initialize()

    def forward(self, x):
        x = self.cnn(x)
        # G
        x_g = self.embedding(x)
        b, c, h, w = x_g.size()
        x_g = torch.mean(x_g.contiguous().view(b, c, -1), dim=2)
        out1 = self.fc(x_g)   # [b, nc]
        # L
        x_l = self.special(x)
        y_l = self.cls1(x_l)    # [b, 14, 1, w]
        y_l0 = F.softmax(y_l, dim=1)
        y_l = F.adaptive_max_pool2d(y_l0, (1, 1)).contiguous().view(b, -1)   # [b, 14]
        out2 = self.cls2(y_l)
        # Dynamic weighting
        weight = self.fusion(F.relu(out1))  # [b, 1]
        weight = torch.cat((weight, 1 - weight), dim=1)    # [b, 2]
        y_all = torch.stack((out1, out2), dim=2)
        out = y_all.matmul(weight.unsqueeze(2)).squeeze(2)

        if self.training:
            ally = y_l0.permute(0, 2, 3, 1).contiguous().view(b*w, -1)  # [b*w, 14], it is p.
            return ally, out1, out2, out
        else:
            return out


if __name__ == "__main__":
    net = CNN_Att4(10)
    print(net)
    im = torch.rand(1, 3, 32, 64)
    out = net(im)
    print(out)
    print(out.size())
