# -*- encoding: utf-8 -*-
"""Networks based on vgg"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from skimage import io, img_as_float
import numpy as np
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
    """ The backbone with Global Squeezer """
    def __init__(self, num_classes, pretrain_use=''):
        super(CNN, self).__init__()
        ks = [3, 3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1, 1]
        ss = [1, 1, 1, 1, 1, 1]
        nm = [64, 64, 128, 128, 256, 256]  # Number of feature maps

        cnn = nn.Sequential()

        def convRelu(i, relu=True):
            nIn = 3 if i == 0 else nm[i - 1]
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
                                nn.Dropout(0.3),
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
        if self.training:
            return out1, out
        else:
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


class CNN_ensemble(CNN):
    """To show the proposed local checker make sense not attributed to ensemble"""
    def __init__(self, num_classes, pretrain_use=''):
        super(CNN_ensemble, self).__init__(num_classes)
        self.embedding2 = copy.deepcopy(self.embedding)
        self.fc2 = copy.deepcopy(self.fc)
        self.pretrain_use = pretrain_use
        self._initialize()

    def forward(self, x):
        x = self.cnn(x)     # [b,256,4,w]
        b = x.size(0)
        # 1
        x1 = self.embedding(x)   # [b, c, 1, w], c=512
        x1 = F.adaptive_avg_pool2d(x1, (1, 1)).contiguous().view(b, -1)     # GAP
        out1 = self.fc(x1)
        # 2
        x2 = self.embedding2(x)   # [b, c, 1, w], c=512
        x2 = F.adaptive_avg_pool2d(x2, (1, 1)).contiguous().view(b, -1)     # GAP
        out2 = self.fc(x2)
        
        if self.training:
            return out1, out2
        else:
            out = 0.5 * out1 + 0.5 * out2
            return out


class CNN_AM(CNN):
    """Combine GAP and GMP"""
    def __init__(self, num_classes, pretrain_use=''):
        super(CNN_AM, self).__init__(num_classes)
        self.special = copy.deepcopy(self.embedding)
        self.fc2 = copy.deepcopy(self.fc)
        self.fusion = nn.Sequential(nn.Linear(num_classes, 1),
                                    nn.Sigmoid(),
                                    )
        self.pretrain_use = pretrain_use
        self._initialize()

    def forward(self, x):
        x = self.cnn(x)  # [b,256,4,w]
        # GAP
        x1 = self.embedding(x)  # [b, c, 1, w], c=512
        b, c = x1.size()[:2]
        x1 = x1.contiguous().view(b, c, -1)
        x1 = torch.mean(x1, dim=2)
        out1 = self.fc(x1)
        # GMP
        x2 = self.special(x)  # [b, c, 1, w], c=512
        x2 = x2.contiguous().view(b, c, -1)
        x2, _ = torch.max(x2, dim=2)
        out2 = self.fc2(x2)
        # Dynamic weighting
        weight = self.fusion(F.relu(out1))  # [b, 1]
        weight = torch.cat((weight, 1 - weight), dim=1)  # [b, 2]
        y_all = torch.stack((out1, out2), dim=2)
        out = y_all.matmul(weight.unsqueeze(2)).squeeze(2)
        if self.training:
            return out1, out2, out
        else:
            return out


class GSPA(CNN):
    def __init__(self, num_classes, pretrain_use=''):
        super(GSPA, self).__init__(num_classes)
        self.special = copy.deepcopy(self.embedding)
        self.cls1 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes+1, kernel_size=1),
        )   # Note!!!
        # (num_classes+1) does nothing, but we keep it due to our late discovery while not willing to change
        # Original intention is to represent the background, but we can't realize it by now!
        self.cls2 = nn.Sequential(nn.Linear(num_classes+1, 32),
                                  nn.ReLU(),
                                  nn.Linear(32, num_classes)
                                  )
        self.fusion = nn.Sequential(nn.Linear(num_classes, 1),
                                    nn.Sigmoid(),
                                    )
        self.pretrain_use = pretrain_use
        self._initialize()

    def forward(self, x, visual=False):
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

        # Visualization
        if visual:
            nc1 = y_l0.size(1)
            out2 = F.softmax(out2, dim=1)
            return y_l0.contiguous().view(nc1, -1), y_l.contiguous().view(nc1, -1), out2.contiguous().view(nc1-1, -1)

        # Dynamic weighting
        weight = self.fusion(F.relu(out1))  # [b, 1]
        weight = torch.cat((weight, 1 - weight), dim=1)  # [b, 2]
        y_all = torch.stack((out1, out2), dim=2)
        out = y_all.matmul(weight.unsqueeze(2)).squeeze(2)
        if self.training:
            ally = y_l0.permute(0, 2, 3, 1).contiguous().view(b*w, -1)  # [b*w, 14], it is p.
            return ally, out1, out2, out
        else:
            return out


class PA(CNN):
    # Only have local representation branch
    def __init__(self, num_classes, pretrain_use=''):
        super(PA, self).__init__(num_classes)
        self.cls1 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes+1, kernel_size=1),
        )
        self.fc = nn.Sequential(nn.Linear(num_classes+1, 32),
                                nn.ReLU(),
                                nn.Linear(32, num_classes)
                                )
        self.pretrain_use = pretrain_use
        self._initialize()

    def forward(self, x):
        x = self.cnn(x)
        # L
        x_l = self.embedding(x)
        b, c, h, w = x_l.size()
        y_l = self.cls1(x_l)    # [b, 14, 1, w]
        y_l0 = F.softmax(y_l, dim=1)
        y_l = F.adaptive_max_pool2d(y_l0, (1, 1)).contiguous().view(b, -1)   # [b, 14]
        out2 = self.fc(y_l)
        if self.training:
            ally = y_l0.permute(0, 2, 3, 1).contiguous().view(b*w, -1)  # [b*w, 14]
            return ally, out2
        else:
            return out2


if __name__ == "__main__":
    im = np.random.rand(32, 128, 3)
    nc = im.shape[2]
    net = GSPA(13)
    print(net)
    im = torch.FloatTensor(im.transpose(2, 0, 1)).unsqueeze(0)
    out = net(im)
    print(out)
    print(out.size())
