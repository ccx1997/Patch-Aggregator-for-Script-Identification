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

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def loaders_default(istrainset, batchsize=16, pad=False, fusion=False, multilabel=False, triplet=False):
    # Get loaders
    root_dir = '/workspace/datasets/script_id/SIW-13/'
    sf, tmp, aug = (True, "trainset", True) if istrainset else (False, "testset", False)
    mid_dir = os.path.join("z_grp_ccx/", tmp)
    widths = [64, 128, 256, 512]
    txts = ["grp64.txt", "grp128.txt", "grp256.txt", "grp512.txt"]
    transf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return getloaders(root_dir, mid_dir, widths, txts, transf, shuffle=sf, bs=batchsize,
                      aug=aug, pad=pad, fusion=fusion, multilabel=multilabel, triplet=triplet)


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
    def __init__(self, k=5, size_average=True):
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


class FocalCrossEntropyLoss(nn.Module):
    r"""-log softmax[x, target] using focal idea"""
    def __init__(self, weight):
        super(FocalCrossEntropyLoss, self).__init__()
        self.weight = weight.cuda() if torch.cuda.is_available() else weight

    def forward(self, x, target):
        ps = F.softmax(x, 1)
        a = self.weight.gather(0, target)
        target = target.unsqueeze(1)
        p = ps.gather(1, target).squeeze(1)
        weight = F.sigmoid(1.5*(p.mean()-p)) * 2
        return -a.dot(weight * torch.log(p)) / a.sum()


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

    def calculate_loss(self, data, i, vis_im, f_loss, **kwargs):
        multilabel = kwargs['multilabel']
        f_losses = kwargs['f_losses']
        images, labels = data
        # Show images in visdom
        if i % 20 == 19:
            vis_im.images(images / 2 + 0.5, win='pic')
        images, labels = images.to(self.device), labels.to(self.device)
        # adaptive to the output of the net
        y = self.net(images)
        # # function to hook gradient at the GAP result
        # def extract(g):
        #     global g0
        #     g0 = g
        # y[0][1].register_hook(extract)
        b = y[1].size(0)
        # loss = softermax(y, k, labels)
        if multilabel:
            loss_ce = f_loss(y[1], labels[:, -1])
            for j in range(3):
                loss_ce = loss_ce + 0.8 * f_losses[j](y[0][j], labels[:, j])
            loss_ce = loss_ce / 4
        else:
            loss_ce = f_loss(y[1], labels)
            # loss_ce.backward(retain_graph=True)
            # loss_w = torch.pow(torch.mean((1 - F.cosine_similarity(y1[0],
            #                    F.sigmoid(g0/torch.std(g0, dim=1, keepdim=True)*0.5), dim=1))) * 0.5, 2)
            # loss_w.backward()
            if y[0].size(0) != b:
                y1_max, _ = y[0].max(dim=1)
                loss_local = -torch.mean(y1_max.log())
                loss_ce = loss_ce + 0.1 * loss_local
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
        # f_loss1 = CompromiseLoss(k=5)
        # loss_ce = f_loss(ys, labels)

        for j, y in enumerate(ys):
            if j == 0:
                width = y.size(0) // images.size(0)
                labels_all = labels.repeat(width, 1).t().contiguous().view(1, -1).squeeze(0)
                # loss_ce = loss_ce + topkloss(y, 3, labels_all, eps=0.1) * weight[j]
                loss_ce = loss_ce + f_loss0(y.log(), labels_all) * weight[j]
                # loss_ce = loss_ce + (0.4 * softermaxloss(y, 3, is_p=True) + 0.6 * f_loss0(y.log(), labels_all)) * weight[j]
            else:
                loss_ce = loss_ce + weight[j] * f_loss(y, labels)

            # loss_ce = loss_ce + f_loss(y, labels) * weight[j]
        return loss_ce

    def calculate_tripletloss(self, data, i, vis_im, f_loss):
        # data: (images, labels, images_pos, labels_pos, images_neg, labels_neg)
        # Show images in visdom
        if i % 20 == 19:
            vis_im.images(data[0] / 2 + 0.5, win='pic')
        data = [item.to(self.device) for item in data]

        # output of the net
        Y = [self.net(data[2*t]) for t in range(3)]  # [(y1, y), (), ()]
        # CrossEntropy Loss
        loss_ces = [f_loss(Y[t][1], data[2*t+1]) for t in range(3)]
        loss_ce = (loss_ces[0] + loss_ces[1] + loss_ces[2]) / 3
        # TripletLoss
        sim_pos = F.cosine_similarity(Y[0][0], Y[1][0], dim=1)
        sim_neg = F.cosine_similarity(Y[0][0], Y[2][0], dim=1)
        loss_tri = torch.mean(torch.log(1 + torch.exp(1.4 * (sim_neg - sim_pos))))
        # Total loss
        loss = loss_ce + loss_tri * 0.5
        return loss

    def training(self, num_epoch, model_param, weight, loss_name="CrossEntropyLoss",
                 lr_min=1e-5, multilabel=False, triplet=False):
        # Prepare for visdom
        vis_im = visdom.Visdom(env='vggccx', port=8098)
        vis_l = visdom.Visdom(env='vggccx', port=8098)
        legend_names = ['AM', 'constvarL', 'Ensemble', 'GAP', 'constL', 'WithoutL3', 'GSPA2']
        if multilabel:
            legend_name = legend_names[1]
        elif triplet:
            legend_name = legend_names[2]
        else:
            legend_name = legend_names[-1]
        assert vis_l.check_connection()

        self.net = self.net.to(self.device)
        f_losses = []
        if multilabel:
            for i in range(3):
                f_losses.append(torch.nn.CrossEntropyLoss(weight=torch.tensor(weight[i]),
                                                          reduction='elementwise_mean').to(self.device))
        try:
            # f_loss = getattr(torch.nn, loss_name)(weight=torch.tensor(weight[-1]),
            #                                       reduction='elementwise_mean').to(self.device)
            f_loss = getattr(torch.nn, loss_name)(reduction='elementwise_mean').to(self.device)
        except:
            f_loss = eval(loss_name)().to(self.device)
        optimizer = self.get_optimizer()
        i_list = []
        avg_loss_list = []
        accr_list = []
        accr_best = 0.0
        dc = 20
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
                    if triplet:
                        loss = self.calculate_tripletloss(data, i, vis_im, f_loss)
                    else:
                        # loss = self.calculate_multiloss(data, i, vis_im, f_loss, [0.1, 0.1, 1.0])
                        # loss = self.calculate_multiloss(data, i, vis_im, f_loss, [0.2, 1.0])
                        # loss = self.calculate_multiloss(data, i, vis_im, f_loss, [0.1, 1.0, 1.0])
                        # loss = self.calculate_multiloss(data, i, vis_im, f_loss, [0.25, 0.25, 0.1, 0.4])
                        # loss = self.calculate_multiloss(data, i, vis_im, f_loss, [0.3, 0.15, 0.55])
                        # loss = self.calculate_multiloss(data, i, vis_im, f_loss, [1.0])
                        loss = self.calculate_multiloss(data, i, vis_im, f_loss, [0.1, 0.1, 0.1, 0.8])
                        # loss = self.calculate_multiloss(data, i, vis_im, f_loss, [0.1, 0.2, 0.2, 0.2, 0.4])
                        # loss = self.calculate_loss(data, i, vis_im, f_loss, multilabel=multilabel, f_losses=f_losses)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                i0 = i + 1
            avg_loss = running_loss / i
            # running_loss = 0.0
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

            print('[%2d,%4d] lr=%.5f loss: %.4f accr(val): %.3f %%' %
                  (epoch + 1, i + 1, self.lr_list[0], avg_loss, accr_cur * 100))
            # i_list.append(i + L_loaders * epoch)
            i_list.append(epoch)
            avg_loss_list.append(avg_loss)
            # i_all = i + L_loaders * epoch
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
                    dc = 20 if self.lr_list[0] > 1e-3 else 12
                    if max(self.lr_list) < lr_min:
                        self.lr_list = [1e-2 for i in self.lr_list]
                        if epoch > 350 and usebest > 0:
                            print("\nNow we use the best model to train as initialization")
                            self.net.load_state_dict(torch.load(model_param))
                            usebest -= 1

                    optimizer = self.get_optimizer()
                # i0 = i + 1
        # plot loss-iter figure
        print("best_loss=%.3f, best_accr=%.3f%%, model params %s saved " % (loss_best, accr_best * 100, model_param))
        # plot loss-i
        plt.switch_backend('agg')
        plt.plot(i_list, avg_loss_list)
        plt.ylabel('loss')
        plt.savefig('loss.png')
        print('*** Finished Training! ***')


if __name__ == "__main__":
    y = torch.rand(3, 4).cuda()
    target = torch.tensor([1, 3, 2]).cuda()
    f = CompromiseLoss().cuda()
    loss = f(y, target)

