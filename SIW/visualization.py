"""Visualization about the process of Patch Aggregator (PA) branch """
import torch
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
from model import vgg
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default='test/greek_000050_1.jpg', help='Choose the textline image')
parser.add_argument("--param", type=str, default='./params/tmp04.pkl')
opt = parser.parse_args()

# Device settings
#
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

# Prepare input data
classes = ['Arabic', 'Cambodian', 'Chinese', 'English', 'Greek', 'Hebrew', 'Japanese', 'Kannada',
           'Korean', 'Mongolian', 'Russian', 'Thai', 'Tibetan']
# classes = ['阿拉伯语', '柬埔寨语', '汉语', '英语', '希腊语', '希伯来语', '日语', '坎纳达语',
#            '韩语', '蒙语', '俄语', '泰语', '藏语']
num_classes = len(classes)
#
# img_name = './test/greek_000350_1.jpg'
img_name = opt.file
print("The image name is " + img_name)
#
image = cv2.imread(img_name, 1)
h, w = image.shape[:2]
image = cv2.resize(image, (int(32*w/h), 32))
tfm = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
image = tfm(image)
image = image.unsqueeze(0).to(device)

# The network to use
net = vgg.GSPA(num_classes, pretrain_use=opt.param)
net.to(device)
net.eval()

# Output the middle result
with torch.no_grad():
    pp, mp, op = net(image, visual=True)

# Predict
confidence, pred = torch.max(op, dim=0)
print("The text in the image is likely to be {} with confidence={}".format(classes[pred.item()], confidence.item()))
# Plot
pp = pp.detach().cpu().numpy()
mp = mp.detach().cpu().numpy()
op = op.detach().cpu().numpy()
pad = np.zeros((num_classes, 1))
p = np.concatenate((pp[:num_classes], pad, mp[:num_classes], pad, op), axis=1)
p = np.concatenate((p, p), axis=0)
p = p.T.reshape(-1, num_classes).T
plt.switch_backend('agg')
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(p, cmap=plt.cm.jet)
# cax = ax.matshow(p, cmap=plt.cm.binary)
fig.colorbar(cax)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.set_yticklabels([''] + classes)
# save
sav_name = 'visual_tmp2.pdf'
plt.savefig(sav_name, format='pdf')
print("Saved in " + sav_name)

