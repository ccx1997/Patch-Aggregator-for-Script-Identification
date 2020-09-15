"""Show Spatial Attention Map directly on the original image, and Make Prediction
  Anyone who want to use this script should pay attention to his/her:
    L18: IMG_NAME
    L34: Network chose
  Be careful that returned value from network should be (weight, output), where weight has w scalar values,
and output is the probability distribution"""
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from model import vgg

# Here we use gpu to calculate
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:0')
# device = torch.device('cpu')

# Prepare an image
IMG_NAME = './test/japanese_000575_1.jpg'
img = cv2.imread(IMG_NAME, 1)
h, w = img.shape[:2]
w = int(32 * w / h)
h = 32
img = cv2.resize(img, (w, h))

# Prepare weights and Predict from Net
classes = ('Arabic', 'Cambodian', 'Chinese', 'English', 'Greek', 'Hebrew', 'Japanese', 'Kannada',
           'Korean', 'Mongolian', 'Russian', 'Thai', 'Tibetan')
num_classes = len(classes)
tfm = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
image = tfm(img)
image = image.unsqueeze(0).to(device)
net = vgg.CNN_Att0(num_classes, pretrain_use='./params/siw_param/vgg0022.pkl')
net.to(device)
net.eval()
with torch.no_grad():
    weight, output = net(image, show=True)
    weight = weight.contiguous().view(1, -1)
_, output = torch.max(output, 1)
print("I think the text line in the image {} is: {}".format(IMG_NAME, classes[output.item()]))
weight = weight.detach().cpu().numpy()
weight = np.minimum(weight * 8, 1.0) * 255
weight = weight.astype(np.uint8)
weight = cv2.resize(weight, (w, h))
weight = cv2.applyColorMap(weight, 2)
cover = img * 0.4 + weight * 0.6
map_dir = './spatialAttentionMap'
if not os.path.exists(map_dir):
    os.makedirs(map_dir)
sav_name = os.path.join(map_dir, IMG_NAME.split('/')[-1])
cv2.imwrite(sav_name, cover)
print('Saved as %s' % sav_name)
