"""Generate results text (at ./test/ch8_test_task2.txt) to upload on ICDAR2017 to see our accuracy"""
import os
import cv2
import torch
import torchvision.transforms as transforms
import argparse
import time
from model import vgg
import dynamic_bar

parser = argparse.ArgumentParser()
parser.add_argument('--param', '-p', type=str, required=True, help='parameter files which the network will need')
parser.add_argument('--net', type=str, default='vgg', help='choose the network')
parser.add_argument('--idc', type=int, default="1", help="Specify a gpu to run")
parser.add_argument('--suffix', type=str, default='', help='choose the prefix in the saving name')
opt = parser.parse_args()
print(opt)
time.sleep(5)

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.idc)

classes = ("Arabic", "Latin", "Chinese", "Japanese", "Korean", "Bangla", "Symbols")
num_classes = len(classes)
device = torch.device("cuda: 0")

def resize_keep(image):
    h, w = image.shape[:2]
    if h >= w:
        neww = 64
        newh = int(neww * h / w)
    else:
        newh = 64
        neww = int(newh * w / h)
    return cv2.resize(image, (neww, newh))

if opt.net == 'gs':
    net = vgg.VGG16(num_classes, pretrain_use=opt.param)
elif opt.net == 'gspa':
    net = vgg.GSPA(num_classes, pretrain_use=opt.param)
else:
    raise ModuleNotFoundError("Please specify the correct net using --net")
net.to(device)
net.eval()

if not os.path.exists('test'):
    os.makedirs('test')

text_name = './test/ch8_test_task2_' + opt.suffix + '.txt'
text = open(text_name, 'w')
num_imgs = 97619
root = '/workspace/datasets/script_id/ICDAR17/TestSet'
transf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.48, 0.48, 0.48), (0.2, 0.2, 0.2))])
for i in range(num_imgs):
    dynamic_bar.progress_bar(i, num_imgs)
    img_name = 'word_' + str(i+1) + '.png'
    image = cv2.imread(os.path.join(root, img_name), 1)
    image = resize_keep(image)
    image = transf(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        y = net(image)
    _, predicted = torch.max(y.data, 1)
    label_p = classes[predicted.item()]
    text.writelines(img_name + ',' + label_p + '\n')

text.seek(text.tell() - 1)
text.truncate()  # delete the last '\n'
text.close()
print('{} is saved!'.format(text_name))
print('Great! Prediction Done!!!')
