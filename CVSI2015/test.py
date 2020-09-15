import os
import torch
from torchvision import transforms
import cv2
import numpy as np
import argparse
from model import vgg
from draw.drawConfmat import plotCM


parser = argparse.ArgumentParser()
parser.add_argument('--idc', type=int, default=1, help="Specify a gpu to run")
parser.add_argument("--param", type=str, required=True, help="Choose the parameters to load")
parser.add_argument("--net", type=str, required=True, help='Choose a model')
opt = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.idc)
device = torch.device("cuda:0")

# Prepare Test Data
root_dir = "/workspace/datasets/script_id/CVSI2015/GroundTruth_TestDataset_CVSI2015/Task4"
classes = ('ARB', 'BEN', 'ENG', 'GUJ', 'HIN', 'KAN', 'ORI', 'PUN', 'TAM', 'TEL')
num_classes = len(classes)
img_lists = os.listdir(root_dir)

# Prepare Network
if opt.net == 'vgg':
    net = vgg.CNN(num_classes, pretrain_use=opt.param)
elif opt.net == 'vgg_att':
    net = vgg.CNN_Att4(num_classes, pretrain_use=opt.param)
else:
    raise ModuleNotFoundError("Please specify the correct net using --net")
net = net.to(device)
net.eval()


def image_process(image):
    h, w = image.shape
    image = cv2.resize(image, (int(32 * w / h), 32))
    image = image / 255
    image = torch.from_numpy(image)
    image = image.unsqueeze(0).to(torch.float32)
    tfm = transforms.Normalize((0.5,), (0.5,))
    return tfm(image)


# Predict
correct = 0
total = 0
class_correct = list(0 for _ in range(num_classes))
class_total = list(0 for _ in range(num_classes))
conf_matrix = np.zeros([num_classes, num_classes])  # (i, j): i-Gt; j-Pr
for img_name in iter(img_lists):
    image = cv2.imread(os.path.join(root_dir, img_name), 0)
    image = image_process(image)
    image = image.unsqueeze(0).to(device)
    label = classes.index(img_name[:3])
    with torch.no_grad():
        outputs = net(image)
    _, predicted = torch.max(outputs.data, 1)
    total += 1
    TF = (predicted[0].item() == label) + 0
    correct += TF
    class_correct[label] += TF
    class_total[label] += 1
    conf_matrix[label, predicted[0].item()] += 1
accr = correct / total
print('Total number of images={}'.format(total))
print('Total number of correct images={}'.format(correct))
print('Accuracy on the Testset: %.2f %%' % (100 * accr))
classes_full = ('Arabic', 'Bengali', 'English', 'Gujrathi', 'Hindi', 'Kannada', 'Oriya', 'Punjabi', 'Tamil', 'Telegu')
for i in range(num_classes):
    print('Accuracy of %5s : %.2f %%' % (classes_full[i], 100 * class_correct[i] / class_total[i]))
plotCM(list(classes), conf_matrix, "./params/Confusion_Matrix_cnn.jpeg")

