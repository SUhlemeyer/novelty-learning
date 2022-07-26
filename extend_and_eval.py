#!/usr/bin/env python3

from ensurepip import version
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from src.imageaugmentations import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from src.models.pspnet import PSPNet
from src.datasets.cityscapes import fulltotrain
from torch.utils.data import Dataset
import math


import torchvision.transforms as tt
from PIL import Image

import src.datasets.cityscapes as cs
import tqdm
import math
import sys
from torch.utils.data import DataLoader, Dataset

from datetime import date

exp = 'experiment4b'
run = 'run0'
nov = 'guardrail'
seed = 693
vers = 4

MODEL = 'PSPNet'
CLASSES = 19
max_epoch = 70


class Train_Data(Dataset):
    def __init__(self,
                 root='/home/uhlemeyer/outputs',
                 map_fun=fulltotrain,
                 transform=None):
        """Load all filenames."""
        super(Train_Data, self).__init__()
        self.root = root
        self.transform = transform
        self.images = []
        self.targets = []
        self.map_fun = map_fun

        for im in os.listdir(os.path.join(self.root, 'Cluster_{}_{}/{}'.format(exp, run, nov), 'images')):
            self.images.append(os.path.join(self.root, 'Cluster_{}_{}/{}'.format(exp, run, nov), 'images', im))
            self.targets.append(os.path.join(self.root, 'Cluster_{}_{}/{}'.format(exp, run, nov), 'semantic_id', im))

        filenames = os.listdir(os.path.join(self.root, 'Cluster_experiment4a_run0/', 'memory'))

        for im in filenames: 
            self.images.append(os.path.join('/home/uhlemeyer/A2D2/images/train', im))
            self.targets.append(os.path.join('/home/uhlemeyer/A2D2/labels_id/train', im))
            



    def __len__(self):
        """Return number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, i):
        # Load images and perform augmentations with PIL
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')

        if self.transform is not None:
            image, target = self.transform(image, target)

        if self.map_fun is not None:
            target = self.map_fun(target)

        return image, target


def loss_ce(inp, target, weight=None):
    b, h, w = target.shape[0], target.shape[1], target.shape[2]
    inp, target = inp.cuda(), target.cuda()
    loss = nn.NLLLoss(ignore_index=255, reduction='sum', weight=weight)
    m = nn.LogSoftmax(dim=1)
    output = loss(m(inp), target)/(b*h*w)
    return output


def loss_d(logits, target, T=1):
    logits = logits[:, :-1, :, :]
    b, c, h, w = target.shape[0], target.shape[1], target.shape[2], target.shape[3]
    m = nn.Softmax(dim=1)
    M0 = m(target/T)
    loss = torch.sum(- M0 * F.log_softmax(logits/T, 1))
    loss = T*T*loss/(b*h*w)
    return loss


def poly_schd(e):
    return math.pow((1 - (e / (max_epoch*dat.__len__())/5)), 0.9)


def calculate_weights(target):
    """
    Calculate weights of the classes based on training crop (NVIDIA)
    """
    target = np.asarray(target).flatten()
    hist = np.zeros(CLASSES+1)
    for i in range(CLASSES+1):
        hist[i] = np.sum(target == i)/len(target)
    hist = hist*(1-hist)+1
    return torch.tensor(1/hist).float().cuda()




def prediction(net, image):
    image = image.cuda()
    with torch.no_grad():
        out = net(image)
    out = out.data.cpu()
    out = F.softmax(out, 1)
    return out.numpy()


def from_path_to_input(path):
    img = Image.open(path)
    trans = tt.Compose([tt.ToTensor(), tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    img = trans(img)
    img = img.unsqueeze(0)
    return img


def load_network(exp, run, vers, cls):
    net = nn.DataParallel(
        PSPNet(cls))
    net.load_state_dict(torch.load('/home/uhlemeyer/weights/PSPNet_best_{}_{}_{}.pth'.format(exp, run, vers))['state_dict'], strict=False)
    net = net.cuda()
    net.eval()
    return net


def color_image(arr):
    predc = [(cs.trainid_to_color[arr[p, q]] if (arr[p, q] != 19) else (255, 102, 0) ) for p in range(arr.shape[0]) for q in range(arr.shape[1])]
    predc = np.asarray(predc).reshape(arr.shape + (3,))
    return predc


def id_image(arr):
    predc = [(cs.trainid_to_id[arr[p, q]] if (arr[p, q] != 19) else 34 ) for p in range(arr.shape[0]) for q in range(arr.shape[1])]
    predc = np.asarray(predc).reshape(arr.shape)
    return predc

class Eval_Data(Dataset):
    
                
    def fulltotrain(target):
        """Transforms labels from full cityscapes labelset to training label set."""
        target = np.array(target)
        id_to_trainid = {7:0,8:1,11:2,12:3,13:4,17:5,19:6,20:7,21:8,22:9,23:10,24:11,25:12,26:13,27:14,28:15,31:16,32:17,33:18, 34:19}
        remapped_target = 255*np.ones(target.shape)
        for k, v in id_to_trainid.items():
            remapped_target[target == k] = v
        return remapped_target.astype('int32')
    
    
    def __init__(self,
                 pred_root='/home/uhlemeyer/Evaluation/a2d2/pred/DeepLabV3+_wideResNet38_guardrail/semantic_id',
                 gt_root='/home/datasets/A2D2/Validation/gt_Cityscapes_IDs',
                 map_fun=fulltotrain,
                 transform=None):
        """Load all filenames."""
        super(Eval_Data, self).__init__()
        self.pred_root = pred_root
        self.gt_root = gt_root
        self.transform = transform
        self.preds = []
        self.targets = []
        self.map_fun = map_fun

        for im in os.listdir(self.pred_root):
            #self.preds.append(os.path.join(self.pred_root, im))
            #self.targets.append(os.path.join(self.gt_root, im.replace('prediction', 'gtFine_labelIDs'))) #gtFine_labelIDs
            self.preds.append(os.path.join(self.pred_root, im))
            self.targets.append(os.path.join(self.gt_root, im))
        
    
    def __len__(self):
        """Return number of images in the dataset."""
        return len(self.preds)

    def __getitem__(self, i):
        # Load images and perform augmentations with PIL
        pred = Image.open(self.preds[i]).convert('L')
        target = Image.open(self.targets[i]).convert('L')

        if self.map_fun is not None:
            pred = self.map_fun(pred)
            target = self.map_fun(target)

        return pred, target


def print_error(message):
    """Print an error message and quit"""
    print('\n-----\nERROR: ' + str(message) + "\n...good bye...")
    sys.exit(-1)


def generate_matrix(num_classes):
    """Generate empty confusion matrix"""
    max_id = num_classes
    return np.zeros(shape=(max_id, max_id), dtype=np.ulonglong)  # longlong for no overflows


def get_iou_score_for_label(label, conf_matrix):
    """Calculate and return IoU score for a particular label"""
    tp = np.longlong(conf_matrix[label, label])
    fn = np.longlong(conf_matrix[label, :].sum()) - tp
    notIgnored = [l for l in range(len(conf_matrix)) if not l == label]
    fp = np.longlong(conf_matrix[notIgnored, label].sum())
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return float(tp) / denom


def get_precision_score_for_label(label, conf_matrix):
    """Calculate and return IoU score for a particular label"""
    tp = np.longlong(conf_matrix[label, label])
    fn = np.longlong(conf_matrix[label, :].sum()) - tp
    notIgnored = [l for l in range(len(conf_matrix)) if not l == label]
    fp = np.longlong(conf_matrix[notIgnored, label].sum())
    denom = (tp + fp)
    if denom == 0:
        return float('nan')
    return float(tp) / denom


def get_recall_score_for_label(label, conf_matrix):
    """Calculate and return IoU score for a particular label"""
    tp = np.longlong(conf_matrix[label, label])
    fn = np.longlong(conf_matrix[label, :].sum()) - tp
    notIgnored = [l for l in range(len(conf_matrix)) if not l == label]
    fp = np.longlong(conf_matrix[notIgnored, label].sum())
    denom = (tp + fn)
    if denom == 0:
        return float('nan')
    return float(tp) / denom


def evaluate_pair(pred, gt, conf_matrix=None, ignore_in_eval_ids=None):
    """
    Main evaluation method. Evaluates pairs of prediction and ground truth with target type 'semantic_train_id',
    then updates confusion matrix
    """

    if ignore_in_eval_ids is not None:
        pred = pred[~np.isin(gt, ignore_in_eval_ids)]
        gt = gt[~np.isin(gt, ignore_in_eval_ids)]

    encoding_value = max(np.max(gt), np.max(pred)).astype(np.int32) + 1
    encoded = (gt.astype(np.int32) * encoding_value) + pred
    values, counts = np.unique(encoded, return_counts=True)
    if conf_matrix is None:
        conf_matrix = np.zeros((encoding_value, encoding_value))
    for value, c in zip(values, counts):
        pred_id = value % encoding_value
        gt_id = int((value - pred_id) / encoding_value)
        conf_matrix[gt_id][pred_id] += c
    return conf_matrix


#random.seed(seed)
# Prepare Data
# trans = Compose([RandomHorizontalFlip(), RandomCrop(1000), ToTensor(), Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
# dat = Train_Data(transform=trans)
# datloader = DataLoader(dat, batch_size=5, shuffle=True, drop_last=True, num_workers=10)


# # Adapt weights
# pretrained_model_path = "/home/uhlemeyer/weights/initials/PSPNet_guardrail_initial.pth"
# dict = torch.load(pretrained_model_path)['state_dict']


# # Prepare Network
# network0 = nn.DataParallel(PSPNet(classes=CLASSES))
# network0.load_state_dict(torch.load(pretrained_model_path)['state_dict'], strict=False)
# # Freeze all weights
# for param in network0.parameters():
#     param.requires_grad = False

# network0.cuda()
# network0.eval()

# torch.random.manual_seed(seed)

# weights_last_layer = torch.load(pretrained_model_path)['state_dict']['module.cls.4.weight']
# bias_last_layer = torch.load(pretrained_model_path)['state_dict']['module.cls.4.bias']
# weights_aux_layer = torch.load(pretrained_model_path)['state_dict']['module.aux.4.weight']
# bias_aux_layer = torch.load(pretrained_model_path)['state_dict']['module.aux.4.bias']

# new_weight = torch.rand((1, weights_last_layer.shape[1], weights_last_layer.shape[2], weights_last_layer.shape[3]), requires_grad=True)
# new_weight = (new_weight * np.sqrt(2 / weights_last_layer.shape[1])).cuda()
# new_weight = torch.cat((weights_last_layer, new_weight), dim=0)
# del dict['module.cls.4.weight']
# dict.update({'module.cls.4.weight': new_weight})

# new_bias = torch.rand(1, requires_grad=True).cuda()
# new_bias = torch.cat((bias_last_layer, new_bias), dim=0)
# del dict['module.cls.4.bias']
# dict.update({'module.cls.4.bias': new_bias})

# aux_weight = torch.rand((1, weights_aux_layer.shape[1], weights_aux_layer.shape[2], weights_aux_layer.shape[3]), requires_grad=True)
# aux_weight = (aux_weight * np.sqrt(2 / weights_aux_layer.shape[1])).cuda()
# aux_weight = torch.cat((weights_aux_layer, aux_weight), dim=0)
# del dict['module.aux.4.weight']
# dict.update({'module.aux.4.weight': aux_weight})

# aux_bias = torch.rand(1, requires_grad=True).cuda()
# aux_bias = torch.cat((bias_aux_layer, aux_bias), dim=0)
# del dict['module.aux.4.bias']
# dict.update({'module.aux.4.bias': aux_bias})


# network = nn.DataParallel(PSPNet(classes=CLASSES+1))
# network.load_state_dict(dict, strict=False)
# for param in network.parameters():
#     param.requires_grad = False


# network.module.cls[0].weight.requires_grad = True
# network.module.cls[1].weight.requires_grad = True
# network.module.cls[4].weight.requires_grad = True
# network.module.ppm.features[0][1].weight.requires_grad = True
# network.module.ppm.features[0][2].weight.requires_grad = True
# network.module.ppm.features[1][1].weight.requires_grad = True
# network.module.ppm.features[1][2].weight.requires_grad = True
# network.module.ppm.features[2][1].weight.requires_grad = True
# network.module.ppm.features[3][2].weight.requires_grad = True


# optimizer = optim.Adam(network.parameters(), lr=5*1e-5, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_schd)

# print('Networks loaded successfully!')

# # Start Training

# lambda_d = 0.5
# best_loss = np.inf

# for epoch in range(0, max_epoch):
#     loss_avg = []
#     print('Epoch {}/{}'.format(epoch + 1, max_epoch))
#     i = 0
#     iters = len(datloader)
#     for x, y in datloader:
#         weights = calculate_weights(y)
#         optimizer.zero_grad()
#         x = x.cuda()
#         y = y.cuda()
#         out = network(x)
#         out_old = network0(x)
#         loss1 = loss_ce(out, y, weight=weights)
#         loss2 = loss_d(out, out_old)
#         loss = (1-lambda_d) * loss1 + lambda_d * loss2
#         print(loss1, loss2)
#         loss.backward()
#         optimizer.step()
#         loss_avg.append(loss.item())
#         print('{} Loss: {}'.format(i, loss.item()))
#         scheduler.step(int(epoch + i / iters))
#         i += 1
#     #scheduler.step()
#     loss_avg = sum(loss_avg) / len(loss_avg)
#     print('Average Loss in {}. epoch: {}'.format(epoch+1, loss_avg))
#     if loss_avg < best_loss:
#         best_loss = loss_avg
#         save_basename = MODEL + "_best_" + "{}_{}_{}.pth".format(exp, run, vers)
#         save_path = os.path.join('/home/uhlemeyer/weights/', save_basename)
#         print('Saving checkpoint', save_path)
#         torch.save({'state_dict': network.state_dict()}, save_path)



today = date.today()


net = load_network(exp, run, vers, cls=20)

if not os.path.exists('/home/uhlemeyer/Evaluation/a2d2_{}/val/pred/{}/{}_{}/semantic_id/'.format(exp, today.strftime("%b-%d-%Y"), run, vers)):
            os.makedirs('/home/uhlemeyer/Evaluation/a2d2_{}/val/pred/{}/{}_{}/semantic_id/'.format(exp, today.strftime("%b-%d-%Y"), run, vers))
if not os.path.exists('/home/uhlemeyer/Evaluation/a2d2_{}/val/pred/{}/{}_{}/semantic_color/'.format(exp, today.strftime("%b-%d-%Y"), run, vers)):
        os.makedirs('/home/uhlemeyer/Evaluation/a2d2_{}/val/pred/{}/{}_{}/semantic_color/'.format(exp, today.strftime("%b-%d-%Y"), run, vers))


for im in tqdm.tqdm(os.listdir('/home/uhlemeyer/A2D2/images/val')):
    img = os.path.join('/home/uhlemeyer/A2D2/images/val', im)
    inp = from_path_to_input(img)
    softmax = prediction(net, inp)
    softmax = np.squeeze(softmax)
    pred = np.argmax(softmax, axis=0)
    pred = np.squeeze(pred)

    pred_id = id_image(pred).astype("uint8")
    Image.fromarray(pred_id).save('/home/uhlemeyer/Evaluation/a2d2_{}/val/pred/{}/{}_{}/semantic_id/'.format(exp, today.strftime("%b-%d-%Y"), run, vers) + im)
    pred_color = color_image(pred).astype("uint8")
    Image.fromarray(pred_color).save(
        '/home/uhlemeyer/Evaluation/a2d2_{}/val/pred/{}/{}_{}/semantic_color/'.format(exp, today.strftime("%b-%d-%Y"), run, vers) + im)


dat = Eval_Data(pred_root='/home/uhlemeyer/Evaluation/a2d2_{}/val/pred/{}/{}_{}/semantic_id'.format(exp, today.strftime("%b-%d-%Y"), run, vers), gt_root='/home/uhlemeyer/A2D2/labels_id/val')
    
loader = DataLoader(dat)

confusion_matrix = generate_matrix(20)
ignore_ids = 255
for pred, gt in tqdm.tqdm(loader):
    pred= np.array(pred).squeeze()
    gt = np.array(gt).squeeze()
    
    evaluate_pair(pred, gt, confusion_matrix, ignore_ids)
classPrecisionList = {}
classRecallList = {}
classIoUList = {}
for label in range(20):
    classPrecisionList[label] = get_precision_score_for_label(label, confusion_matrix)
    classRecallList[label] = get_recall_score_for_label(label, confusion_matrix)
    classIoUList[label] = get_iou_score_for_label(label, confusion_matrix)
print("Precision: ", classPrecisionList, "\nRecall: ", classRecallList, "\nIoU: ", classIoUList)
sum_iou = 0
sum_precision = 0
sum_recall = 0
c = 14
anomaly = [19]

for k in range(20):
    if classIoUList[k] > 0 and k not in anomaly:
        sum_iou += classIoUList[k]
        sum_precision += classPrecisionList[k]
        sum_recall += classRecallList[k]
sum_iou2 = sum_iou
sum_precision2 = sum_precision
sum_recall2 = sum_recall
for k in range(20):
    if classIoUList[k] > 0 and k in anomaly:
        sum_iou2 += classIoUList[k]
        sum_precision2 += classPrecisionList[k]
        sum_recall2 += classRecallList[k]
print("Means old: IoU {:.2f}, Precision {:.2f}, Recall {:.2f}".format((sum_iou/c)*100, (sum_precision/c)*100, (sum_recall/c)*100))
print("Means all: IoU {:.2f}, Precision {:.2f}, Recall {:.2f}".format(((sum_iou2)/(c+1))*100, ((sum_precision2)/(c+1))*100, ((sum_recall2)/(c+1))*100))

for train_id in [0,1,2,4,5,6,7,8,10,11,13,14,17,18,19]:
    print('{:.2f} & {:.2f} & {:.2f} \\\ \\hline'.format(classIoUList[train_id]*100, classPrecisionList[train_id]*100, classRecallList[train_id]*100))
print("Means:")
print('{:.2f} & {:.2f} & {:.2f} \\\ \\hline'.format((sum_iou/c)*100, (sum_precision/c)*100, (sum_recall/c)*100))
print('{:.2f} & {:.2f} & {:.2f} \\\ \\hline'.format(((sum_iou2)/(c+1))*100, ((sum_precision2)/(c+1))*100, ((sum_recall2/(c+1))*100)))