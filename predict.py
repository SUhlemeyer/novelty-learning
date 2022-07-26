import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image

from src.models.deepv3 import DeepWV3Plus
import src.datasets.cityscapes_human as cs
import tqdm

from datetime import date

today = date.today()


def prediction(net, image):
    image = image.cuda()
    with torch.no_grad():
        out = net(image)
    out = out.data.cpu()
    out = F.softmax(out, 1)
    return out.numpy()


def from_path_to_input(path):
    img = Image.open(path)
    trans = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    img = trans(img)
    img = img.unsqueeze(0)
    return img


def load_network(exp, run, version, cls):
    net = nn.DataParallel(
        DeepWV3Plus(cls))
    net.load_state_dict(torch.load('/home/uhlemeyer/weights/DeepLabv3plus_best_{}_{}_{}.pth'.format(exp, run, version))['state_dict'], strict=False)
    net = net.cuda()
    net.eval()
    return net


def color_image(arr):
    predc = [(cs.trainid_to_color[arr[p, q]] if (arr[p, q] != 17) else (255, 102, 0) ) for p in range(arr.shape[0]) for q in range(arr.shape[1])]
    predc = np.asarray(predc).reshape(arr.shape + (3,))
    return predc


def id_image(arr):
    predc = [(cs.trainid_to_id[arr[p, q]] if (arr[p, q] != 17) else 24 ) for p in range(arr.shape[0]) for q in range(arr.shape[1])]
    predc = np.asarray(predc).reshape(arr.shape)
    return predc

exp = 'resnet152'
nov = 'human'
run = 'run0'
version = 4
net = load_network(exp, run, version, cls=18)
time = today.strftime("%b-%d-%Y")

if not os.path.exists('/home/uhlemeyer/Evaluation/cityscapes_{}/val/pred/{}/{}/{}_{}/semantic_id/'.format(nov, time, exp, run, version)):
            os.makedirs('/home/uhlemeyer/Evaluation/cityscapes_{}/val/pred/{}/{}/{}_{}/semantic_id/'.format(nov, time, exp,  run, version))
if not os.path.exists('/home/uhlemeyer/Evaluation/cityscapes_{}/val/pred/{}/{}/{}_{}/semantic_color/'.format(nov, time, exp,  run, version)):
        os.makedirs('/home/uhlemeyer/Evaluation/cityscapes_{}/val/pred/{}/{}/{}_{}/semantic_color/'.format(nov, time, exp,  run, version))


for city in tqdm.tqdm(os.listdir('/home/datasets/cityscapes/leftImg8bit/val')):
    for im in tqdm.tqdm(os.listdir(os.path.join('/home/datasets/cityscapes/leftImg8bit/val', city))):
        img = os.path.join('/home/datasets/cityscapes/leftImg8bit/val', city, im)
        inp = from_path_to_input(img)
        softmax = prediction(net, inp)
        softmax = np.squeeze(softmax)
        pred = np.argmax(softmax, axis=0)
        pred = np.squeeze(pred)

        pred_id = id_image(pred).astype("uint8")
        Image.fromarray(pred_id).save('/home/uhlemeyer/Evaluation/cityscapes_{}/val/pred/{}/{}/{}_{}/semantic_id/'.format(nov, time, exp, run, version) + im)
        pred_color = color_image(pred).astype("uint8")
        Image.fromarray(pred_color).save(
            '/home/uhlemeyer/Evaluation/cityscapes_{}/val/pred/{}/{}/{}_{}/semantic_color/'.format(nov, time, exp,  run, version) + im)
