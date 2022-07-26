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


def load_network(ckpt, cls):
    net = nn.DataParallel(
        DeepWV3Plus(cls))
    net.load_state_dict(torch.load(ckpt)['state_dict'], strict=False)
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

def main(img, save_path, ckpt, cls):

    net = load_network(ckpt, cls)

    inp = from_path_to_input(img)
    softmax = prediction(net, inp)
    softmax = np.squeeze(softmax)
    pred = np.argmax(softmax, axis=0)
    pred = np.squeeze(pred)

    pred_color = color_image(pred).astype("uint8")
    Image.fromarray(pred_color).save(save_path)
    return 0

if __name__ == '__main__':
    main(img = '/home/datasets/cityscapes/leftImg8bit/test/berlin/berlin_000057_000019_leftImg8bit.png', save_path = '/home/uhlemeyer/berlin_000057_000019_init.png', ckpt = '/home/uhlemeyer/weights/initials/cityscapes_human_initial.pth', cls=17)