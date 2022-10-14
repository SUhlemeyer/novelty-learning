import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image

import tqdm
from src.metaseg.utils import init_segmentation_network
from datetime import date
import hydra

today = date.today()


def prediction(net, image):
    image = image.cuda()
    with torch.no_grad():
        out = net(image)
    out = out.data.cpu()
    out = F.softmax(out, 1)
    return out.numpy()


def from_path_to_input(path, dataset):
    img = Image.open(path)
    trans = Compose([ToTensor(), Normalize(dataset.mean, dataset.std)])
    img = trans(img)
    img = img.unsqueeze(0)
    return img


def color_image(arr, trainid2color):
    predc = [trainid2color[arr[p, q]] for p in range(arr.shape[0]) for q in range(arr.shape[1])]
    predc = np.asarray(predc).reshape(arr.shape + (3,))
    return predc


def id_image(arr, trainid2id):
    predc = [trainid2id[arr[p, q]] for p in range(arr.shape[0]) for q in range(arr.shape[1])]
    predc = np.asarray(predc).reshape(arr.shape)
    return predc


def inference_i(image_path, net, dataset, save_dir):
    inp = from_path_to_input(image_path, dataset)
    softmax = prediction(net, inp)
    softmax = np.squeeze(softmax)
    pred = np.argmax(softmax, axis=0)
    pred = np.squeeze(pred)

    im = image_path.split('/')[-1]

    if not os.path.exists(os.path.join(save_dir, 'semantic_id')):
                os.makedirs(os.path.join(save_dir, 'semantic_id'))
    if not os.path.exists(os.path.join(save_dir, 'semantic_color')):
            os.makedirs(os.path.join(save_dir, 'semantic_color'))

    pred_id = id_image(pred, dataset.trainid_to_id).astype("uint8")
    Image.fromarray(pred_id).save(os.path.join(save_dir, 'semantic_id', im))
    pred_color = color_image(pred, dataset.trainid_to_color).astype("uint8")
    Image.fromarray(pred_color).save(os.path.join(save_dir, 'semantic_color', im))


def predict_all_images(cfg, split, debug_len = None):

    dataset_name = cfg.experiments[cfg.experiment]['dataset']
    model_name = cfg.experiments[cfg.experiment]['model']
    model = cfg[model_name]
    nmb_classes = cfg[model_name]['num_classes'] + cfg.experiments[cfg.experiment]['k']
    ckpt_path = os.path.join(cfg.weight_dir, 'extended', cfg.run, cfg.experiment + '_best.pth')

    net = init_segmentation_network(model, ckpt_path, nmb_classes)
    dataset = hydra.utils.instantiate(cfg[dataset_name], split)

    time = today.strftime("%b-%d-%Y")
    save_dir = os.path.join(cfg.io_root, 'eval', dataset_name, split, 'pred', time, cfg.run)

    if debug_len == None:
        debug_len = len(dataset)
    for i in tqdm.tqdm(range(debug_len)):
        image_path = dataset.images[i]
        inference_i(image_path, net, dataset, save_dir)