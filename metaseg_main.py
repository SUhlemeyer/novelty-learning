import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
from PIL import Image

from src.metaseg.utils import train_regression_input, test_regression_input, meta_boost, \
                              visualize_segments
from src.helpers import init_segmentation_network
from torchvision.transforms import Compose, ToTensor, Normalize
from src.metaseg.metrics import compute_metrics_components
from multiprocessing import Pool, set_start_method
from functools import partial

import hydra


class MetaRegression(object):
    def __init__(self, root, dataset, dataset_name, split, model, model_name, ckpt_path, nmb_classes, transform):

        self.segmentation_network = init_segmentation_network(model, ckpt_path, nmb_classes)
        self.dataset = dataset
        self.transform = transform

        self.metrics_save_dir = os.path.join(root, 'metasegio', dataset_name, model_name, split, "metrics")
        self.components_save_dir = os.path.join(root, 'metasegio', dataset_name, model_name, split, "components")
        self.probs_save_dir = os.path.join(root, 'metasegio', dataset_name, model_name, split, "probs")
        self.input_save_dir = os.path.join(root, 'metasegio', dataset_name, model_name, split, "input")

        if not os.path.exists(self.metrics_save_dir):
            os.makedirs(self.metrics_save_dir)
            print("Created:", self.metrics_save_dir)
        if not os.path.exists(self.components_save_dir):
            os.makedirs(self.components_save_dir)
            print("Created:", self.components_save_dir)
        if not os.path.exists(self.probs_save_dir):
            os.makedirs(self.probs_save_dir)
            print("Created:", self.probs_save_dir)
        if not os.path.exists(os.path.join(self.input_save_dir, 'pred')):
            os.makedirs(os.path.join(self.input_save_dir, 'pred'))
            print("Created:", os.path.join(self.input_save_dir, 'pred'))
        if not os.path.exists(os.path.join(self.input_save_dir, 'gt')):
            os.makedirs(os.path.join(self.input_save_dir, 'gt'))
            print("Created:", os.path.join(self.input_save_dir, 'gt'))

    def delete_model(self):
        del self.segmentation_network

    def get_segment_metrics(self, i):
        (image, target_id), image_path = self.dataset[i], self.dataset.images[i]
        id2trainid = self.dataset.id_to_trainid

        target_id = np.asarray(target_id)
        target = np.copy(target_id)
        for k, v in id2trainid.items():
            target[target_id == k] = v
        target = target.astype('int32')

        metrics_save_path = os.path.join(self.metrics_save_dir, os.path.basename(image_path))[:-4] + ".p"
        components_save_path = os.path.join(self.components_save_dir, os.path.basename(image_path))[:-4] + ".p"
        probs_save_path = os.path.join(self.probs_save_dir, os.path.basename(image_path))[:-4] + ".npy"
        pred_save_path = os.path.join(self.input_save_dir, 'pred', os.path.basename(image_path))
        gt_save_path = os.path.join(self.input_save_dir, 'gt', os.path.basename(image_path))

        if os.path.isfile(metrics_save_path):
            metrics = pickle.load(open(metrics_save_path, "rb"))
            components = pickle.load(open(components_save_path, "rb"))
        elif os.path.isfile(probs_save_path):
            probs = np.load(probs_save_path)
            metrics, components = compute_metrics_components(probs, target)
            pickle.dump(metrics, open(metrics_save_path, "wb"))
            pickle.dump(components, open(components_save_path, "wb"))
        else:
            print('Compute probs first!')
            self.compute_and_save_probs(i)

        if not os.path.isfile(gt_save_path):
            Image.fromarray(target.astype('uint8')).save(gt_save_path)
        if not os.path.isfile(pred_save_path):
            probs = np.load(probs_save_path)
            probs = np.squeeze(probs)
            pred = np.argmax(probs, axis=-1)
            pred = np.squeeze(pred)
            Image.fromarray(pred.astype('uint8')).save(pred_save_path)

        return metrics, components, image_path

    def compute_and_save_probs(self, i):
        (image, target), image_path = self.dataset[i], self.dataset.images[i]

        probs_save_path = os.path.join(self.probs_save_dir, os.path.basename(image_path))[:-4] + ".npy"
        if not os.path.isfile(probs_save_path):
            with torch.no_grad():
                x = self.transform(image).unsqueeze_(0).cuda()
                y = self.segmentation_network(x).permute(0, 2, 3, 1)
            probs = F.softmax(y, -1).data.cpu().numpy()[0]
            np.save(probs_save_path, probs)


def meta_main(cfg, dataset_name, split):
    dataset = hydra.utils.instantiate(cfg[dataset_name], split=split)
    model_name = cfg.experiments[cfg.experiment]['model']
    model = cfg[model_name]
    nmb_classes = cfg[model_name]['num_classes']
    ckpt_path = os.path.join(cfg.weight_dir, cfg.experiments[cfg.experiment]['init_weights'])
    
    transform = Compose([ToTensor(), Normalize(dataset.mean, dataset.std)])

    regressor = MetaRegression(cfg.io_root, dataset, dataset_name, split, model, model_name, ckpt_path, nmb_classes, transform)

    print('Predicting images and saving probabilities...')
    for i in tqdm.tqdm(range(len(dataset))):
        regressor.compute_and_save_probs(i)
    print('...done')
    regressor.delete_model()

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    print('Computing metrics and components...')
    with Pool(20) as p:
        p.map(partial(regressor.get_segment_metrics), range(len(dataset)))
    print('...done')


def visualize(cfg, dataset_name, split):
    dataset = hydra.utils.instantiate(cfg[dataset_name], split=split)
    model_name = cfg.experiments[cfg.experiment]['model']
    model = cfg[model_name]
    nmb_classes = cfg[model_name]['num_classes']
    ckpt_path = os.path.join(cfg.weight_dir, cfg.experiments[cfg.experiment]['init_weights'])
    
    transform = Compose([ToTensor(), Normalize(dataset.mean, dataset.std)])

    regressor = MetaRegression(cfg.io_root, dataset, dataset_name, split, model, model_name, ckpt_path, nmb_classes, transform)
    print('Visualizing quality estimates...')
    x_train, y_train, x_mean, x_std, c_mean, c_std = train_regression_input(metrics_dir=
                                                                            os.path.join(cfg.io_root,
                                                                                            'metasegio',
                                                                                            cfg.experiments[cfg.experiment]['train_dataset'],
                                                                                            model_name,
                                                                                            cfg.experiments[cfg.experiment]['train_split'],
                                                                                            'metrics'),
                                                                            nmb_classes=nmb_classes)

    for i in tqdm.tqdm(range(len(dataset))):
        metrics, components, image_path = regressor.get_segment_metrics(i)

        x_test, _ = test_regression_input(test_metrics=metrics, test_nclasses=nmb_classes,
                                            xa_mean=x_mean, xa_std=x_std, classes_mean=c_mean, classes_std=c_std)
        y_test = meta_boost(x_train, y_train, x_test, chkp_path=cfg[cfg.experiments[cfg.experiment]['meta_model']]['weights'])
        visualize_segments(y_test, components, os.path.join(cfg.io_root, 'metasegio', dataset_name, model_name,
                                                            split, "regression_masks",
                                                            os.path.basename(image_path).replace('.png', '_score.png')))