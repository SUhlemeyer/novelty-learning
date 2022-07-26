import os
import pickle
import time
import torch
import torch.nn.functional as F
import numpy as np
import importlib

from config import roots, settings, datasets, models
from src.metaseg.utils import init_segmentation_network, train_regression_input, test_regression_input, meta_boost, visualize_segments
# from src.datasets.cityscapes_human import Cityscapes as Exp1
# from src.datasets.cityscapes_bus import Cityscapes as Exp2
# from src.datasets.a2d2 import A2D2
from torchvision.transforms import Compose, ToTensor, Normalize
from src.metaseg.metrics import compute_metrics_components

from multiprocessing import Process, Pool, set_start_method
from functools import partial


class meta_regression(object):
    def __init__(self, args, roots, dataset, transform):
        self.segmentation_network = init_segmentation_network(args)
        self.dataset = dataset
        self.transform = transform

        self.metrics_save_dir = os.path.join(roots.io_root, settings.DATASET, settings.MODEL, settings.SPLIT, "metrics")
        self.components_save_dir = os.path.join(roots.io_root, settings.DATASET, settings.MODEL, settings.SPLIT, "components")
        self.probs_save_dir = os.path.join(roots.io_root, settings.DATASET, settings.MODEL, settings.SPLIT, "probs")
        if not os.path.exists(self.metrics_save_dir):
            os.makedirs(self.metrics_save_dir)
            print("Created:", self.metrics_save_dir)
        if not os.path.exists(self.components_save_dir):
            os.makedirs(self.components_save_dir)
            print("Created:", self.components_save_dir)
        if not os.path.exists(self.probs_save_dir):
            os.makedirs(self.probs_save_dir)
            print("Created:", self.probs_save_dir)

    def get_segment_metrics(self, i):
        (image, target_id), image_path = self.dataset[i], self.dataset.images[i]

        target_id = np.asarray(target_id)
        target = np.copy(target_id)
        for k, v in self.dataset.id2train_id.items():
            target[target == k] = v
        target = target.astype('int32')

        metrics_save_path = os.path.join(self.metrics_save_dir, os.path.basename(image_path))[:-4] + ".p"
        components_save_path = os.path.join(self.components_save_dir, os.path.basename(image_path))[:-4] + ".p"
        probs_save_path = os.path.join(self.probs_save_dir, os.path.basename(image_path))[:-4] + ".npy"

        if os.path.isfile(metrics_save_path):
            metrics = pickle.load(open(metrics_save_path, "rb"))
            components = pickle.load(open(components_save_path, "rb"))
            pickle.dump(metrics, open(metrics_save_path, "wb"))
        elif os.path.isfile(probs_save_path):
            start = time.time()
            probs = np.load(probs_save_path)
            metrics, components = compute_metrics_components(probs, target)
            pickle.dump(metrics, open(metrics_save_path, "wb"))
            pickle.dump(components, open(components_save_path, "wb"))
            end = time.time()
            hours, rem = divmod(end - start, 3600)
            minutes, seconds = divmod(rem, 60)
            print("\nProcessed: {} in {:0>2}:{:05.2f}".format(image_path, int(minutes), seconds))
        else:
            start = time.time()
            x = self.transform(image).unsqueeze_(0).cuda()
            y = self.segmentation_network(x).permute(0, 2, 3, 1)
            probs = F.softmax(y, -1).data.cpu().numpy()[0]
            metrics, components = compute_metrics_components(probs, target)
            pickle.dump(metrics, open(metrics_save_path, "wb"))
            pickle.dump(components, open(components_save_path, "wb"))
            end = time.time()
            hours, rem = divmod(end - start, 3600)
            minutes, seconds = divmod(rem, 60)
            print("\nProcessed: {} in {:0>2}:{:05.2f}".format(image_path, int(minutes), seconds))
        return metrics, components, image_path

    def compute_and_save_probs(self, i):
        (image, target), image_path = self.dataset[i], self.dataset.images[i]
        probs_save_path = os.path.join(self.probs_save_dir, os.path.basename(image_path))[:-4] + ".npy"
        if not os.path.isfile(probs_save_path):
            start = time.time()
            with torch.no_grad():
                x = self.transform(image).unsqueeze_(0).cuda()
                y = self.segmentation_network(x).permute(0, 2, 3, 1)
            probs = F.softmax(y, -1).data.cpu().numpy()[0]
            np.save(probs_save_path, probs)
            end = time.time()
            hours, rem = divmod(end - start, 3600)
            minutes, seconds = divmod(rem, 60)
            print("Saved {} in {:0>2}:{:05.2f}".format(probs_save_path, int(minutes), seconds))


def main():
    args = dict(
        dataset=settings.DATASET.name,
        model_name=settings.MODEL,
        pretrained_model=models[settings.MODEL].model_weights,
    )

    dataset =getattr(importlib.import_module(datasets[args['dataset']].module_name), datasets[args['dataset']].class_name)(
        **datasets[args['dataset']].kwargs,
    )
    transform = Compose([ToTensor(), Normalize(dataset.mean, dataset.std)])

    # if settings.LOADER_TYPE == 'cityscapes_human':
    #     dataset = Exp1(root=roots.dataset_root, split=settings.SPLIT)
    #     transform = Compose([ToTensor(), Normalize(dataset.mean, dataset.std)])
    # elif settings.LOADER_TYPE == 'cityscapes_bus':
    #     dataset = Exp2(root=roots.dataset_root, split=settings.SPLIT)
    #     transform = Compose([ToTensor(), Normalize(dataset.mean, dataset.std)])
    # elif settings.LOADER_TYPE == 'a2d2':
    #     dataset = A2D2(root=roots.dataset_root, split=settings.SPLIT)
    #     transform = Compose([ToTensor(), Normalize(dataset.mean, dataset.std)])
    # else:
    #     print('Loader not implemented!')
    #     exit()

    x_train, y_train, x_mean, x_std, c_mean, c_std = train_regression_input()
    regressor = meta_regression(args, roots, dataset, transform)

    if settings.COMPUTE_PROBS:
        with Pool(2) as p:
            p.map(partial(regressor.compute_and_save_probs), range(len(dataset)))
    if settings.COMPUTE_METRICS:
        with Pool(20) as p:
            p.map(partial(regressor.get_segment_metrics), range(len(dataset)))












    for i in range(len(dataset)):
        metrics, components, image_path = regressor.get_segment_metrics(i)


        x_test = test_regression_input(test_metrics=metrics, test_nclasses=19, xa_mean=x_mean, xa_std=x_std,
                                       classes_mean=c_mean, classes_std=c_std)


        y_test = meta_boost(x_train, y_train, x_test)

        visualize_segments(y_test, components, os.path.join(roots.io_root, "regression_masks",
                                                            os.path.basename(image_path).replace('.png', '_ms.png')))


if __name__ == '__main__':
    main()
