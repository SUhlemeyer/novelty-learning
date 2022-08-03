from metaseg_main import meta_main, visualize
from compute_embeddings import embedding_main
from detect_cluster import get_cluster
from extend import extend_model

import sys
import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import importlib
import tqdm
from PIL import Image

import hydra
from omegaconf import DictConfig
from torchvision.transforms import Compose, Normalize, ToTensor


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg: DictConfig):

    if cfg.tasks.metaseg_train:
        print("Compute and save metrics for training data...")
        meta_main(cfg, cfg.experiments[cfg.experiment]['train_dataset'], cfg.experiments[cfg.experiment]['train_split'])
        print("...done")

    if cfg.tasks.metaseg_test:
        print("Compute and save metrics for test data...")
        meta_main(cfg, cfg.experiments[cfg.experiment]['dataset'], cfg.experiments[cfg.experiment]['split'])
        print("...done")

    if cfg.tasks.metaseg_visualize:
        visualize(cfg, cfg, cfg.experiments[cfg.experiment]['train_dataset'], cfg.experiments[cfg.experiment]['train_split'])

    if cfg.tasks.compute_embeddings:
        print("Start computing embeddings...")
        embedding_main(cfg)
        print("...done")

    if cfg.tasks.detect_clusters:
        print("Looking for novel classes...")
        get_cluster(cfg)
        print("...done")
    
    if cfg.tasks.extend_model:
        print("Training of extended model...")
        extend_model(cfg)
        print("...done")

    




if __name__ == '__main__':
    sys.argv.append('hydra.run.dir=./')
    main()