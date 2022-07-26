from metaseg_main import meta_main, visualize
from compute_embeddings import embedding_main

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

    # print("Compute and save metrics for training data...")
    # meta_main(cfg, cfg.experiments[cfg.experiment]['train_dataset'], cfg.experiments[cfg.experiment]['train_split'])
    # print("...done")

    # print("Compute and save metrics for test data...")
    # meta_main(cfg, cfg.experiments[cfg.experiment]['dataset'], cfg.experiments[cfg.experiment]['split'])
    # print("...done")

    print("Start computing embeddings...")
    # visualize(cfg, cfg, cfg.experiments[cfg.experiment]['train_dataset'], cfg.experiments[cfg.experiment]['train_split'])
    embedding_main(cfg)
    print("...done")

    




if __name__ == '__main__':
    sys.argv.append('hydra.run.dir=./')
    main()