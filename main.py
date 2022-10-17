from metaseg_main import meta_main, visualize
from compute_embeddings import embedding_main
from detect_cluster import get_cluster
from extend import extend_model
from predict import predict_all_images

import sys
import hydra
from omegaconf import DictConfig


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
        visualize(cfg, cfg.experiments[cfg.experiment]['train_dataset'], cfg.experiments[cfg.experiment]['train_split'])

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

    if cfg.tasks.infer_validation_data:
        print("Inference of validation data...")
        predict_all_images(cfg, 'val', debug_len = None)
        print("...done")
    




if __name__ == '__main__':
    sys.argv.append('hydra.run.dir=./')
    main()