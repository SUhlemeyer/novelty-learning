import os
import tqdm
import numpy as np
from PIL import Image
import torch
import pickle as pkl
from torchvision.transforms import Compose, Normalize, ToTensor
from src.helpers import embedding_net_init, get_image_index_to_components, get_component_gt, get_component_pred, \
                    wrapper_cutout_components, load_components, load_pred_gt


from src.metaseg.utils import test_regression_input, train_regression_input, meta_boost
from multiprocessing import Pool
from sklearn.metrics import r2_score

import hydra


class Embedding(object):
    def __init__(self, root, dataset, dataset_name, split, emb_model, model_name, transform):
        self.dataset = dataset
        self.transform = transform
        self.metrics_save_dir = os.path.join(root, dataset_name, model_name, split, "metrics")
        self.components_save_dir = os.path.join(root, dataset_name, model_name, split, "components")
        self.embedding_network = embedding_net_init(emb_model)

    def get_segment_metrics(self, i):
        image_path = self.dataset.images[i]

        metrics_save_path = os.path.join(self.metrics_save_dir, os.path.basename(image_path))[:-4] + ".p"
        components_save_path = os.path.join(self.components_save_dir, os.path.basename(image_path))[:-4] + ".p"

        if os.path.isfile(metrics_save_path):
            metrics = pkl.load(open(metrics_save_path, "rb"))
            components = pkl.load(open(components_save_path, "rb"))
            return metrics, components, image_path
        else:
            print('Error.. Compute metrics first!')
            exit()

    def get_embedding(self, image):
        with torch.no_grad():
            inp = self.transform(image).unsqueeze_(0).cuda()
            out = self.embedding_network(inp)
        return out.data.cpu().squeeze().numpy()


def embedding_main(cfg):

    dataset = hydra.utils.instantiate(cfg[cfg.experiments[cfg.experiment]['dataset']], split=cfg.experiments[cfg.experiment]['split'])
    model_name = cfg.experiments[cfg.experiment]['model']
    nmb_classes = cfg[model_name]['num_classes']
    transform = Compose([ToTensor(), Normalize(dataset.mean, dataset.std)])
    embedder = Embedding(cfg.io_root, dataset, cfg.experiments[cfg.experiment]['dataset'], cfg.experiments[cfg.experiment]['split'], cfg[cfg.embedding_network], model_name, transform)
    print("Embedder initialized")


    xa_all = []
    ya_true = []
    start_all = list([0])
    pred = []
    image_paths = []

    print('Loading training data...')
    xa, ya, x_mean, x_std, c_mean, c_std = train_regression_input(metrics_dir=os.path.join(cfg.io_root,
                                                                                           cfg.experiments[cfg.experiment]['train_dataset'],
                                                                                           model_name,
                                                                                           cfg.experiments[cfg.experiment]['train_split'],
                                                                                           'metrics'),
                                                                  nmb_classes=nmb_classes)

    print('Loading metrics...')
    for i in tqdm.tqdm(range(len(dataset))):
        metrics, components, image_path = embedder.get_segment_metrics(i)
        xa_tmp, ya_tmp = test_regression_input(test_metrics=metrics, test_nclasses=nmb_classes, xa_mean=x_mean,
                                               xa_std=x_std, classes_mean=c_mean, classes_std=c_std)
        xa_all.append(xa_tmp)
        ya_true.append(ya_tmp)
        image_paths.append(str(image_path))
        start_all += [start_all[-1] + len(metrics["S"])]
        pred += metrics['class']
    pred = np.asarray(pred)
    xa_all = np.concatenate(xa_all).squeeze()
    ya_true = np.concatenate(ya_true).squeeze()
    print(xa_all.shape)


    print('Predicting IoU...')
    meta_model = cfg.experiments[cfg.experiment]['meta_model']
    y_test = meta_boost(xa, ya, xa_all, ckpt_path=os.path.join(cfg.weight_dir, cfg[meta_model]['weights']))

    print("R^2 Boosting: ", r2_score(ya_true, y_test))

    pred_class_selection = cfg.experiments[cfg.experiment]['pred_class_selection']

    print('Filtering segments...')
    inds = np.zeros(pred.shape[0]).astype(np.bool)
    inds = np.logical_or(inds, (y_test < cfg.anom_threshold))
    inds = np.logical_and(inds, np.isin(pred, pred_class_selection))
    inds = np.argwhere(inds).flatten()
    component_image_mapping = get_image_index_to_components(inds, start_all)

    p_args = [(embedder.components_save_dir,
               v,
               image_paths[k],
               y_test[start_all[k]:start_all[k + 1]],
               cfg.experiments[cfg.experiment]['dataset'] + '_' + cfg.experiments[cfg.experiment]['split'],
               128,
               128,
               128,
               128,
               model_name) for k, v in
              component_image_mapping.items()]

    print('Extracting component information...')

    with Pool(20) as p:
        r = list(tqdm.tqdm(p.imap(wrapper_cutout_components, p_args), total=len(p_args)))
    r = [c for c in r if len(c['component_indices']) > 0]

    print('Computing embeddings...')

    crops = {
        'embeddings': [],
        'image_path': [],
        'component_index': [],
        'box': [],
        'gt': [],
        'pred': [],
        'datasets': [],
        'model_name': [],
        'image_level_index': [],
        'iou_pred': []
    }

    for c in tqdm.tqdm(r):
        preds, gt = load_pred_gt(image_path=c['image_path'],
                                 gt_dir=os.path.join(cfg.io_root, cfg.experiments[cfg.experiment]['dataset'], model_name,
                                                     cfg.experiments[cfg.experiment]['split'], 'input', 'gt'),
                                 pred_dir=os.path.join(cfg.io_root, cfg.experiments[cfg.experiment]['dataset'], model_name,
                                                       cfg.experiments[cfg.experiment]['split'], 'input', 'pred'))

        crops['image_path'].append(c['image_path'])
        crops['model_name'].append(c['model_name'])
        crops['datasets'].append(c['datasets'])
        crops['iou_pred'].append(c['iou_pred'])

        image = Image.open(c['image_path']).convert('RGB')

        for i, b in enumerate(c['boxes']):
            crops['embeddings'].append(embedder.get_embedding(image.crop(b)))
            crops['box'].append(b)
            crops['component_index'].append(c['component_indices'][i])
            crops['image_level_index'].append(len(crops['image_path']) - 1)
            crops['gt'].append(get_component_gt(gt, c['segment_indices'][i]))
            crops['pred'].append(get_component_pred(preds, c['segment_indices'][i]))

    print('Features: ', len(crops['embeddings']))

    print('Saving data...')
    with open(os.path.join(cfg.io_root, cfg.experiments[cfg.experiment]['dataset'], model_name, 'embeddings_{}.p'.format(cfg.run)), 'wb') as f:
        pkl.dump(crops, f)