import numpy as np
import os
import pickle as pkl
from scipy.ndimage.measurements import label
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import random
import matplotlib.pyplot as plt
import hydra


def embedding_net_init(model):
    net = hydra.utils.instantiate(model).cuda().eval()
    return net


def get_image_index_to_components(component_indices, start):
    out = {}

    for i in range(len(start) - 1):
        index = component_indices[np.logical_and(start[i] <= component_indices,
                                                 component_indices < start[i + 1])]
        out[i] = [j - start[i] + 1 for j in index]
    return out


def wrapper_cutout_components(args):
    return cutout_components(*args)


def cutout_components(components_save_dir,
                      component_indices,
                      im_path,
                      iou_pred,
                      dataset,
                      min_height,
                      min_width,
                      min_crop_height,
                      min_crop_width,
                      model_name
                      ):
    # scene = im_path.split('/')[-3]
    # town = im_path.split('/')[-4]
    comp_name = os.path.basename(im_path).split('.')[0]
    read_path = os.path.join(components_save_dir, comp_name + '.p')
    components = pkl.load(open(read_path, "rb"))

    crops = {'datasets': dataset,
             'model_name': model_name,
             'embeddings': [],
             'image_path': im_path,
             'boxes': [],
             'iou_pred': iou_pred,
             'component_indices': [],
             'segment_indices': [],
             'img_crops': []}
    mask = np.zeros(components.shape)
    for i in component_indices:
        mask[np.abs(components) == i] = 1
    structure = np.ones((3, 3), dtype=np.int)
    segments, _ = label(mask, structure)

    for cindex in np.unique(segments)[1:]:
        segment_indices = np.argwhere(segments == cindex)
        if segment_indices.shape[0] > 0:
            upper, left = segment_indices.min(0)
            lower, right = segment_indices.max(0)
            if (lower - upper) < min_height or (right - left) < min_width:
                continue

            if (right - left) < min_crop_width:
                margin = min_crop_width - (right - left)
                if left - (margin // 2) < 0:
                    left = 0
                    right = left + min_crop_width
                elif right + (margin // 2) > components.shape[1]:
                    right = components.shape[1]
                    left = right - min_crop_width

                if right > components.shape[1] or left < 0:
                    raise IndexError('Image with shape {} is too small for a {} x {} crop'.format(
                        components.shape, min_crop_height, min_crop_width))
            if (lower - upper) < min_crop_height:
                margin = min_crop_height - (lower - upper)
                if upper - (margin // 2) < 0:
                    upper = 0
                    lower = upper + min_crop_height
                elif lower + (margin // 2) > components.shape[0]:
                    lower = components.shape[0]
                    upper = lower - min_crop_height

                if lower > components.shape[0] or upper < 0:
                    raise IndexError('Image with shape {} is too small for a {} x {} crop'.format(
                        components.shape, min_crop_height, min_crop_width))

            crops['boxes'].append((left, upper, right, lower))
            crops['component_indices'].append(np.unique(np.abs(components[segments == cindex])))
            crops['segment_indices'].append(segment_indices)
    return crops


def get_component_gt(gt, segment_indices):
    cls, cls_counts = np.unique(gt[segment_indices[:, 0], segment_indices[:, 1]], return_counts=True)
    return cls[np.argsort(cls_counts)[-1]]


def get_component_pred(pred, segment_indices):
    return pred[segment_indices[0, 0], segment_indices[0, 1]]


def load_components(image_path, save_dir):
    comp_name = image_path.split('/')[-1].replace('png', 'p')
    comp_dir = os.path.join(save_dir, comp_name)
    comp = pkl.load(open(comp_dir, 'rb'))
    return comp


def load_pred_gt(image_path, gt_dir, pred_dir):
    pred = np.asarray(Image.open(os.path.join(pred_dir, os.path.basename(image_path))))
    gt = np.asarray(Image.open(os.path.join(gt_dir, os.path.basename(image_path))))
    return pred, gt


def pca_emb(data):
    print('Computing PCA...')
    n_comp = 50 if 50 < min(len(data['embeddings']),
                            data['embeddings'][0].shape[0]) else min(len(data['embeddings']),
                                                                    data['embeddings'][0].shape[0])
    embeddings = PCA(
        n_components=n_comp
    ).fit_transform(np.stack(data['embeddings']).reshape((-1, data['embeddings'][0].shape[0])))

    return embeddings


def tsne_emb(data, load_dir):
    if not os.path.exists(load_dir):
        pca_data = pca_emb(data)
        print('Computing t-SNE for plotting')
        tsne_embedding = TSNE(n_components=2,
                            perplexity=30,
                            learning_rate=200.0,
                            early_exaggeration=12.0,
                            n_iter=1000,
                            verbose=3,
                            ).fit_transform(pca_data)
        pkl.dump(tsne_embedding, open(load_dir, 'wb'))
    else:
        print('Loading t-SNE for plotting')
        tsne_embedding = pkl.load(open(load_dir, 'rb'))
    
    return tsne_embedding


def dbscan_detect(data, eps, min_samples, metric, t, save_path):
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(data)
    data = np.asarray(data)
    label = model.labels_ + 1
    core = model.core_sample_indices_
    for i in range(len(label)):
        if i not in core:
            label[i] = 0
    cluster = []
    indices = []
    n_clusters = len(np.unique(label))
    # v = 0
    for i in range(1, n_clusters):
        # r = random.random()
        # g = random.random()
        # b = random.random()
        # color = (r, g, b)
        cluster.append([data[label == i]])
        indices.append(np.where([label == i])[1])
        # x, y = np.array_split(cluster[v][0], 2, axis=1)
        # plt.scatter(x, y, c=color)
        # v += 1
    tic = 0
    nmb = []
    for i in range(1, n_clusters):
        if len(cluster[tic][0]) > t:
            print(len(cluster[tic][0]))
            nmb.append(i)
            r = random.random()
            g = random.random()
            b = random.random()
            color = (r, g, b)
            x, y = np.array_split(cluster[tic][0], 2, axis=1)
            plt.scatter(x, y, c=color)
            tic += 1
        else:
            del cluster[tic]
    xlim, ylim = np.array_split(data, 2, axis=1)
    plt.xlim(xlim.min() * 1.1, xlim.max() * 1.1)
    plt.ylim(ylim.min() * 1.1, ylim.max() * 1.1)
    plt.savefig(save_path)
    return cluster, label, nmb, indices


def delete_knowns(data, load_dir):
    ix = []
    level = []

    for i in range(len(data['component_index'])):
        comp_indices = data['component_index'][i]
        ili = data['image_level_index'][i]
        comp_path = os.path.join(load_dir, 'components', (data['image_path'][ili].split('/')[-1]).replace('png', 'p'))
        pred_path = os.path.join(load_dir, 'input/pred', data['image_path'][ili].split('/')[-1])
        pred = np.asarray(Image.open(pred_path))
        comp_org = pkl.load(open(comp_path, 'rb'))
        comp = comp_org.flatten()
        cls_counts = []
        for j in comp_indices:
            cls_counts.append(np.count_nonzero(comp == j))
        cls_counts = np.asarray(cls_counts)

        cls_big = comp_indices[cls_counts > 0]
        pred_mask = np.inf*np.ones((pred.shape))
        for k in cls_big:
            pred_mask[comp_org == k] = pred[comp_org == k]
        cls, cnts = np.unique(pred_mask[pred_mask<np.inf], return_counts=True)
        cls2 = cls[cnts > 0]
        
        
        if len(cls2) > 0:
            ix.append(i)
            level.append(ili)

    return ix, np.array(level)