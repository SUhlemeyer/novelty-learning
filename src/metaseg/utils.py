import torch
import torch.nn as nn
import pickle as pkl
import numpy as np
import os
import tqdm

from PIL import Image
from sklearn import ensemble
import hydra



def init_segmentation_network(model, ckpt_path, num_classes, gpu):
    # device = torch.device( "cuda:2" if torch.cuda.is_available( ) else "cpu" )
    print("Checkpoint file:", ckpt_path)
    print("Load PyTorch model", end="", flush=True)
    network = hydra.utils.instantiate(model, num_classes=num_classes)
    #network = nn.DataParallel(network)
    #network = network.module
    network.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)
    network = network.cuda().eval()
    print("... ok")
    return network


def meta_boost(x_train, y_train, x_test, ckpt_path='./booster.pickle.dat'):
    if not os.path.exists(ckpt_path):
        print("Train Meta Regressor")
        reg = ensemble.GradientBoostingRegressor()
        reg.fit(x_train, y_train)
        print("Save checkpoint file:", ckpt_path)
        pkl.dump(reg, open(ckpt_path, "wb"))
    else:
        print("Load checkpoint file:", ckpt_path)
        reg = pkl.load(open(ckpt_path, "rb"))
    y_test_pred = np.clip(reg.predict(x_test), 0, 1)
    return y_test_pred


# Helpers
# ---------------------------------------------------------------------------------------------
def metrics_to_nparray(metrics, names, normalize=False, non_empty=False, all_metrics=(), mean=None, std=None):
    I = range(len(metrics['S_in']))
    if non_empty:
        I = np.asarray(metrics['S_in']) > 0
    M = np.asarray([np.asarray(metrics[m])[I] for m in names])
    MM = []
    if not all_metrics:
        MM = M.copy()
    else:
        MM = np.asarray([np.asarray(all_metrics[m])[I] for m in names])
    if normalize:
        mu = np.mean(MM, axis=-1) if mean is None else mean
        std = np.std(MM, axis=-1) if std is None else std
        for i in range(M.shape[0]):
            if names[i] != "class":
                M[i] = (np.asarray(M[i]) - mu[i]) / (std[i] + 1e-10)
        M = np.squeeze(M.T)
        return M, mu, std
    M = np.squeeze(M.T)
    return M


def metrics_to_dataset(metrics, nclasses, non_empty=True, all_metrics=(), xa_mean=None,
                       xa_std=None, classes_mean=None, classes_std=None):
    x_names = sorted(
        [m for m in metrics if m not in ["class", "iou", "iou0"] and "cprob" not in m and "E_max" not in m])
    class_names = ["cprob" + str(i) for i in range(nclasses) if "cprob" + str(i) in metrics]

    if xa_mean is not None and xa_std is not None:
        xa, xa_mean, xa_std = metrics_to_nparray(metrics,
                                                 x_names,
                                                 normalize=True,
                                                 non_empty=non_empty,
                                                 all_metrics=all_metrics,
                                                 mean=xa_mean,
                                                 std=xa_std)
    else:
        xa, xa_mean, xa_std = metrics_to_nparray(metrics,
                                                 x_names,
                                                 normalize=True,
                                                 non_empty=non_empty,
                                                 all_metrics=all_metrics)

    if classes_mean is not None and classes_std is not None:
        classes, classes_mean, classes_std = metrics_to_nparray(metrics,
                                                                class_names,
                                                                normalize=True,
                                                                non_empty=non_empty,
                                                                all_metrics=all_metrics,
                                                                mean=classes_mean,
                                                                std=classes_std)
    else:
        classes, classes_mean, classes_std = metrics_to_nparray(metrics,
                                                                class_names,
                                                                normalize=True,
                                                                non_empty=non_empty,
                                                                all_metrics=all_metrics)

    ya = metrics_to_nparray(metrics, ["iou"], normalize=False, non_empty=non_empty)
    y0a = metrics_to_nparray(metrics, ["iou0"], normalize=False, non_empty=non_empty)
    return xa, classes, ya, y0a, x_names, class_names, xa_mean, xa_std, classes_mean, classes_std


def train_regression_input(metrics_dir, nmb_classes):
    files = sorted([f for f in os.listdir(metrics_dir) if os.path.splitext(f)[-1] == '.p'])
    num_imgs = len(files)
    metrics, _ = concatenate_metrics(num_imgs, metrics_dir, files)
    xa, classes, ya, _, _, _, xa_mean, xa_std, classes_mean, classes_std = metrics_to_dataset(metrics,
                                                                                              nclasses=nmb_classes,
                                                                                              non_empty=False,
                                                                                              )
    xa = np.concatenate((xa, classes), axis=-1)
    return xa, ya, xa_mean, xa_std, classes_mean, classes_std


def test_regression_input(test_metrics=None, test_nclasses=None, xa_mean=None, xa_std=None,
                          classes_mean=None, classes_std=None):
    xa_test, classes_test, ya_test, *_ = metrics_to_dataset(test_metrics, test_nclasses, non_empty=False, xa_mean=xa_mean,
                                                   xa_std=xa_std, classes_mean=classes_mean, classes_std=classes_std)
    xa_test = np.concatenate((xa_test, classes_test), axis=-1)
    return xa_test, ya_test


def concatenate_metrics(num_imgs, metrics_dir, im_path):
    read_path = os.path.join(metrics_dir, im_path[0])
    metrics = pkl.load(open(read_path, "rb"))
    num_imgs = list(range(1, num_imgs))
    start = list([0, len(metrics["S"])])

    for i, k in enumerate(tqdm.tqdm(num_imgs, total=len(num_imgs) + 1, initial=1)):
        read_path = os.path.join(metrics_dir, im_path[k])
        m = pkl.load(open(read_path, "rb"))
        start += [start[-1] + len(m["S"])]
        for j in metrics:
            metrics[j] += m[j]

    return metrics, start


def visualize_segments(metric, comp, save_path):

    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    r = np.asarray(metric)
    r = 1 - 0.5 * r
    g = np.asarray(metric)
    b = 0.3 + 0.35 * np.asarray(metric)

    r = np.concatenate((r, np.asarray([0, 1])))
    g = np.concatenate((g, np.asarray([0, 1])))
    b = np.concatenate((b, np.asarray([0, 1])))

    components = np.asarray(comp.copy(), dtype='int16')
    components[components < 0] = len(r) - 1
    components[components == 0] = len(r)

    img = np.zeros(components.shape + (3,))
    x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    x = x.reshape(-1)
    y = y.reshape(-1)

    img[x, y, 0] = r[components[x, y] - 1]
    img[x, y, 1] = g[components[x, y] - 1]
    img[x, y, 2] = b[components[x, y] - 1]

    img = np.asarray(255 * img).astype('uint8')
    Image.fromarray(img).save(save_path)

