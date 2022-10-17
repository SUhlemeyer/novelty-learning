import numpy as np
from PIL import Image
import pickle as pkl
import os
from src.helpers import tsne_emb, dbscan_detect, delete_knowns
import hydra
from omegaconf import DictConfig



@hydra.main(config_path=".", config_name="config.yaml")
def get_cluster(cfg: DictConfig):

    dataset = hydra.utils.instantiate(cfg[cfg.experiments[cfg.experiment]['train_dataset']], cfg.experiments[cfg.experiment]['train_split'])

    save_dir = os.path.join(cfg.io_root, 'Cluster', cfg.experiment + '_' + cfg.run)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    load_dir = os.path.join(cfg.io_root, cfg.experiments[cfg.experiment]['dataset'], cfg.experiments[cfg.experiment]['model'])
    embed_dir = os.path.join(load_dir, 'embeddings.p')
    data = pkl.load(open(embed_dir, 'rb'))
    
    ix, level = delete_knowns(data, os.path.join(load_dir, cfg.experiments[cfg.experiment]['split']))
    data_tsne = tsne_emb(data, os.path.join(load_dir, 'tsne_data_{}.p'.format(cfg.run)))[ix]

   
    _, lbl, nmb, _ = dbscan_detect(data_tsne,  cfg.experiments[cfg.experiment]['eps'],  cfg.experiments[cfg.experiment]['min_samples'], "euclidean", 30, os.path.join(save_dir, 'dbscan_visualization.png'))

    for _, n in enumerate(nmb):
            if not os.path.exists(save_dir + '/cluster_{}/'.format(n)):
                os.makedirs(save_dir + '/cluster_{}/'.format(n))
            ind = np.flatnonzero(lbl == n)
            ili = level[ind]
            component_indices = np.asarray(data['component_index'])[ix]
            component_indices = component_indices[ind]
            boxes = np.asarray(data['box'])[ix]
            boxes = boxes[ind]
            paths = np.asarray(data['image_path'])[ili]

            if not os.path.exists(os.path.join(save_dir, 'cluster_{}/images/'.format(n))):
                os.makedirs(os.path.join(save_dir, 'cluster_{}/images/'.format(n)))
            if not os.path.exists(os.path.join(save_dir, 'cluster_{}/ood-objects/'.format(n))):
                os.makedirs(os.path.join(save_dir, 'cluster_{}/ood-objects/'.format(n)))
            if not os.path.exists(os.path.join(save_dir, 'cluster_{}/semantic_id/'.format(n))):
                os.makedirs(os.path.join(save_dir, 'cluster_{}/semantic_id/'.format(n)))
            if not os.path.exists(os.path.join(save_dir, 'cluster_{}/semantic_color/'.format(n))):
                os.makedirs(os.path.join(save_dir, 'cluster_{}/semantic_color/'.format(n)))

            for k, path in enumerate(paths):
                img = Image.open(path)
                img_name = path.split('/')[-1]
                img.save(os.path.join(save_dir, 'cluster_{}/images/'.format(n), img_name))

                prediction = np.array(Image.open(os.path.join(
                    load_dir, cfg.experiments[cfg.experiment]['split'], 'input/pred', img_name)))
                predID = [dataset.trainid_to_id[prediction[p, q]] for p in range(prediction.shape[0]) for q in
                          range(prediction.shape[1])]
                predID = np.asarray(predID).reshape(prediction.shape)

                Image.fromarray(predID.astype('uint8')).save(save_dir + '/cluster_{}/semantic_id/'.format(n) + img_name)
                
                

            for i in range(len(ind)):
                Image.open(paths[i]).crop(boxes[i]).save(os.path.join(save_dir, 'cluster_{}/ood-objects/'.format(n), 'ood_{}.png'.format(i)))


            for k, lvl in enumerate(ili):
                img_name = paths[k].split('/')[-1]
                components = np.asarray(pkl.load(open(os.path.join(
                                      load_dir, cfg.experiments[cfg.experiment]['split'], 'components',
                                      img_name.replace('png', 'p')), 'rb')))
                pred_mask = (Image.open(os.path.join(save_dir, 'cluster_{}/semantic_id'.format(n), img_name)))
                pred_mask = np.array(pred_mask)

                iou = np.asarray(data['iou_pred'][lvl])
                iou_threshold = np.max(iou[component_indices[k]-1]) + 0.2

                comp1_tmp = np.reshape(components, (components.shape[0]*components.shape[1]))
                comp2_tmp = np.zeros(comp1_tmp.shape)
                sgmts = component_indices[k]
                for s in sgmts:
                    comp2_tmp[np.abs(comp1_tmp) == s] = s
                comp_in_box = np.reshape(comp2_tmp, components.shape)

                for i in range(len(iou)):
                    if iou[i] <= iou_threshold:
                        pred_mask[np.abs(comp_in_box) == i+1] = 200 + n
                    elif iou[i] > iou_threshold and cfg.experiments[cfg.experiment]['ignore_background']:
                        pred_mask[np.abs(comp_in_box) == i+1] = 0
                Image.fromarray(pred_mask.astype('uint8')).save(os.path.join(save_dir, 'cluster_{}/semantic_id'.format(n), img_name))
                predc = [(dataset.id_to_color[pred_mask[p, q]] if (pred_mask[p, q]<200) else (255, 102, 0  )) for p in range(pred_mask.shape[0]) for q in
                          range(pred_mask.shape[1])]
                predc = np.asarray(predc).reshape(pred_mask.shape + (3,))

                Image.fromarray(predc.astype('uint8')).save(save_dir + '/cluster_{}/semantic_color/'.format(n) + img_name)

if __name__ == '__main__':
    get_cluster()
