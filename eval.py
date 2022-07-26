#!/usr/bin/python

import numpy as np
from PIL import Image

import sys, os
import tqdm

from torch.utils.data import DataLoader, Dataset



class Eval_Data(Dataset):
    
                
    def fulltotrain(target):
        """Transforms labels from full cityscapes labelset to training label set."""
        target = np.array(target)
        id_to_trainid = {7:0,8:1,11:2,12:3,13:4,17:5,19:6,20:7,21:8,22:9,23:10,24:11,25:11,26:13,27:14,28:15,31:16,32:17,33:18, 34:19}
        remapped_target = 255*np.ones(target.shape)
        for k, v in id_to_trainid.items():
            remapped_target[target == k] = v
        return remapped_target.astype('int32')
    
    
    def __init__(self,
                 pred_root='/home/uhlemeyer/Evaluation/a2d2/pred/DeepLabV3+_wideResNet38_guardrail/semantic_id',
                 gt_root='/home/datasets/A2D2/Validation/gt_Cityscapes_IDs',
                 map_fun=fulltotrain,
                 transform=None):
        """Load all filenames."""
        super(Eval_Data, self).__init__()
        self.pred_root = pred_root
        self.gt_root = gt_root
        self.transform = transform
        self.preds = []
        self.targets = []
        self.map_fun = map_fun

        for im in os.listdir(self.pred_root):
            #self.preds.append(os.path.join(self.pred_root, im))
            #self.targets.append(os.path.join(self.gt_root, im.replace('prediction', 'gtFine_labelIDs'))) #gtFine_labelIDs
            self.preds.append(os.path.join(self.pred_root, im))
            self.targets.append(os.path.join(self.gt_root, im.split('_')[0], im.replace('leftImg8bit', 'gtFine_labelIds')))
        
    
    def __len__(self):
        """Return number of images in the dataset."""
        return len(self.preds)

    def __getitem__(self, i):
        # Load images and perform augmentations with PIL
        pred = Image.open(self.preds[i]).convert('L')
        target = Image.open(self.targets[i]).convert('L')

        if self.map_fun is not None:
            pred = self.map_fun(pred)
            target = self.map_fun(target)

        return pred, target


def print_error(message):
    """Print an error message and quit"""
    print('\n-----\nERROR: ' + str(message) + "\n...good bye...")
    sys.exit(-1)


def generate_matrix(num_classes):
    """Generate empty confusion matrix"""
    max_id = num_classes
    return np.zeros(shape=(max_id, max_id), dtype=np.ulonglong)  # longlong for no overflows


def get_iou_score_for_label(label, conf_matrix):
    """Calculate and return IoU score for a particular label"""
    tp = np.longlong(conf_matrix[label, label])
    fn = np.longlong(conf_matrix[label, :].sum()) - tp
    notIgnored = [l for l in range(len(conf_matrix)) if not l == label]
    fp = np.longlong(conf_matrix[notIgnored, label].sum())
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return float(tp) / denom


def get_precision_score_for_label(label, conf_matrix):
    """Calculate and return IoU score for a particular label"""
    tp = np.longlong(conf_matrix[label, label])
    fn = np.longlong(conf_matrix[label, :].sum()) - tp
    notIgnored = [l for l in range(len(conf_matrix)) if not l == label]
    fp = np.longlong(conf_matrix[notIgnored, label].sum())
    denom = (tp + fp)
    if denom == 0:
        return float('nan')
    return float(tp) / denom


def get_recall_score_for_label(label, conf_matrix):
    """Calculate and return IoU score for a particular label"""
    tp = np.longlong(conf_matrix[label, label])
    fn = np.longlong(conf_matrix[label, :].sum()) - tp
    notIgnored = [l for l in range(len(conf_matrix)) if not l == label]
    fp = np.longlong(conf_matrix[notIgnored, label].sum())
    denom = (tp + fn)
    if denom == 0:
        return float('nan')
    return float(tp) / denom


def evaluate_pair(pred, gt, conf_matrix=None, ignore_in_eval_ids=None):
    """
    Main evaluation method. Evaluates pairs of prediction and ground truth with target type 'semantic_train_id',
    then updates confusion matrix
    """

    if ignore_in_eval_ids is not None:
        pred = pred[~np.isin(gt, ignore_in_eval_ids)]
        gt = gt[~np.isin(gt, ignore_in_eval_ids)]

    encoding_value = max(np.max(gt), np.max(pred)).astype(np.int32) + 1
    encoded = (gt.astype(np.int32) * encoding_value) + pred
    values, counts = np.unique(encoded, return_counts=True)
    if conf_matrix is None:
        conf_matrix = np.zeros((encoding_value, encoding_value))
    for value, c in zip(values, counts):
        pred_id = value % encoding_value
        gt_id = int((value - pred_id) / encoding_value)
        conf_matrix[gt_id][pred_id] += c
    return conf_matrix


def main():
    """
    Computes the mIoU for validation during training
    :param predictor: prediction object (just like in .prediction_wrappers.py)
    :param loader: validation dataset loader containing info: num_train_ids, ignore_in_eval_ids
    :return: miou score
    """
    #dat = Eval_Data(pred_root='/home/uhlemeyer/metasegio/cityscapes_val_bus/input/pred', gt_root='/home/datasets/cityscapes/gtFine/val')
    
    dat = Eval_Data(pred_root='/home/uhlemeyer/Evaluation/cityscapes_human/val/pred/Jun-03-2022/run0/semantic_id', gt_root='/home/datasets/cityscapes/gtFine/val')
    
    #dat = Eval_Data(pred_root='/home/uhlemeyer/Evaluation/a2d2/pred/DeepLabV3+_wideResNet38/semantic_id', gt_root='/home/datasets/A2D2/Validation/gt_Cityscapes_IDs')
    loader = DataLoader(dat)
    
    confusion_matrix = generate_matrix(20)
    ignore_ids = 255
    for pred, gt in tqdm.tqdm(loader):
        pred= np.array(pred).squeeze()
        gt = np.array(gt).squeeze()
        
        evaluate_pair(pred, gt, confusion_matrix, ignore_ids)
    classPrecisionList = {}
    classRecallList = {}
    classIoUList = {}
    for label in range(20):
        classPrecisionList[label] = get_precision_score_for_label(label, confusion_matrix)
        classRecallList[label] = get_recall_score_for_label(label, confusion_matrix)
        classIoUList[label] = get_iou_score_for_label(label, confusion_matrix)
    print("Precision: ", classPrecisionList, "\nRecall: ", classRecallList, "\nIoU: ", classIoUList)
    #classPrecisionList[19]=0
    sum_iou = 0
    sum_precision = 0
    sum_recall = 0
    c = 17
    anomaly = [11]
    
    for k in range(20):
        if classIoUList[k] > 0 and k not in anomaly:
            sum_iou += classIoUList[k]
            sum_precision += classPrecisionList[k]
            sum_recall += classRecallList[k]
    sum_iou2 = sum_iou
    sum_precision2 = sum_precision
    sum_recall2 = sum_recall
    for k in range(20):
        if classIoUList[k] > 0 and k in anomaly:
            sum_iou2 += classIoUList[k]
            sum_precision2 += classPrecisionList[k]
            sum_recall2 += classRecallList[k]
    print("Means old: IoU {:.2f}, Precision {:.2f}, Recall {:.2f}".format((sum_iou/c)*100, (sum_precision/c)*100, (sum_recall/c)*100))
    print("Means all: IoU {:.2f}, Precision {:.2f}, Recall {:.2f}".format(((sum_iou2)/(c+1))*100, ((sum_precision2)/(c+1))*100, ((sum_recall2)/(c+1))*100))
    
    for train_id in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]:
        print('{:.2f} & {:.2f} & {:.2f} \\\ \\hline'.format(classIoUList[train_id]*100, classPrecisionList[train_id]*100, classRecallList[train_id]*100))
    print("Means:\n")
    print('{:.2f} & {:.2f} & {:.2f} \\\ \\hline'.format((sum_iou/c)*100, (sum_precision/c)*100, (sum_recall/c)*100))
    print('{:.2f} & {:.2f} & {:.2f} \\\ \\hline'.format(((sum_iou2)/(c+1))*100, ((sum_precision2)/(c+1))*100, ((sum_recall2)/(c+1))*100))

if __name__ == "__main__":
    main()
