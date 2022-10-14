import os
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import random


class Label(object):
    def __init__(self, name, id, trainid, category, catid, hasinstances,
                 ignoreineval, color):
        self.name = name
        self.id = id
        self.trainid = trainid
        self.category = category
        self.catid = catid
        self.hasinstances = hasinstances
        self.ignoreineval = ignoreineval
        self.color = color

    def __call__(self):
        print("name: %s\nid: %d\ntrainid: %d\ncategory: %s\ncatid:\
         %d\nhasinstances: %d\nignoreineval: %d\ncolor:%s" % (self.name,
                                                              self.id,
                                                              self.trainid,
                                                              self.category,
                                                              self.catid,
                                                              self.hasinstances,
                                                              self.ignoreineval,
                                                              str(self.color)))


num_classes = 18
num_categories = 8
void_ind = 255

# mean = (0.485, 0.456, 0.406)  # values from github repo where the model originates from
# std = (0.229, 0.224, 0.225)


labels = [
    #       name                     id         trainId          category        catId     hasInstances   ignoreInEval         color
    Label('unlabeled',              0,          void_ind,       'void',             0,      False,          True,           (0  , 0  , 0  )),
    Label('ego vehicle',            1,          void_ind,       'void',             0,      False,          True,           (0  , 0  , 0  )),
    Label('rectification border',   2,          void_ind,       'void',             0,      False,          True,           (0  , 0  , 0  )),
    Label('out of roi',             3,          void_ind,       'void',             0,      False,          True,           (0  , 0  , 0  )),
    Label('static',                 4,          void_ind,       'void',             0,      False,          True,           (0  , 0  , 0  )),
    Label('dynamic',                5,          void_ind,       'void',             0,      False,          True,           (111, 74 , 0  )),
    Label('ground',                 6,          void_ind,       'void',             0,      False,          True,           (81 , 0  , 81 )),
    Label('road',                   7,          0,              'flat',             1,      False,          False,          (128, 64 , 128)),
    Label('sidewalk',               8,          1,              'flat',             1,      False,          False,          (244, 35 , 232)),
    Label('parking',                9,          void_ind,       'flat',             1,      False,          True,           (250, 170, 160)),
    Label('rail track',             10,         void_ind,       'flat',             1,      False,          True,           (230, 150, 140)),
    Label('building',               11,         2,              'construction',     2,      False,          False,          (70 , 70 , 70 )),
    Label('wall',                   12,         3,              'construction',     2,      False,          False,          (102, 102, 156)),
    Label('fence',                  13,         4,              'construction',     2,      False,          False,          (190, 153, 153)),
    Label('guard rail',             14,         void_ind,       'construction',     2,      False,          True,           (180, 165, 180)),
    Label('bridge',                 15,         void_ind,       'construction',     2,      False,          True,           (150, 100, 100)),
    Label('tunnel',                 16,         void_ind,       'construction',     2,      False,          True,           (150, 120, 90 )),
    Label('pole',                   17,         5,              'object',           3,      False,          False,          (153, 153, 153)),
    Label('polegroup',              18,         void_ind,       'object',           3,      False,          True,           (153, 153, 153)),
    Label('traffic light',          19,         6,              'object',           3,      False,          False,          (250, 170, 30 )),
    Label('traffic sign',           20,         7,              'object',           3,      False,          False,          (220, 220, 0  )),
    Label('vegetation',             21,         8,              'nature',           4,      False,          False,          (107, 142, 35 )),
    Label('terrain',                22,         9,              'nature',           4,      False,          False,          (152, 251, 152)),
    Label('sky',                    23,         10,             'sky',              5,      False,          False,          (70 , 130, 180)),
    Label('person',                 24,         11,             'human',            6,      True,           False,          (220, 20 , 60 )),
    Label('rider',                  25,         12,             'human',            7,      True,           False,          (255, 0  , 0  )),
    Label('car',                    26,         13,             'vehicle',          7,      True,           False,          (0  , 0  , 142)),
    Label('truck',                  27,         14,             'vehicle',          7,      True,           False,          (0  , 0  , 70 )),
    Label('bus',                    28,         18,             'vehicle',          7,      True,           True,           (255, 102, 0  )),
    Label('caravan',                29,         void_ind,       'vehicle',          7,      True,           True,           (0  , 0  , 90 )),
    Label('trailer',                30,         void_ind,       'vehicle',          7,      True,           True,           (0  , 0  , 110)),
    Label('train',                  31,         15,             'vehicle',          7,      True,           False,          (0  , 80 , 100)),
    Label('motorcycle',             32,         16,             'vehicle',          7,      True,           False,          (0  , 0  , 230)),
    Label('bicycle',                33,         17,             'vehicle',          7,      True,           False,          (119, 11 , 32 )),
    Label('license plate',          -1,         -1,             'vehicle',          7,      False,          True,           (0  , 0  , 142)),
    Label('unlabeled',              void_ind,   void_ind,       'void',             0,      False,          True,           (0  , 0  , 0  ))
]

id_to_trainid = {label.id: (label.trainid if (label.id < 200) else 18) for label in labels}
id_to_color = {label.id: (label.color if (label.id < 200) else (255, 102, 0  )) for label in labels}
color_to_id = {label.color: label.id for label in labels}
id_to_name = {label.id: label.name for label in labels}
trainid_to_id = {label.trainid: (label.id if (label.trainid != void_ind) else 0) for label in labels}
trainid_to_name = {label.trainid: label.name for label in labels}
trainid_to_color = {label.trainid: label.color for label in labels}
name_to_rgba = {label.name: tuple(i / 255.0 for i in label.color) + (1.0,) for label in labels}
id_to_catid = {label.id: label.catid for label in labels}
trainid_to_catid = {label.trainid: label.catid for label in labels}
id_to_categoryname = {label.id: label.category for label in labels}
trainid_to_categoryname = {label.trainid: label.category for label in labels}

discover_mapping = {label.id: (trainid_to_name[label.trainid], trainid_to_color[label.trainid])
                    for label in labels}
pred_mapping = {label.trainid: (trainid_to_name[label.trainid], trainid_to_color[label.trainid]) for label in labels}


def fulltotrain(target):
    """Transforms labels from full cityscapes labelset to training label set."""
    remapped_target = target.clone()
    for k, v in id_to_trainid.items():
        remapped_target[target == k] = v
    return remapped_target


class Cityscapes(Dataset):
    def __init__(self, split='train',
                 root="/home/datasets/cityscapes",
                 map_fun=None,
                 transform=None,
                 label_mapping=None,
                 pred_mapping=None,
                 id_to_trainid = id_to_trainid,
                 id_to_color = id_to_color,
                 trainid_to_id = trainid_to_id,
                 trainid_to_color = trainid_to_color):
        """Load all filenames."""
        super(Cityscapes, self).__init__()
        if pred_mapping is None:
            pred_mapping = pred_mapping
        if label_mapping is None:
            label_mapping = discover_mapping
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.root = root
        self.split = split
        self.transform = transform
        self.label_mapping = label_mapping
        self.pred_mapping = pred_mapping
        self.images = []
        self.targets = []
        self.map_fun = map_fun
        self.id_to_trainid = id_to_trainid
        self.id_to_color = id_to_color
        self.trainid_to_id = trainid_to_id
        self.trainid_to_color = trainid_to_color
        for root, _, filenames in os.walk(os.path.join(self.root, 'leftImg8bit',
                                                       self.split)):
            for filename in filenames:
                if os.path.splitext(filename)[1] == '.png':
                    filename_base = '_'.join(filename.split('_')[:-1])
                    self.images.append(os.path.join(root, filename_base +
                                                    '_leftImg8bit.png'))

                    if self.split in ['train', 'val']:
                        target_root = os.path.join(self.root,
                                                   'gtFine', self.split,
                                                   os.path.basename(root))
                        self.targets.append(os.path.join(target_root,
                                                         filename_base +
                                                         '_gtFine_labelIds.png'))

    def __len__(self):
        """Return number of images in the datasets."""
        return len(self.images)

    def __getitem__(self, i):
        # Load images and perform augmentations with PIL
        image = Image.open(self.images[i]).convert('RGB')
        if self.split in ['train', 'val']:
            target = Image.open(self.targets[i]).convert('L')
        else:
            target = Image.fromarray(255 * np.ones((image.size[1], image.size[0])).astype('uint8'))

        if self.transform is not None:
            image, target = self.transform(image, target)

        if self.map_fun is not None:
            target = self.map_fun(target)

        return image, target