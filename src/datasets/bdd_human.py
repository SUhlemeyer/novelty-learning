import os
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import json


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


num_classes = 19
num_categories = 8
void_ind = 255
cityscapes_void = void_ind
mean = (0.485, 0.456, 0.406)    # values from github repo where the model originates from
std = (0.229, 0.224, 0.225)


# certainly predicted class weights
# inverse class frequency: 1/fc
class_weights = [13.125335865494446, # road
                 238.3252895627352, # sidewalk
                 5.231706629299469,   # building
                 372.1681457586619, # wall
                 113.46970059025203, # fence
                 88.63735807212268, # pole
                 751.0481676150063, # traffic light
                 331.7430548012021, # traffic sign
                 5.592955253034644, # vegetation
                 366.3121837812603, # terrain
                 2.0953097584503784, # sky
                 1120.9601534414574, # person
                 6057.1627631131205, # rider
                 28.62002696121724, # car
                 225.68301618046542, # truck
                 1006.1506296016964, # bus
                 944.6020920419622, # train
                 18487.476661028515, # motorcycle
                 36219.37538515202] # bicycle


# median weighted class frequencies: median(fc)/fc
# class_weights = (0.6908071508154972,
#                  12.543436292775537,
#                  0.2753529804894458,
#                  19.58779714519273,
#                  5.972089504750106,
#                  4.6651241090590885,
#                  39.528850927105594,
#                  17.46016077901064,
#                  0.2943660659491918,
#                  19.27958862006633,
#                  0.11027946097107255,
#                  58.997902812708276,
#                  318.79804016384844,
#                  1.5063172084851177,
#                  11.87805348318239,
#                  52.95529629482613,
#                  49.7158995811559,
#                  973.0250874225535,
#                  1906.2829150080013,
#                  0.019744487178015965) # unlabeled



labels = [
    #       name               id    trainId   category         catId     hasInstances   ignoreInEval   color
    Label("unlabeled",          0,      255,    "void",         0,          False,          True,       (0, 0, 0)),
    Label("dynamic",            1,      255,    "void",         0,          False,          True,       (111, 74, 0)),
    Label("ego vehicle",        2,      255,    "void",         0,          False,          True,       (0, 0, 0)),
    Label("ground",             3,      255,    "void",         0,          False,          True,       (81, 0, 81)),
    Label("static",             4,      255,    "void",         0,          False,          True,       (0, 0, 0)),
    Label("parking",            5,      255,    "flat",         1,          False,          True,       (250, 170, 160)),
    Label("rail track",         6,      255,    "flat",         1,          False,          True,       (230, 150, 140)),
    Label("road",               7,      0,      "flat",         1,          False,          False,      (128, 64, 128)),
    Label("sidewalk",           8,      1,      "flat",         1,          False,          False,      (244, 35, 232)),
    Label("bridge",             9,      255,    "construction", 2,          False,          True,       (150, 100, 100)),
    Label("building",           10,     2,      "construction", 2,          False,          False,      (70, 70, 70)),
    Label("fence",              11,     4,      "construction", 2,          False,          False,      (190, 153, 153)),
    Label('something',          12,     255,    'construction', 2,          False,          False,      (102, 102, 156)),
    Label("guard rail",         13,     255,    "construction", 2,          False,          True,       (180, 165, 180)),
    Label("tunnel",             14,     255,    "construction", 2,          False,          True,       (150, 120, 90)),
    Label("wall",               15,     3,      "construction", 2,          False,          False,      (102, 102, 156)),
    Label("banner",             16,     255,    "object",       3,          False,          True,       (250, 170, 100)),
    Label("billboard",          17,     255,    "object",       3,          False,          True,       (220, 220, 250)),
    Label("lane divider",       18,     255,    "object",       3,          False,          True,       (255, 165, 0)),
    Label("parking sign",       19,     255,    "object",       3,          False,          False,      (220, 20, 60)),
    Label("pole",               20,     5,      "object",       3,          False,          False,      (153, 153, 153)),
    Label("polegroup",          21,     255,    "object",       3,          False,          True,       (153, 153, 153)),
    Label("street light",       22,     255,    "object",       3,          False,          True,       (220, 220, 100)),
    Label("traffic cone",       23,     255,    "object",       3,          False,          True,       (255, 70, 0)),
    # Label("traffic device",     24,     255,    "object",       3,          False,          True,       (220, 220, 220)),
    Label("novel class",        24,     255,    "object",       3,          False,          True,       (255, 102, 0)),
    Label("traffic light",      25,     6,      "object",       3,          False,          False,      (250, 170, 30)),
    Label("traffic sign",       26,     7,      "object",       3,          False,          False,      (220, 220, 0)),
    Label("traffic sign frame", 27,     255,    "object",       3,          False,          True,       (250, 170, 250),),
    Label("terrain",            28,     9,      "nature",       4,          False,          False,      (152, 251, 152)),
    Label("vegetation",         29,     8,      "nature",       4,          False,          False,      (107, 142, 35)),
    Label("sky",                30,     10,     "sky",          5,          False,          False,      (70, 130, 180)),
    Label("person",             31,     255,     "human",        6,          True,           False,      (220, 20, 60)),
    Label("rider",              32,     255,     "human",        6,          True,           False,      (255, 0, 0)),
    Label("bicycle",            33,     16,     "vehicle",      7,          True,           False,      (119, 11, 32)),
    Label("bus",                34,     13,     "vehicle",      7,          True,           False,      (0, 60, 100)),
    Label("car",                35,     11,     "vehicle",      7,          True,           False,      (0, 0, 142)),
    Label("caravan",            36,     255,    "vehicle",      7,          True,           True,       (0, 0, 90)),
    Label("motorcycle",         37,     15,     "vehicle",      7,          True,           False,      (0, 0, 230)),
    Label("trailer",            38,     255,    "vehicle",      7,          True,           True,       (0, 0, 110)),
    Label("train",              39,     14,     "vehicle",      7,          True,           False,      (0, 80, 100)),
    Label("truck",              40,     12,     "vehicle",      7,          True,           False,      (0, 0, 70)),
]

bdd_to_cityscapes = {
    0: 0,
    1: 5,
    2: 1,
    3: 6,
    4: 4,
    5: 9,
    6: 10,
    7: 7,
    8: 8,
    9: 15,
    10: 11,
    11: 13,
    12: cityscapes_void,
    13: 14,
    14: 16,
    15: 12,
    16: cityscapes_void,
    17: cityscapes_void,
    18: cityscapes_void,
    19: cityscapes_void,
    20: 17,
    21: 18,
    22: cityscapes_void,
    23: cityscapes_void,
    24: cityscapes_void,
    25: 19,
    26: 20,
    27: cityscapes_void,
    28: 22,
    29: 21,
    30: 23,
    31: 24,
    32: 25,
    33: 33,
    34: 28,
    35: 26,
    36: 29,
    37: 32,
    38: 30,
    39: 31,
    40: 27,
}

id_to_trainid = {label.id: label.trainid for label in labels}
id_to_color = {label.id: label.color for label in labels}
color_to_id = {label.color: (label.id if (label.color != tuple([0, 0, 0])) else 0) for label in labels}
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
pred_mapping = {label.trainid: (trainid_to_name[label.trainid] if (label.trainid != void_ind) else 'unlabeled',
                                trainid_to_color[label.trainid] if (label.trainid != void_ind) else (0, 0, 0))
                for label in labels}


def fulltotrain(target):
    """Transforms labels from full cityscapes labelset to training label set."""
    remapped_target = target.clone()
    for k, v in id_to_trainid.items():
        remapped_target[target == k] = v
    return remapped_target


def traintofull(target):
    """Transforms labels from training label set to the full label set."""
    remapped_target = target.clone()
    for k, v in trainid_to_id.items():
        remapped_target[target == k] = v
    return remapped_target


def traintocolor(target):
    return fulltocolor(traintofull(target))


def fulltocolor(target):
    """Maps labels to their RGB colors in cityscapes."""
    colors = [(label.id, label.color) for label in labels if label.id != -1]
    colors.sort(key=lambda x: x[0])
    colors = np.array([x[1] for x in colors], dtype=np.uint8)

    target = target.numpy()
    if len(target.shape) == 2:
        b = 1
        h = target.shape[0]
        w = target.shape[1]
    elif len(target.shape) == 3:
        b, h, w = target.shape
    else:
        b, _, h, w = target.shape
    target = target.reshape(b, -1)

    rgb_target = np.concatenate([np.expand_dims(colors[t].reshape(h, w, 3).transpose([2, 0, 1]), 0) for t in target])
    return torch.from_numpy(rgb_target)



class BDD(Dataset):
    def __init__(self, split='train',
                 root="/home/uhlemeyer/data/bdd100k",
                 transform=None,
                 label_mapping=pred_mapping,
                 pred_mapping=pred_mapping):
        """Load all filenames."""
        super(BDD, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.label_mapping = label_mapping
        self.pred_mapping = pred_mapping
        self.images = []
        self.targets = []
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        for root, _, filenames in os.walk(os.path.join(self.root, 'images',
                                                       self.split)):
            for filename in filenames:

                filename_base = filename.split('.')[0]
                self.images.append(os.path.join(root, filename))

                if self.split in ['train', 'val']:
                    target_root = os.path.join(self.root,
                                               'labels', self.split)
                    self.targets.append(os.path.join(target_root,
                                                     filename_base +
                                                     '_train_id.png'))

    def __len__(self):
        """Return number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, i):
        # Load images and perform augmentations with PIL
        image = Image.open(self.images[i]).convert('RGB')
        if self.split in ['train', 'val']:
            target = Image.open(self.targets[i]).convert('L')
        else:
            target = Image.fromarray(255 * np.ones((image.size[1], image.size[0])).astype('uint8'))
            # target = Image.fromarray(np.full(image.size[:-1], void_ind).astype('uint8'), mode='L')

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target#, self.images[i]

