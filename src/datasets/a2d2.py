import torch.utils.data as data

import numpy as np
from PIL import Image
import os

import logging
from src.datasets.cityscapes import fulltotrain as cs_fulltotrain


class Label(object):
    def __init__(self, name, id, trainid, hex_color, color):
        self.name = name
        self.id = id
        self.trainid = trainid
        self.hex_color = hex_color
        self.color = color

    def __call__(self):
        print('name: {}\nid: {}\ntrainid: {}\nhex_color:{}\ncolor:{}'.format(self.name,
                                                                             self.id,
                                                                             self.trainid,
                                                                             self.hex_color,
                                                                             str(self.color)))


void_ind = 255
cityscapes_void = void_ind
# mean = (0.485, 0.456, 0.406)    # values from github repo where the model originates from
# std = (0.229, 0.224, 0.225)

labels = [
    #       name                    id          trainId           hex_color                  color
    Label('Animals',                0,             0,             '#ccff99',            (204, 255, 153)),
    Label('Bicycle',                1,             1,             '#b65906',            (182, 89, 6)),
    Label('Bicycle',                2,             1,             '#963204',            (150, 50, 4)),
    Label('Bicycle',                3,             1,             '#5a1e01',            (90, 30, 1)),
    Label('Bicycle',                4,             1,             '#5a1e1e',            (90, 30, 30)),
    Label('Blurred area',           5,             2,             '#60458f',            (96, 69, 143)),
    Label('Buildings',              6,             3,             '#f1e6ff',            (241, 230, 255)),
    Label('Car',                    7,             4,             '#ff0000',            (255, 0, 0)),
    Label('Car',                    8,             4,             '#c80000',            (200, 0, 0)),
    Label('Car',                    9,             4,             '#960000',            (150, 0, 0)),
    Label('Car',                    10,            4,            '#800000',             (128, 0, 0)),
    Label('Curbstone',              11,            5,            '#808000',             (128, 128, 0)),
    Label('Dashed line',            12,            6,            '#8000ff',             (128, 0, 255)),
    Label('Drivable cobblestone',   13,            7,            '#b432b4',             (180, 50, 180)),
    Label('Ego car',                14,            8,            '#48d1cc',             (72, 209, 204)),
    Label('Electronic traffic',     15,            9,            '#ff46b9',             (255, 70, 185)),
    Label('Grid structure',         16,            10,            '#eea2ad',            (238, 162, 173)),
    Label('Irrelevant signs',       17,            11,            '#400040',            (64, 0, 64)),
    Label('Nature object',          18,            12,            '#93fdc2',            (147, 253, 194)),
    Label('Non-drivable street',    19,            13,            '#8b636c',            (139, 99, 108)),
    Label('Obstacles / trash',      20,            14,            '#ff0080',            (255, 0, 128)),
    Label('Painted driv. instr.',   21,            15,            '#c87dd2',            (200, 125, 210)),
    Label('Parking area',           22,            16,            '#9696c8',            (150, 150, 200)),
    Label('Pedestrian',             23,            17,            '#cc99ff',            (204, 153, 255)),
    Label('Pedestrian',             24,            17,            '#bd499b',            (189, 73, 155)),
    Label('Pedestrian',             25,            17,            '#ef59bf',            (239, 89, 191)),
    Label('Poles',                  26,            18,            '#fff68f',            (255, 246, 143)),
    Label('RD normal street',       27,            19,            '#ff00ff',            (255, 0, 255)),
    Label('RD restricted area',     28,            20,            '#960096',            (150, 0, 150)),
    Label('Rain dirt',              29,            21,            '#352e52',            (53, 46, 82)),
    Label('Road blocks',            30,            22,            '#b97a57',            (185, 122, 87)),
    Label('Sidebars',               31,            23,            '#e96400',            (233, 100, 0)),
    Label('Sidewalk',               32,            24,            '#b496c8',            (180, 150, 200)),
    Label('Signal corpus',          33,            25,            '#212cb1',            (33, 44, 177)),
    Label('Sky',                    34,            26,            '#87ceff',            (135, 206, 255)),
    Label('Slow drive area',        35,            27,            '#eee9bf',            (238, 233, 191)),
    Label('Small vehicles',         36,            28,            '#00ff00',            (0, 255, 0)),
    Label('Small vehicles',         37,            28,            '#00c800',            (0, 200, 0)),
    Label('Small vehicles',         38,            28,            '#009600',            (0, 150, 0)),
    Label('Solid line',             39,            29,            '#ffc125',            (255, 193, 37)),
    Label('Speed bumper',           40,            30,            '#6e6e00',            (110, 110, 0)),
    Label('Tractor',                41,            31,            '#000064',            (0, 0, 100)),
    Label('Traffic guide obj.',     42,            32,            '#9f79ee',            (159, 121, 238)),
    Label('Traffic sign',           43,            33,            '#00ffff',            (0, 255, 255)),
    Label('Traffic sign',           44,            33,            '#1edcdc',            (30, 220, 220)),
    Label('Traffic sign',           45,            33,            '#3c9dc7',            (60, 157, 199)),
    Label('Traffic signal',         46,            25,            '#0080ff',            (0, 128, 255)),
    Label('Traffic signal',         47,            25,            '#1e1c9e',            (30, 28, 158)),
    Label('Traffic signal',         48,            25,            '#3c1c64',            (60, 28, 100)),
    Label('Truck',                  49,            34,            '#ff8000',            (255, 128, 0)),
    Label('Truck',                  50,            34,            '#c88000',            (200, 128, 0)),
    Label('Truck',                  51,            34,            '#968000',            (150, 128, 0)),
    Label('Utility vehicle',        52,            35,            '#ffff00',            (255, 255, 0)),
    Label('Utility vehicle',        53,            35,            '#ffffc8',            (255, 255, 200)),
    Label('Zebra crossing',         54,            36,            '#d23273',            (210, 50, 115)),
    Label('Void',                   void_ind,      void_ind,      '#000000',            (0, 0, 0))
]

a2d2_to_cityscapes = {
    0: 5,
    1: 33,
    2: 33,
    3: 33,
    4: 33,
    5: cityscapes_void,
    6: 11,
    7: 26,
    8: 26,
    9: 26,
    10: 26,
    11: 8,
    12: 7,
    13: 7,
    14: 1,
    15: cityscapes_void,
    16: 13,
    17: cityscapes_void,
    18: 21,
    19: cityscapes_void,
    20: cityscapes_void,
    21: 7,
    22: 9,
    23: 24,
    24: 24,
    25: 24,
    26: 17,
    27: 7,
    28: 7,
    29: cityscapes_void,
    30: 34,
    31: 20,
    32: 8,
    33: 19,
    34: 23,
    35: 7,
    36: 32,
    37: 32,
    38: 32,
    39: 7,
    40: 7,
    41: cityscapes_void,
    42: 20,
    43: 20,
    44: 20,
    45: 20,
    46: 19,
    47: 19,
    48: 19,
    49: 27,
    50: 27,
    51: 27,
    52: 30,
    53: 30,
    54: 7,
    void_ind: 0
    }

id_to_trainid = {label.id: (label.trainid if (label.id < 200) else 18) for label in labels}
id_to_color = {label.id: (label.color if (label.id < 200) else (255, 102, 0  )) for label in labels}
id_to_hexcolor = {label.id: label.hex_color for label in labels}
id_to_name = {label.id: label.name for label in labels}
color_to_trainid = {label.color: label.trainid for label in labels}
color_to_id = {label.color: label.id for label in labels}
color_to_name = {label.color: label.name for label in labels}
trainid_to_name = {label.trainid: label.name for label in labels}
trainid_to_id = {label.trainid: (a2d2_to_cityscapes[label.id] if (label.trainid != void_ind) else 0) for label in labels}
trainid_to_color = {label.trainid: label.color for label in labels}
trainid_to_rgba = {label.trainid: tuple(i / 255.0 for i in label.color) + (1.0,) for label in labels}
name_to_rgba = {label.name: tuple(i / 255.0 for i in label.color) + (1.0, ) for label in labels}


discover_mapping = {label.id: (trainid_to_name[label.trainid], trainid_to_color[label.trainid])
                    for label in labels}


def hex_to_rgb(hex_value):
    hex_value = hex_value.lstrip('#')
    lv = len(hex_value)
    return tuple(int(hex_value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def fulltotrain(target):
    """Transforms labels from full A2D2 labelset to training label set."""
    remapped_target = target.clone()
    for k, v in id_to_trainid.items():
        remapped_target[target == k] = v
    return remapped_target

def a2d2tocityscapes(target):
    remapped_target = target.clone()
    for k, v in a2d2_to_cityscapes.items():
        remapped_target[target == k] = v
    return remapped_target



class A2D2(data.Dataset):
    def __init__(self, split='test',
                 root='/home/uhlemeyer/A2D2',
                 transform=None,
                 label_map_fun=None,
                 pre_computed_labels=False,
                 label_mapping=None,
                 id_to_trainid = id_to_trainid,
                 id_to_color = id_to_color,
                 trainid_to_id = trainid_to_id,
                 trainid_to_color = trainid_to_color):
        super(A2D2).__init__()
        if label_mapping is None:
            label_mapping = discover_mapping
        self.log = logging.getLogger(__name__ + '.A2D2')

        self.root = root
        self.split = split
        self.target_root = os.path.join(self.root.replace('datasets', 'uhlemeyer'), 'labels_id', self.split)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        if not os.path.isdir(self.root):
            self.log.error('"{}" does not exist!'.format(self.root))
            raise ValueError('"{}" does not exist!'.format(self.root))


        self.label_map_fun = label_map_fun
        self.transform = transform
        self.pre_computed_labels = pre_computed_labels
        self.label_mapping = label_mapping
        self.id_to_trainid = id_to_trainid
        self.id_to_color = id_to_color
        self.trainid_to_id = trainid_to_id
        self.trainid_to_color = trainid_to_color
        self.images = []
        self.targets = []

        for root, _, filenames in os.walk(os.path.join(self.root, 'images',
                                                       self.split)):
            for filename in filenames:
                if os.path.splitext(filename)[1] == '.png':
                    self.images.append(os.path.join(root, filename))

                    if self.split in ['train', 'val']:
                        # target_root = os.path.join(self.root,
                        #                            'labels_id', self.split)
                        self.targets.append(os.path.join(self.target_root,
                                                         filename))

        # create color matrix for indexing the class ids
        colors = np.array([label.color for label in labels])
        self.color_mat = np.zeros((256, 256, 256), dtype=np.uint8)
        self.color_mat[colors[:, 0], colors[:, 1], colors[:, 2]] = np.array([label.id for label in labels],
                                                                            dtype=np.uint8)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        if self.split in ['train', 'val']: 
            target = Image.open(self.targets[index]).convert('L')
        else:
            target = Image.fromarray(255*np.ones((image.size[1], image.size[0])).astype('uint8'))

        if self.transform is not None:
            image, target = self.transform(image, target)

        if self.label_map_fun is not None:
            target = cs_fulltotrain(target)

        return image, target, self.images[index]

    def __len__(self):
        return len(self.images)
