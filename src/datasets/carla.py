from collections import namedtuple
from pathlib import Path

import numpy as np
from PIL import Image
import os

from torch.utils.data import Dataset


class CARLA(Dataset):
    Label = namedtuple('Label', ['name', 'id', 'train_id', 'color'])
    labels = [
        Label('unlabeled', 0, 255, ( 0, 0, 0)),
        Label('building', 1, 0, ( 70, 70, 70)),
        Label('fence', 2, 1, (100, 40, 40)),
        Label('other', 3, 255, ( 55, 90, 80)),
        Label('pedestrian', 4, 2, (220, 20, 60)),
        Label('pole', 5, 3, (153, 153, 153)),
        Label('road line', 6, 4, (128, 64, 128)),
        Label('road', 7, 4, (128, 64, 128)),
        Label('sidewalk', 8, 5, (244, 35, 232)),
        Label('vegetation', 9, 6, (107, 142, 35)),
        Label('vehicle', 10, 7, ( 0, 0, 142)),
        Label('wall', 11, 8, (102, 102, 156)),
        Label('traffic sign', 12, 9, (220, 220, 0)),
        Label('sky', 13, 10, ( 70, 130, 180)),
        Label('ground', 14, 255, ( 81, 0, 81)),
        Label('bridge', 15, 255, (150, 100, 100)),
        Label('rail track', 16, 255, (230, 150, 140)),
        Label('guard rail', 17, 255, (180, 165, 180)),
        Label('traffic light', 18, 11, (250, 170, 30)),
        Label('static', 19, 255, (110, 190, 160)),
        Label('dynamic', 20, 255, (170, 120, 50)),
        Label('water', 21, 255, ( 45, 60, 150)),
        Label('terrain', 22, 12, (145, 170, 100)),
    ]

    ignore_label = 255
    num_classes = 13 # TODO: automate

    def __init__(self, root='/home/uhlemeyer/data/CARLAsmoothed', split='train', transform=None):

        super(CARLA, self).__init__()

        self.root = root
        self.split = split
        self.transform = transform
        self.mean = (0.5371, 0.5335, 0.4979)
        self.std = (0.2539, 0.2369, 0.2394)

        self.id_to_trainid = {label.id : label.train_id for label in self.labels}
        self.trainid_to_name = {label.train_id : label.name for label in self.labels}
        self.trainid_to_id = {label.train_id : label.id for label in self.labels}

        self.color_mapping = {label.train_id : label.color for label in reversed(self.labels)}

        self.images = []
        self.targets = []

        for town in os.listdir(os.path.join(self.root, self.split)):
            town_dir = Path(os.path.join(self.root, self.split, town))
            for image_file in sorted(town_dir.glob('scene_*/01_cam/*.png')):
                self.images.append(str(image_file))
                target_file = image_file.parents[1] / '02_semseg_raw' / image_file.name
                self.targets.append(str(target_file))

    def __len__(self):
        """Return number of images in the datasets."""
        return len(self.images)

    def __getitem__(self, i):
        image = Image.open(self.images[i]).convert('RGB')
        if self.split in ['train', 'val']:
            target = Image.open(self.targets[i]).convert('L')
        else:
            target = Image.fromarray(255 * np.ones((image.size[1], image.size[0])).astype('uint8'))
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target
