#!/usr/bin/env python3

import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from src.imageaugmentations import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from torch.utils.data import Dataset
import math
from omegaconf import DictConfig
import hydra

class Train_Data(Dataset):
    def __init__(self, dataset_root, data_dict, 
                 num_novel = 0,
                 label_root='./datasets/pseudo_label',
                 memory_root = None,
                 target_root=None,
                 id2trainid=None,
                 map_fun=None,
                 transform=None, 
                 num_classes = 19,
                 exp = None):
        """Load all filenames."""
        super(Train_Data, self).__init__()
        self.label_root = os.path.join(label_root, exp)
        self.target_root = target_root
        self.memory_root = memory_root

        self.transform = transform

        self.dataset_root = dataset_root
        self.data_dict = data_dict
        self.images = []
        self.targets = []
        self.map_fun = map_fun
        self.id2trainid = id2trainid
        self.num_classes = num_classes
        self.num_novel = num_novel

        for i, path in enumerate(self.data_dict['images']):
            self.images.append(os.path.join(self.dataset_root, path))
            self.targets.append(os.path.join(self.label_root, self.data_dict['targets'][i]))
   
        for i, path in enumerate(self.data_dict['memory_images']):
            self.images.append(os.path.join(self.memory_root, path))
            self.targets.append(os.path.join(self.target_root, self.data_dict['memory_targets'][i]))
            

    def __len__(self):
        """Return number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, i):
        # Load images and perform augmentations with PIL
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')

        if self.transform is not None:
            image, target = self.transform(image, target)

        if self.map_fun is not None:
            if i < len(self.data_dict['images']):
                target = self.map_fun(target, self.id2trainid, self.num_classes + self.num_novel)
            else:
                target = self.map_fun(target, self.id2trainid, self.num_classes)

        return image, target

class Trainer():

    def __init__(self, cfg: DictConfig):
        self.max_epoch = cfg.max_epoch
        self.num_novel = cfg.experiments[cfg.experiment]['k']
        self.num_classes = cfg[cfg.experiments[cfg.experiment]['model']].num_classes
        self.num_workers = cfg.num_workers
        self.bs = cfg.bs
        self.lam = cfg.lam
        self.ckpt = os.path.join(cfg.weight_dir, cfg.experiments[cfg.experiment]['init_weights'])
        self.model = cfg[cfg.experiments[cfg.experiment]['model']]
        self.layers_ext = cfg.extended_layers[cfg.experiments[cfg.experiment]['model']]
        self.layers_trainable = cfg.trainable_layers[cfg.experiments[cfg.experiment]['model']]
        self.seed = cfg.seeds[cfg.run]
        self.exp = cfg.experiment
        self.run = cfg.run
        self.save_ckpt = os.path.join(cfg.weight_dir, cfg.experiments[cfg.experiment]['init_weights'].replace('initials', 'extended'))
        if not os.path.exists(self.save_ckpt[:-8]):
            os.makedirs(self.save_ckpt[:-8])
        self.dataset = hydra.utils.instantiate(cfg[cfg.experiments[cfg.experiment]['dataset']], cfg.experiments[cfg.experiment]['split'])
        self.dataset_train = hydra.utils.instantiate(cfg[cfg.experiments[cfg.experiment]['train_dataset']], cfg.experiments[cfg.experiment]['train_split'])
        self.train_dataset = pkl.load(open(os.path.join('./datasets', cfg.experiments[cfg.experiment]['dict']),'rb'))
        self.sample_size = len(self.train_dataset['images']) + len(self.train_dataset['memory_images'])

    def loss_ce(self, inp, target, weight=None):
        b, h, w = target.shape[0], target.shape[1], target.shape[2]
        inp, target = inp.cuda(), target.cuda()
        loss = nn.NLLLoss(ignore_index=255, reduction='sum', weight=weight)
        m = nn.LogSoftmax(dim=1)
        output = loss(m(inp), target)/(b*h*w)
        return output

    def loss_d(self, logits, target, T=1):
        logits = logits[:, :-self.num_novel, :, :]
        b, c, h, w = target.shape[0], target.shape[1], target.shape[2], target.shape[3]
        m = nn.Softmax(dim=1)
        M0 = m(target/T)
        loss = torch.sum(- M0 * F.log_softmax(logits/T, 1))
        loss = T*T*loss/(b*h*w)
        return loss

    def poly_schd(self, e):
        return math.pow((1 - (e / (self.max_epoch*self.sample_size)/self.bs)), 0.9)

    def calculate_weights(self, target):
        """
        Calculate weights of the classes based on training crop (NVIDIA)
        """
        target = np.asarray(target).flatten()
        hist = np.zeros(self.num_classes + self.num_novel)
        for i in range(self.num_classes + self.num_novel):
            hist[i] = np.sum(target == i)/len(target)
        hist = hist*(1-hist)+1
        return torch.tensor(1/hist).float().cuda()


    def fulltotrain(self, target, id_to_trainid, num_classes):
        """Transforms labels from full cityscapes labelset to training label set."""
        remapped_target = target.clone()
        for k, v in id_to_trainid.items():
            if v < num_classes:
                remapped_target[target == k] = v
            else:
                remapped_target[target == k] = 255
        return remapped_target

    def prep_data(self):
        trans = Compose([RandomHorizontalFlip(), RandomCrop(1000), ToTensor(), Normalize(self.dataset_train.mean, self.dataset_train.std)])
        data = Train_Data(self.dataset.root, self.train_dataset, memory_root = self.dataset_train.root, target_root=self.dataset_train.target_root, id2trainid=self.dataset_train.id_to_trainid, map_fun=self.fulltotrain, transform=trans, num_classes = self.num_classes, num_novel = self.num_novel, exp = self.exp)
        datloader = DataLoader(data, batch_size=self.bs, shuffle=True, drop_last=True, num_workers=self.num_workers)
        return datloader

    def prep_distill_model(self):
        model = hydra.utils.instantiate(self.model, num_classes = self.num_classes)
        network0 = nn.DataParallel(model)
        network0.load_state_dict(torch.load(self.ckpt)['state_dict'], strict=False)
        for param in network0.parameters():
            param.requires_grad = False

        network0.cuda().eval()
        return network0
    
    def prep_model(self):
        ckpt = torch.load(self.ckpt)
        if 'state_dict' in ckpt.keys():
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        for layer in self.layers_ext:
            params = state_dict[layer]
            if 'weight' in layer:
                new = torch.rand((self.num_novel, params.shape[1], params.shape[2], params.shape[3]), requires_grad=True)
                new = (new * np.sqrt(2 / params.shape[1])).cuda()
                new = torch.cat((params, new), dim=0)
            elif 'bias' in layer:
                new = torch.rand(1, requires_grad=True).cuda()
                new = torch.cat((params, new), dim=0)
            del state_dict[layer]
            state_dict.update({layer: new})
    
        network = hydra.utils.instantiate(self.model, num_classes = self.num_classes+self.num_novel)
        if any('module' in key for key in state_dict.keys()):
            network = nn.DataParallel(network)
        network.load_state_dict(state_dict, strict=False)
        for param in network.parameters():
            param.requires_grad = True

        for name, param in network.named_parameters():
            if not any(s in name for s in self.layers_trainable):
                param.requires_grad = False

        network.cuda()
        network.train()

        optimizer = optim.Adam(network.parameters(), lr=5*1e-5, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.poly_schd)
        return network, optimizer, scheduler

    def train(self):

        # Start Training
        best_loss = np.inf
        datloader = self.prep_data()
        network0 = self.prep_distill_model()
        network, optimizer, scheduler = self.prep_model()


        for epoch in range(0, self.max_epoch):
            loss_avg = []
            print('epoch {}/{}'.format(epoch + 1, self.max_epoch))
            i = 0
            iters = len(datloader)
            for x, y in datloader:
                weights = self.calculate_weights(y)
                optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                out = network(x)
                out_old = network0(x)
                loss1 = self.loss_ce(out, y, weight=weights)
                loss2 = self.loss_d(out, out_old)
                loss = (1-self.lam) * loss1 + self.lam * loss2
                print(loss1, loss2)
                loss.backward()
                optimizer.step()
                loss_avg.append(loss.item())
                print('{} Loss: {}'.format(i, loss.item()))
                scheduler.step(int(epoch + i / iters))
                i += 1
            loss_avg = sum(loss_avg) / len(loss_avg)
            print('Average Loss in {}. epoch: {}'.format(epoch+1, loss_avg))
            if loss_avg < best_loss:
                best_loss = loss_avg
                print('Saving checkpoint', self.save_ckpt)
                torch.save({'state_dict': network.state_dict()}, self.save_ckpt)


def extend_model(cfg):
    tr = Trainer(cfg)
    tr.train()
    


