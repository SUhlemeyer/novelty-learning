import torch
import torch.nn as nn
import csv
import pandas as pd

from utils.convolutions import conv3x3, convkxk
from time import time
# from LFA.LFA import LFA

import numpy as np

class BaseMgBlock(nn.Module):
    """
    Module that provides the basic functions
        :arg
        in_channels: number of input channels
        out_channels: number of output channels
        num_smoothings: number of smoothing steps on each resolution, usually the same amount on each resolution
        num_layer: layers are enumerated, due to implementation details # todo: tbc
        device: number of GPU
    """

    def __init__(self, in_channels, out_channels, num_smoothings: int, num_layer: int, device, Ablock_args, Bblock_args):
        super(BaseMgBlock, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_smoothings = num_smoothings
        self.num_layer = num_layer
        self.A_weights = []
        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU()

        self.Ablock_args = Ablock_args
        self.Bblock_args = Bblock_args

        # block in block_args = [richardson, block, scalar, polynomial]
        #
        # self.B = 'block'
        # self.scalar = False
        # self.A.weights = self.define_kernel()

        # todo: include MGIC

        # self.Bblock = self.make_block('B', out_channels, self.num_smoothings, **Bblock_args)
        # self.Ablock = self.make_block('A', out_channels, self.num_smoothings, **Ablock_args)

        self.batchnorms = self.make_batchnorms(self.num_smoothings, out_channels)

    # def calc_preconditioning_factor(self):
    #     E = LFA(self.convA, self.A_groups, 32 // (self.num_layer + 1) + 1)
    #     emin, emax = np.real(E.min()), np.real(E.max())
    #     return 2 / (emin + emax)
        # self.convB = conv3x3(out_channels, out_channels)

    def define_kernel(self):
        filter = []

        block = [[0.25 * 0.25, 0.5 * 0.25, 0.25 * 0.25],
                 [0.5 * 0.25, 1.0 * 0.25, 0.5 * 0.25],
                 [0.25 * 0.25, 0.5 * 0.25, 0.25 * 0.25]]

        for ch in range(self.out_channels):
            filter.append(block)
        tensor = nn.Parameter(torch.tensor(filter))

        return tensor.reshape(self.out_channels, 1, 3, 3)

    def check_weights(self):
        if (not torch.equal(self.A_weights[0], self.A_weights[1])
                or (not torch.equal(self.A_weights[0], self.A_weights[1])
                    and not torch.equal(self.A_weights[0], self.A_weights[2])
                    and not torch.equal(self.A_weights[1], self.A_weights[2]))):
            raise ValueError('Sorry, weights of convolution A are not the same!')

    def keys(self, i):

        if self.Bblock_args['shared']:
            keyB = 'B'
        else:
            keyB = 'B' + str(i)

        if self.Ablock_args['shared']:
            keyA = 'A'
        else:
            keyA = 'A' + str(i)

        return keyA, keyB

    def bn_keys(self, i, s):

        key1 = 'bn' + str(2 * i * s)
        key2 = 'bn' + str(2 * i * s + 1)

        return key1, key2

    # def bn_keys(self, i):
    #
    #     key1 = 'bn' + str(2 * i)
    #     key2 = 'bn' + str(2 * i + 1)
    #
    #     return key1, key2

    def _get_smoothing_batchnorms(self, i, v, s, postsmoothing: bool):
    # def _get_smoothing_batchnorms(self, i, postsmoothing: bool):
        """
        returns batchnormalization for smoothing operation based in level index
        if postsmoothing:
            new batchnorm operations
            """
        # num = self.num_pre_smoothings
        # idx = num ** i if i != 0 else 0

        # key1, key2 = self.bn_keys(i, s)

        if self._get_name() in ('MgBlockMGIC'):
            key1, key2 = self.bn_keys(i, s)
            if postsmoothing:
                bn1, bn2 = self.Vcycle_ops[str(v)]['post_batchnorms'][key1], self.Vcycle_ops[str(v)]['post_batchnorms'][key2]
            else:
                bn1, bn2 = self.Vcycle_ops[str(v)]['pre_batchnorms'][key1], self.Vcycle_ops[str(v)]['pre_batchnorms'][key2]

            # if postsmoothing:
            #     bn1, bn2 = self.post_batchnorms[key1], self.post_batchnorms[key2]
            # else:
            #     bn1, bn2 = self.pre_batchnorms[key1], self.pre_batchnorms[key2]
        else:
            key1, key2 = self.bn_keys(i, 1)
            bn1, bn2 = self.batchnorms[key1], self.batchnorms[key2]

        return bn1, bn2

    def get_operations(self, i):
        if self.Bblock_args['block'] == 'polynomial':
            keyA, _ = self.keys(i)
            return self.Ablock[keyA], self.Bblock

        else:
            keyA, keyB = self.keys(i)
            return self.Ablock[keyA], self.Bblock[keyB]

    def smoothing_step(self, f, u0, i):

       # bn1, bn2 = self._get_smoothing_batchnorms(i, None,  False)
        bn1, bn2 = self._get_smoothing_batchnorms(i, False)
        # key1, key2 = self.bn_keys(i)
        # bn1 = self.batchnorms[key1]
        # bn2 = self.batchnorms[key2]

        A, B = self.get_operations(i)

        r = f - A(u0)

        r = bn1(r)
        r = self.relu(r)
        u = B(r)
        u = bn2(u)
        u = self.relu(u)

        return u + u0, self.A_weights

    def smoothing(self, f, u):
        for i in range(self.num_smoothings):
            u, self.A_weights = self.smoothing_step(f, u, i)
        return u, self.A_weights

    @staticmethod
    def make_batchnorms(num_smoothings: int, out_channels: int) -> {}:
        keys = []
        for s in range(2 * num_smoothings):
            keys.append('bn' + str(s))

        return nn.ModuleDict({key: nn.BatchNorm2d(out_channels) for key in keys})

    def groups(self, **kwargs):
        groups_size = kwargs['groups_size']

        if not bool(groups_size):
            return kwargs['groups']
        else:
            return self.out_channels//int(groups_size)

    def make_block(self, name, out_channels, smoothings, **kwargs):
        """
        designs different blocks for operations A and B
            polynomial of degree n
            richardson: trainable or non-trainable scalar, initial value derived from LFA
            shared: module reused
        """

        block = kwargs['block']
        shared = bool(kwargs['shared'])
        k = int(kwargs['k'])

        groups = self.groups(**kwargs)

        if block == 'polynomial':
            return self.make_polynomial(**kwargs)

        elif block == 'MGIC':
            return None

        # elif block == 'richardson':
        #     #todo: fix it
        #     E = LFA(self.A, groups, 32 // (self.num_layer + 1) + 1)
        #     emin, emax = E.min(), E.max()
        #     if self.scalar:
        #         return 2 / (emin + emax)
        #     else:
        #         return torch.nn.Parameter(torch.tensor(2 / (emin + emax)))

        if shared:
            return nn.ModuleDict({name: convkxk(out_channels, out_channels, groups, k)})

        keys = []
        for s in range(smoothings):
            keys.append(name + str(s))
        return nn.ModuleDict({key: convkxk(out_channels, out_channels, groups, k) for key in keys})

    #todo: add polynomial block
    def make_polynomial(self, **kwargs):
        deg_poly = kwargs['degree']

        # deg_poly = 2
        print("Degree of polynomial is :", deg_poly)

        init = torch.empty(deg_poly + 1).normal_(mean=0, std=1)
        alpha = torch.nn.Parameter(init, requires_grad=True)

        print("values of alphas are :", alpha)
        print("No of co-efficients are:", torch.numel(alpha))

        return alpha

    def forward(self, x):
        pass