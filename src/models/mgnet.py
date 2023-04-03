import torch
import torch.nn as nn
import numpy as np
import csv

from utils.convolutions import conv1x1, conv3x3, convkxk
# from utils.eigenvalpues import calc_ev


from base.base_mgnet import BaseMgBlock


class MgBlock_pre(BaseMgBlock):

    """ Mg Block with predefined filters in Restriction
    :argument

     filters: optional, if true weights of restriction and projection are set to

                        [[0.25, 0.5, 0.25]
                 0.25 *  [0.5,  1.0,  0.5]
                         [0.25, 0.5, 0.25]]

    train_filters: optional, if true preset weights (filters) are updated
    """
    def __init__(self, in_channels: int, out_channels: int, num_smoothings: int,
                 num_layer: int, device, filters: bool, train_filters: bool):
        super(MgBlock_pre, self).__init__(in_channels, out_channels, num_smoothings, num_layer, device)

        self.convR = conv3x3(in_channels, out_channels, stride=2, groups=in_channels)
        self.convI = conv3x3(in_channels, out_channels, stride=2, groups=in_channels)
        self.num_layer = num_layer

    def smoothing_step(self, f, u0, i):

        key1 = 'bn' + str(2 * i)
        key2 = 'bn' + str(2 * i + 1)

        if self.Bblock['shared']:
            keyB = 'B0'
        else:
            keyB = 'B' + str(i)

        # self.convB = self.convBs[keyB]

        self.bn1 = self.batchnorms[key1]
        self.bn2 = self.batchnorms[key2]

        u = f - self.Ablock(u0)

        self.A_weights.append(self.Ablock.weight)
        u = self.bn1(u)
        u = self.activation(u)
        u *= self.convBs

        #todo: check batchnorm
        u = self.bn2(u)
        # u = self.relu(u)

        return u + u0, self.A_weights

    def smoothing(self, f, u):
        for i in range(self.num_smoothings):
            u, self.A_weights = self.smoothing_step(f, u, i)
            # u, self.A_weights = self.smoothing_step_withev(f, u, i)
        return u, self.A_weights

    def forward(self, x):

        f = x[0]
        u = x[1]

        self.A_weights = []
        # u = torch.zeros(f.shape, device=self.device)

        if self.num_layer > 1:
            u = self.convI(u)
            f = self.convR(f) + self.Ablock(u)
            self.A_weights.append(self.Ablock.weight)
            # u = torch.zeros(f.shape, device=self.device)

        u, self.A_weights = self.smoothing(f, u)

        # self.checkweights()

        v = self.Ablock(u)
        f = f - v
        f = self.activation(f)

        return f, u


class MgBlock_down(BaseMgBlock):

    """ Mg Block with predefined filters in Restriction
    :argument

     filters: optional, if true weights of restriction and projection are set to

                        [[0.25, 0.5, 0.25]
                 0.25 *  [0.5,  1.0,  0.5]
                         [0.25, 0.5, 0.25]]

    train_filters: optional, if true preset weights (filters) are updated
    """
    def __init__(self, in_channels: int, out_channels: int, num_smoothings: int,
                 num_layer: int, device, filters: bool, train_filters: bool, Ablock_args, Bblock_args):
        super(MgBlock_down, self).__init__(in_channels, out_channels, num_smoothings, num_layer, device)

        self.convR = conv3x3(in_channels, out_channels, stride=2, groups=in_channels)
        self.convI = conv3x3(in_channels, out_channels, stride=2, groups=in_channels)

    def forward(self, x):
        f = x[0]
        u0 = x[1]

        self.A_weights = []
        if self.num_layer > 1:
            u0 = self.convI(u0)
            f = self.convR(f) + self.Ablock(u0)
            self.A_weights.append(self.Ablock.weight)
            # u = torch.zeros(f.shape, device=self.device)

        u, self.A_weights = self.smoothing(f, u0)

        # self.checkweights()

        v = self.Ablock(u)
        f = f - v
        f = self.activation(f)

        return f, u, u0


class MgBlock_poly(BaseMgBlock):

    """ Mg Block with predefined filters in Restriction
    :argument

     filters: optional, if true weights of restriction and projection are set to

                        [[0.25, 0.5, 0.25]
                 0.25 *  [0.5,  1.0,  0.5]
                         [0.25, 0.5, 0.25]]

    train_filters: optional, if true preset weights (filters) are updated
    """
    def __init__(self, in_channels: int, out_channels: int, num_smoothings: int,
                 num_layer: int, device, filters: bool, train_filters: bool, Ablock_args, Bblock_args):
        super(MgBlock_poly, self).__init__(in_channels, out_channels, num_smoothings, num_layer, device, Ablock_args, Bblock_args)

        self.convR = convkxk(in_channels, out_channels, stride=2, groups=in_channels, k=3)
        self.convI = convkxk(in_channels, out_channels, stride=2, groups=in_channels, k=3)

        if not bool(Ablock_args['shared']):
            self.Ablock.A = convkxk(out_channels, out_channels, Ablock_args['groups'], k=3)

    def poly_nomial1(self, u):
        # highest degree
        deg_poly = self.Bblock_args['degree']
        alpha = self.Bblock

        W = (torch.mul(u, alpha[deg_poly])).to(self.device)
        y = self.Ablock.A(W)

        # 2nd highest degree
        for i in range(deg_poly - 1, 0, -1):
            R = torch.mul(u, alpha[i]).to(self.device)
            y = torch.add(y, R).to(self.device)
            y = self.Ablock.A(y).to(self.device)

            # constant coefficient
        R = torch.mul(u, alpha[0]).to(self.device)
        y = torch.add(y, R).to(self.device)
        #y=self.poly_nomial2(y).to(device)

        return y

    def smoothing_step(self, f, u0, i):

        key1, key2 = self.bn_keys(i)

        bn1 = self.batchnorms[key1]
        bn2 = self.batchnorms[key2]

        A, _ = self.get_operations(i)

        # u=self.poly_nomial2(u0)
        u = f - A(u0)

        # self.A_weights.append(self.convA.weight)

        u = bn1(u)
        u = self.activation(u)

        # -----------------------------------------------------------------
        # Adding polynomial function in smoothing operation
        # idx1 = torch.randperm(u.size(dim=1))

        # u = u[:, idx1, :, :]
        u = self.poly_nomial1(u)

        u = bn2(u)
        u = self.activation(u)

        return u + u0, self.A_weights

    def smoothing(self, f, u):
        for i in range(self.num_smoothings):
            u, self.A_weights = self.smoothing_step(f, u, i)
        return u, self.A_weights

    def forward(self, x):

        f = x[0]
        u = x[1]

        self.A_weights = []
        if self.num_layer > 1:
            u = self.convI(u)
            f = self.convR(f) + self.Ablock.A(u)
            self.A_weights.append(self.Ablock.A.weight)

        u, self.A_weights = self.smoothing(f, u)

        # self.checkweights()

        v = self.Ablock.A(u)
        f = f - v
        f = self.activation(f)

        return f, u


class MgBlock_dw(BaseMgBlock):

    """ Mg Block with predefined filters in Restriction
    :argument

     filters: optional, if true weights of restriction and projection are set to

                        [[0.25, 0.5, 0.25]
                 0.25 *  [0.5,  1.0,  0.5]
                         [0.25, 0.5, 0.25]]

    train_filters: optional, if true preset weights (filters) are updated
    """
    def __init__(self, in_channels: int, out_channels: int, num_smoothings: int,
                 num_layer: int, device, filters: bool, train_filters: bool, Ablock_args, Bblock_args):
        super(MgBlock_dw, self).__init__(in_channels, out_channels, num_smoothings, num_layer, device, Ablock_args, Bblock_args)

        self.convR = convkxk(in_channels, out_channels, stride=2, groups=in_channels, k=3)
        self.convI = convkxk(in_channels, out_channels, stride=2, groups=in_channels, k=3)

        if not bool(Ablock_args['shared']):
            self.Ablock.A = convkxk(out_channels, out_channels, Ablock_args['groups'], k=3)

    def forward(self, x):

        f = x[0]
        u = x[1]

        self.A_weights = []
        if self.num_layer > 1:
            u = self.convI(u)
            f = self.convR(f) + self.Ablock.A(u)
            self.A_weights.append(self.Ablock.A.weight)
            # u = torch.zeros(f.shape, device=self.device)
        #elif self.num_layer == 0:
        #     u = self.convI(u)

        u, self.A_weights = self.smoothing(f, u)

        # self.checkweights()

        v = self.Ablock.A(u)
        f = f - v
        f = self.activation(f)

        return f, u


class CNN_block(nn.Module):
    def __init__(self, channels):
        super(CNN_block, self).__init__()
        #self.activation = nn.Tanh()
        # self.activation = nn.LeakyReLU()
        self.relu = nn.ReLU()

        '''CNN Block '''
        self.conv1 = conv1x1(channels, channels, groupsize=channels)
        self.conv2 = conv3x3(channels, channels, stride=1, groups=channels)
        self.conv3 = conv1x1(channels, channels, groupsize=channels)

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """
            x_l+1 = x_l + K_l3 σ(N (K_l2 (σ(N (K_l1  σ(N (x_l ))))
        """
        x0 = x
        x = self.activation(self.bn1(x))
        x = self.conv1(x)
        x = self.activation(self.bn2(x))
        x = self.conv2(x)
        x = self.activation(self.bn3(x))
        x = self.conv3(x)
        x += x0
        return x


class MgBlock_I0(BaseMgBlock):
#todo: check if is working
    def __init__(self, in_channels, out_channels, smoothings: int, num_layer: int, device):
        super(MgBlock_I0, self).__init__(in_channels, out_channels, smoothings, num_layer, device)
        self.convR = conv3x3(in_channels, out_channels, stride=2, groups=in_channels)  # depthwise convolution

    def forward(self, x):

        f = x[0]
        u = x[1]

        self.A_weights = []
        # u = torch.zeros(f.shape, device=self.device)

        if self.num_layer > 1:
            f = self.convR(f)
            self.A_weights.append(self.Ablock.weight)
            u = torch.zeros(f.shape, device=self.device)

        if torch.sum(u) != 0:
            print('u is not empty')

        u, self.A_weights = self.smoothing(f, u)
        # u = self.quat_smoothing(f, u)
        # self.checkweights()

        v = self.Ablock(u)
        f -= v
        f = self.activation(f)

        return f, u


class MgNet(nn.Module):
    def __init__(self, block, device, num_classes, in_channels, out_channels, smoothings, filters, train_filters, Ablock_args, Bblock_args):
        super(MgNet, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.panel = in_channels[0]
        max_layer = len(in_channels)

        #self.relu = nn.ReLU()
        # self.activation = nn.Tanh()
        self.activation = nn.ReLU()
        # self.activation = nn.LeakyReLU()

        self.bn = nn.BatchNorm2d(self.panel)
        #self.conv1 = nn.Conv2d(1, self.panel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
        #                        bias=False)
        self.conv1 = nn.Conv2d(3, self.panel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                              bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[max_layer-1], self.num_classes)
        # self.fc = nn.Linear(out_channels[3], self.num_classes)
        # self.fc = nn.Linear(out_channels[2], 10)

        self.mg_block1 = self.make_mg_block(block, in_channels[0], out_channels[0], smoothings[0], 1, self.device, filters, train_filters, Ablock_args, Bblock_args)
        self.mg_block2 = self.make_mg_block(block, in_channels[1], out_channels[1], smoothings[1], 2, self.device, filters, train_filters, Ablock_args, Bblock_args)
        self.mg_block3 = self.make_mg_block(block, in_channels[2], out_channels[2], smoothings[2], 3, self.device, filters, train_filters, Ablock_args, Bblock_args)
        self.mg_block4 = self.make_mg_block(block, in_channels[3], out_channels[3], smoothings[3], 4, self.device, filters, train_filters, Ablock_args, Bblock_args)
        # --------
        # self.mg_block5 = self.make_mg_block(block, in_channels[4], out_channels[4], smoothings[3], 5, self.device,
        #                                     filters, train_filters, Ablock_args, Bblock_args)
        # self.mg_block6 = self.make_mg_block(block, in_channels[5], out_channels[5], smoothings[3], 6, self.device,
        #                                     filters, train_filters, Ablock_args, Bblock_args)

        # self.mg_block7 = self.make_mg_block(block, in_channels[6], out_channels[6], smoothings[3], 7, self.device,
        #                                     filters, train_filters, Ablock_args, Bblock_args)
        # self.mg_block8 = self.make_mg_block(block, in_channels[7], out_channels[7], smoothings[3], 8, self.device,
        #                                     filters, train_filters, Ablock_args, Bblock_args)

    def make_mg_block(self, block, in_channels, out_channels, smoothings, num_layer, device, filters, train_filters,  Ablock_args, Bblock_args):

        # return MgBlock_I0(in_channels, out_channels, smoothings, num_layer, device)
        #      MgBlock_dw(in_channels, out_channels, smoothings, num_layer, device, kernel)
        return block(in_channels, out_channels, smoothings, num_layer, device, filters, train_filters, Ablock_args, Bblock_args)

    def forward(self, x):

        out = self.conv1(x)
        out = self.activation(out)

        #---------------------------------------

        u0 = torch.zeros(out.shape, device=self.device)

        x = (out, u0)
        out, u = self.mg_block1(x)

        x = (out, u)
        out, u = self.mg_block2(x)

        x = (out, u)
        out, u = self.mg_block3(x)

        x = (out, u)
        out, u = self.mg_block4(x)

        #------

        # x = (out, u)
        # out, u = self.mg_block5(x)
        #
        # x = (out, u)
        # out, u = self.mg_block6(x)

        # x = (out, u)
        # out, u = self.mg_block7(x)
        #
        # x = (out, u)
        # out, u = self.mg_block8(x)
        # #------

        out = self.avg_pool(u)
        out = torch.flatten(out, 1)
        #out = self.fc(out)

        return out, u0