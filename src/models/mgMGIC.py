import torch
import torch.nn as nn
import numpy as np

from easydict import EasyDict as edict

from utils.convolutions import conv1x1, trans_conv1x1, convkxk
from base.base_mgnet import BaseMgBlock


class BaseMgMGIC(BaseMgBlock):
    def __init__(self, in_channels: int, out_channels: int, num_smoothings: int,
                 num_layer: int, device, filters: bool, train_filters: bool, Ablock_args, Bblock_args):
        super(BaseMgMGIC, self).__init__(in_channels, out_channels, num_smoothings, num_layer, device, Ablock_args,
                                         Bblock_args)

        # non-linear mg
        self.linear = False  # todo: put in configs

        # restriction can be initialized  # todo: put in configs
        self.R_init = False

        ''' Resolution level transfer operations'''

        self.convR = convkxk(in_channels, out_channels, stride=2, groups=in_channels, k=3)
        self.convI = convkxk(in_channels, out_channels, stride=2, groups=in_channels, k=3)

        ''' MGIC parameters  and Operations'''
        # groups-size for restriction and interpolation
        # Todo: put in configs
        self.R_groups_size = 2

        # number of pre-and post-smoothing
        self.num_pre_smoothings = 1
        self.num_post_smoothings = 3
        self.groups_size = int(self.Ablock_args['groups_size'])
        g_c = 64
        self.inchannel_levels = int(np.log2(out_channels // g_c)) + 1
        print(self.inchannel_levels)

        # coarsest grid g_c : number of channels on coarsest level
        if self.in_channels % 2 != 0:
            raise ValueError('Number of in channels must be divisible by two.')

    def make_inchannelblock(self, name, channels, groups_size):
        """
        args:
            name: R, P, A, B, defines operations
            v: number of current smoothing step
            channels: number of channels, input = output channels,
            groups_size: number of channel in one group

        Defines operations for in channel operations R, P, A, B
            1. channels_lst: storage for number of channels, which is halved on every level
            2. module dicts with convolutions:
                restriction R: (channels, channels//2, groups = channels // groups_size, kernel_size = 1)
                prolongation P: (channels //2, channels, groups = channels // groups_size, kernel_size = 1)
                smoothing operations A, B
                if not on coarsest level: (channels, channels, groups = channels // groups_size, kernel_size = 3)
                coarsest level: (channels, channels, groups = 1, kernel_size = 3)

        """
        keys = []
        channels_lst = []

        [keys.append(name + str(j)) for j in range(self.inchannel_levels)]
        [channels_lst.append(channels // (2 ** j)) for j in range(self.inchannel_levels)]

        # if v == self.num_smoothings:
        if name in ('R', 'I'):
            op = nn.ModuleDict(
                    {key: conv1x1(channels_lst[idx], channels_lst[idx] // 2, channels_lst[idx] // groups_size)
                     for idx, key in enumerate(keys[: -1])})
            if self.Ablock_args['R_init']:
                print('R is initialized')
                for r in op:
                    op[r].weight = nn.Parameter(op[r].weight / torch.sum(op[r].weight, 0))
            return op

        elif name == 'P':
            return nn.ModuleDict(
                {key: conv1x1(channels_lst[idx] // 2, channels_lst[idx], channels_lst[idx] // groups_size)
                 for idx, key in enumerate(keys[:-1])})
        # elif name == 'B' and not self.Bblock_args['shared']:
        #     self._make_notshared_AB(name, channels, groups_size)
        #     # todo: make it run
        #     self._make_notshared_AB(name, channels, groups_size)
        else:
            # return nn.ModuleDict(
            #     {key: convkxk(channels_lst[idx], channels_lst[idx], channels_lst[idx] // groups_size, k=3)
            #      for idx, key in enumerate(keys)})
            # todo: make coarse grid (?)
            return nn.ModuleDict(
                {key: convkxk(channels_lst[idx], channels_lst[idx], groups=channels_lst[idx] // groups_size, k=3)
                if idx != len(keys) - 1 else convkxk(channels_lst[idx], channels_lst[idx], groups=1, k=3)
                 for idx, key in enumerate(keys)})

    def make_inchannel_batchnorms(self, channels: int, postsmoothing: bool) -> {}:
        """ Batchnorms for normalizing within smoothing """
        if postsmoothing:
            s = self.num_post_smoothings
        else:
            s = self.num_pre_smoothings

        keys = []
        channels_lst = []

        # s = self.num_post_smoothings * self.num_smoothings
        # s = self.num_post_smoothings

        [keys.append('bn' + str(j)) for j in range(self.inchannel_levels * 2 * s)]
        [channels_lst.append(channels // (2 ** (k // (2 * s)))) for k in range(len(keys))]

        return nn.ModuleDict({key: nn.BatchNorm2d(channels_lst[idx]) for idx, key in enumerate(keys)})

    def make_P_batchnorms(self, channels: int) -> {}:
        """ Batchnorms for normalizing after prolongation """
        keys = []
        channels_lst = []

        [keys.append('bn' + str(j)) for j in range(self.inchannel_levels)]
        [channels_lst.append(channels // (2 ** j)) for j in range(self.inchannel_levels)]

        return nn.ModuleDict({key: nn.BatchNorm2d(channels_lst[idx]) for idx, key in enumerate(keys[: -1])})

    def _make_notshared_AB(self, name, channels, groups_size):
        keys = []
        channels_lst = []
        s = self.num_post_smoothings
        [keys.append(name + str(j)) for j in range(self.inchannel_levels * s)]
        [channels_lst.append(channels // (2 ** (j // s))) for j in range(len(keys))]

        return nn.ModuleDict(
            {key: convkxk(channels_lst[idx], channels_lst[idx], channels_lst[idx] // groups_size, k=3)
             for idx, key in enumerate(keys)})

    def forward(self, x):
        pass


class MgBlockMGIC(BaseMgMGIC):

    def __init__(self, in_channels: int, out_channels: int, num_smoothings: int,
                 num_layer: int, device, filters: bool, train_filters: bool, Ablock_args, Bblock_args):

        super(MgBlockMGIC, self).__init__(in_channels, out_channels, num_smoothings, num_layer, device, filters,
                                          train_filters, Ablock_args,
                                          Bblock_args)

        self.inchannelAblock = self.make_inchannelblock('A', out_channels, self.groups_size)
        self.inchannelBblock = self.make_inchannelblock('B', out_channels, self.groups_size)
        self.Vcycle_ops = self.make_V_op(out_channels)

    def make_V_op(self, out_channels):
        op_dict = nn.ModuleDict()
        if not self.linear:
            [op_dict.update({str(v): nn.ModuleDict(
                {'inchannelR': self.make_inchannelblock('R', out_channels, self.R_groups_size),
                 'inchannelP': self.make_inchannelblock('P', out_channels, self.R_groups_size),
                 'inchannelI': self.make_inchannelblock('I', out_channels, self.R_groups_size),
                 'pre_batchnorms': self.make_inchannel_batchnorms(out_channels, False),
                 'post_batchnorms': self.make_inchannel_batchnorms(out_channels, True),
                 'prolong_batchnorms': self.make_P_batchnorms(out_channels),
                 })}) for v in range(self.num_smoothings)]
        else:
            [op_dict.update({str(v): nn.ModuleDict(
                {'inchannelR': self.make_inchannelblock('R', out_channels, self.R_groups_size),
                 'inchannelP': self.make_inchannelblock('P', out_channels, self.R_groups_size),
                 'pre_batchnorms': self.make_inchannel_batchnorms(out_channels, False),
                 'post_batchnorms': self.make_inchannel_batchnorms(out_channels, True),
                 'prolong_batchnorms': self.make_P_batchnorms(out_channels),
                 })}) for v in range(self.num_smoothings)]

        return op_dict

    def prolong(self, u, u0, i):
        """
        Transfers u from coarser to finer grid (in-channel)
            - if initial u on coarser grid is not 0, correction u0 - u is calculated first
            - prolongated solution is normalized
        """
        P = self.Vcycle_ops[str(self.v)]['inchannelP']['P' + str(i)]
        bn = self.Vcycle_ops[str(self.v)]['prolong_batchnorms']['bn' + str(i)]
        # P = self.inchannelP['P' + str(i)]
        # bn = self.prolong_batchnorms['bn' + str(i)]
        if u0 is not None:
            return bn(P(u - u0))
        else:
            return bn(P(u))

    def restrict(self, r, j):
        """ Restrict the residual to the next coarser grid """
        R = self.Vcycle_ops[str(self.v)]['inchannelR']['R' + str(j)]
        # R = self.inchannelR['R' + str(j)]
        return R(r)

    def transfer(self, u, j):
        I = self.Vcycle_ops[str(self.v)]['inchannelI']['I' + str(j)]
        # I = self.inchannelI['I' + str(j)]
        return I(u)

    def in_channel_smoothing_step(self, f, u0, j, postsmoothing: bool):
        """
        Performs an in-channel smoothing step
            - get batchnorms and operations
            f: restricted residiual
            u0: initial solution if None u0=0
            j: level index
            s: smoothing step index
            if postsmoothing:
                new batchnorm operations
        """
        if postsmoothing:
            bn1, bn2 = self._get_smoothing_batchnorms(j, self.v, self.num_post_smoothings, postsmoothing)
        else:
            bn1, bn2 = self._get_smoothing_batchnorms(j, self.v, 1, postsmoothing)

        A, B = self._get_mgic_operations(j)
        if u0 is None:
            u0 = torch.zeros(f.shape, device=self.device)

        u = Smoothing((A, B), (bn1, bn2))((f, u0))

        # # smoothing
        # r = f - A(u0)
        # r = bn1(r)
        # r = relu(r)
        # u = B(r)
        # u = bn2(u)
        # u = relu(u)
        #return u + u0

        return u

    def _get_mgic_operations(self, j):
        keyA, keyB = self.keys(j)
        return self.inchannelAblock[keyA + str(j)], self.inchannelBblock[keyB + str(j)]

    def pre_smoothing(self, f, u, j):
        for s in range(self.num_pre_smoothings):
            u = self.in_channel_smoothing_step(f, u, j, False)
        return u

    def post_smoothing(self, f, u, j):
        for s in range(self.num_pre_smoothings):
            u = self.in_channel_smoothing_step(f, u, j, True)
        return u

    def MGIC(self, f):
        # if not self.linear:
        #     u = torch.zeros(f.shape, device=self.device)
        #     u = self.nonlinear_MGIC(f, u, 0)
        # else:
        #     u = self.linear_MGIC(f, 0)
        for v in range(self.num_smoothings):
            self.v = v
            if self.v == 0:
                u = None
            if not self.linear:
                u = torch.zeros(f.shape, device=self.device)
                u = self.nonlinear_MGIC(f, u, 0)
            else:
                u = self.linear_MGIC(f, u, 0)
        return u

    def nonlinear_MGIC(self, f, u, j):
        """
        Recursive call of multigrid-in-channel level
            1. performs smoothing u = u + B(f-Au)
            2. if not coarsest level:
                calculate residual r = f - Au
                restrict to coarser level f_c = R(r)
                call MGIC() to update u until coarsest level
            on every level back up:
            3. prolongates and corrects u = u + P(u)
            4. performs post-smoothing u = u + B(f-Au)

            return u: solution after an in-channels V-cycle
        """
        # pre-smoothing
        u = self.pre_smoothing(f, u, j)

        if j != range(self.inchannel_levels)[-1]:  # hint: 5 levels: j \in {0,1,2,3,4}

            # residual
            r = f - self.inchannelAblock['A' + str(j)](u)

            # restriction
            f_c = self.restrict(r, j) # todo +
            u0 = self.transfer(u, j)

            # recursive V cycle
            u_c = self.nonlinear_MGIC(f_c, u0, j + 1)

            # prolongation and correction
            u = u + self.prolong(u_c, u0, j)

        # post-smoothing
        u = self.post_smoothing(f, u, j)

        return u

    def linear_MGIC(self, f, u, j):
        """
        Recursive call of multigrid-in-channel level
            1. performs smoothing u = u + B(f-Au)
            2. if not coarsest level:
                calculate residual r = f - Au
                restrict to coarser level f_c = R(r)
                call MGIC() to update u until coarsest level
            on every level back up:
            3. prolongates and corrects u = u + (Pu)
            4. performs post-smoothing u = u + B(f-Au)

            return u: solution after an in-channels V-cycle
        """
        # pre-smoothing
        if self.v == 0:
            u = self.pre_smoothing(f, None, j)
        elif self.v != 0 and j > 0:
            u = self.pre_smoothing(f, None, j)
        elif self.v > 0 and j == 0:
            pass

        # u = self.pre_smoothing(f, v or None, j)

        if j != range(self.inchannel_levels)[-1]:  # hint: 5 levels: j \in {0,1,2,3,4}

            # residual
            r = f - self.inchannelAblock['A' + str(j)](u)

            # restriction
            f_c = self.restrict(r, j)

            # recursive V cycle
            # u_c = self.linear_MGIC(f_c, v, j + 1)
            u_c = self.linear_MGIC(f_c, u, j + 1)

            # prolongation and correction
            u = u + self.prolong(u_c, None, j)

        # post-smoothing
        u = self.post_smoothing(f, u, j)

        return u

    def forward(self, x):
        f = x[0]
        u = x[1]

        if self.num_layer > 1:
            u = self.convI(u)
            f = self.convR(f) + self.inchannelAblock['A0'](u)
        # ------------------------------------------------------------------------------
        u = self.MGIC(f)
        # ------------------------------------------------------------------------------

        r = f - self.inchannelAblock['A0'](u)
        # r = self.relu(r)

        return r, u


class Smoothing(nn.Module):
    def __init__(self, layers, batchnorms):
        super().__init__()
        # self.layer_dict = layers
        self.A, self.B = layers
        self.bn1, self.bn2 = batchnorms

        #self.relu = nn.ReLU()
        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()
        # self.activation = nn.LeakyReLU()

    def forward(self, x):
        f = x[0]
        u0 = x[1]
        # smoothing
        r = f - self.A(u0)
        r = self.bn1(r)
        r = self.activation(r)
        u = self.B(r)
        u = self.bn2(u)
        u = self.activation(u)

        return u + u0
