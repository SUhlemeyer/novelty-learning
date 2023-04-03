import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_planes, out_planes, groups, stride=(1, 1)):
    """1x1 convolution"""
    if not in_planes % groups == 0:
        raise ValueError('Number if channels is not divisible by groupsize')
    else:
        groups = groups # in_planes // groupsize
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), groups=groups, stride=stride, bias=False)


def trans_conv1x1(in_planes, out_planes, groupsize, stride=(1, 1)):
    """1x1 convolution"""
    if not in_planes % groupsize == 0:
        raise ValueError('Number if channels is not divisible by groupsize')
    else:
        groups = in_planes // groupsize

    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=(1, 1), groups=groups, stride=stride, bias=False)


def conv5x5(in_channels, out_channels, groups, stride=(1, 1)):
    """Helper function for 2d convolution with 3x3 filter and padding"""
    if groups == 'c_in':
        groups = in_channels
    else:
        groups = groups
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=(5, 5),
                     stride=stride,
                     padding=(1, 1),
                     bias=False,
                     groups=groups)


def convkxk(in_channels, out_channels, groups, k, stride=(1, 1)):
    """Helper function for 2d convolution with 3x3 filter and padding"""

    if groups == 'c_in':
        groups = in_channels
    elif groups == 0:
        groups = 1
    else:
        groups = groups

    if k == 3:
        return nn.Conv2d(in_channels, out_channels,
              kernel_size=(k, k),
              stride=stride,
              padding=(1, 1),
              bias=False,
              groups=groups)
    elif k == 1:
        return nn.Conv2d(in_channels, out_channels,
                         kernel_size=(k, k),
                         stride=stride,
                         padding=(1, 1),
                         bias=False,
                         groups=groups)


    elif k == 5:
        return nn.Conv2d(in_channels, out_channels,
                     kernel_size=(k, k),
                     stride=stride,
                     padding=(2, 2),
                     # padding=(2, 2),
                     bias=False,
                     groups=groups)


def conv3x3(in_channels, out_channels, groups, stride=(1, 1)):
    """Helper function for 2d convolution with 3x3 filter and padding"""
    if groups == 'c_in':
        groups = in_channels
    else:
        groups = groups
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=(3, 3),
                     stride=stride,
                     padding=(1, 1),
                     bias=False,
                     groups=groups)


def deconv3x3(in_channels, out_channels, groups, stride=(1, 1)):
    return nn.ConvTranspose2d(in_channels, out_channels,
                              kernel_size=(3, 3),
                              stride=stride,
                              padding=(1, 1),
                              bias=False,
                              groups=groups)


def conv7x7(in_channels, out_channels, stride=2):
    """Helper function for 2d convolution with 7x7 filter and padding"""
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=(7, 7),
                     stride=(stride, stride),
                     padding=(3, 3),
                     bias=False)