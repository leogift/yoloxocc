#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn
from timm.layers import DropPath

from yoloxocc.utils.model_utils import fuse_conv_and_bn

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""
    def __init__(self, inplace=True):
        super().__init__()

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu"):
    if name == "silu":
        module = SiLU
    elif name == "relu":
        module = nn.ReLU
    else:
        module = nn.Identity
        
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/relu block"""

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        ksize, 
        stride=1, 
        groups=1,
        act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act)()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
        
    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck"""
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=0.5,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = BaseConv(in_channels, hidden_channels, 3, stride=1, act=act)
        self.conv2 = BaseConv(hidden_channels, out_channels, 1, stride=1, act="")

        self.use_add = in_channels == out_channels
        self.act = get_activation(act)()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)

        if self.use_add:
            y = y + x
        
        return self.act(y)


class RepSConv(nn.Module):
    # Rep Conv
    def __init__(
        self,
        in_channels,
        out_channels,
        act="silu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cv = BaseConv(in_channels, out_channels, 3, stride=1, act="")
        
        self.cv_1x1 = BaseConv(in_channels, out_channels, 1, stride=1, act="")
        
        self.act = get_activation(act)()

    def forward(self, x):
        y = self.cv(x) + self.cv_1x1(x)
        y = self.act(y)

        return y

    def fuseforward(self, x):
        y = self.repSConv(x)
        return self.act(y)

    def get_equivalent_kernel_bias(self, cv, cv_1x1):
        kernel1x1, bias1x1 = self._fuse_bn_tensor(cv_1x1)
        kernel3x3, bias3x3 = self._fuse_bn_tensor(cv)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0

        if hasattr(branch, "bn"):
            branch.conv = fuse_conv_and_bn(branch.conv, branch.bn)  # update conv
            delattr(branch, "bn")  # remove batchnorm
            branch.forward = branch.fuseforward  # update forward

        kernel = branch.conv.weight
        bias = branch.conv.bias
        return kernel, bias

    def switch_to_deploy(self):
        if hasattr(self, "repSConv"):
            return

        self.repSConv = nn.Conv2d(in_channels=self.in_channels, 
                                out_channels=self.out_channels,
                                kernel_size=3, 
                                stride=1,
                                padding=1,
                                groups=1,
                                bias=True)
        
        self.repSConv.weight.data, self.repSConv.bias.data = self.get_equivalent_kernel_bias(self.cv, self.cv_1x1)

        for para in self.parameters():
            para.detach_()

        delattr(self, "cv")
        delattr(self, "cv_1x1")


class RepBottleneck(nn.Module):
    # Rep bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=0.5,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = RepSConv(in_channels, hidden_channels, act=act)
        self.conv2 = BaseConv(hidden_channels, out_channels, 1, stride=1, act="")

        self.use_add = in_channels == out_channels
        self.act = get_activation(act)()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)

        if self.use_add:
            y = y + x

        return self.act(y)


##### Layer #####
class C2aLayer(nn.Module):
    def __init__(self,
        in_channels,
        out_channels=None,
        n=1,
        act="silu",
        drop_rate=0.
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.hidden_channels = out_channels//2  # hidden channels
        
        self.cv1 = BaseConv(in_channels, 2 * self.hidden_channels, 1, stride=1, act=act)
        self.m = nn.ModuleList(
            RepBottleneck(
                self.hidden_channels, self.hidden_channels, act=act
            )
            for _ in range(n)
        )
        self.cv2 = BaseConv(2 * self.hidden_channels, out_channels, 1, stride=1, act=act)
        self.droppath = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.cv1(x)
        x_1, x_2 = x.split((self.hidden_channels, self.hidden_channels), 1)

        for idx in range(len(self.m)):
            x_2 = self.m[idx](x_2)

        if self.training:
            y = torch.cat((self.droppath(x_1), x_2), dim=1)
        else:
            y = torch.cat((x_1, x_2), dim=1)

        return self.cv2(y)


class C2PPLayer(nn.Module):
    """Pyramid Pooling and Squeeze Excitation"""
    def __init__(self, 
        in_channels,
        out_channels, 
        n=1,
        act="silu",
        drop_rate=0.,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.hidden_channels = out_channels//2  # hidden channels
        
        self.cv1 = BaseConv(in_channels, 2 * self.hidden_channels, 1, stride=1, act=act)
        self.pp = nn.ModuleList()
        for idx in range(3):
            self.pp.append(
                nn.Sequential(
                    *[
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                        for _n in range(n)
                    ],
                )
            )
        self.cv2 = BaseConv(2 * self.hidden_channels, out_channels, 1, stride=1, act=act)
        self.droppath = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.cv1(x)
        x_1, x_2 = x.split((self.hidden_channels, self.hidden_channels), 1)

        for idx in range(3):
            x_2 = self.pp[idx](x_2) + x_2

        if self.training:
            y = torch.cat((self.droppath(x_1), x_2), dim=1)
        else:
            y = torch.cat((x_1, x_2), dim=1)

        return self.cv2(y)
