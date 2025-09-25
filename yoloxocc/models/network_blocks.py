#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
import math

from yoloxocc.utils.model_utils import fuse_conv_and_bn
from yoloxocc.utils import special_multiples


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


class SeparableConv(nn.Module):
    """A DWConv2d -> Batchnorm -> silu/relu
      -> PWConv2d -> Batchnorm block"""

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        ksize, 
        stride=1, 
        act="silu"
    ):
        super().__init__()
        
        self.dwconv = BaseConv(
            in_channels, 
            in_channels, 
            ksize, 
            stride=stride, 
            groups=in_channels, 
            act=act
        )

        self.pwconv = BaseConv(in_channels, out_channels, 1, stride=1, act="")
        
    def forward(self, x):
        y = self.dwconv(x)
        y = self.pwconv(y)

        return y
        

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
        hidden_channels = special_multiples(out_channels * expansion)

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


class RepSConv3x3(nn.Module):
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

        self.cv_1x1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=True)
        self.cv_1x3 = nn.Conv2d(in_channels, out_channels, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.cv_3x1 = nn.Conv2d(in_channels, out_channels, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.act = get_activation(act)()

    def forward(self, x):
        y = self.cv(x) + self.cv_1x1(x) + self.cv_1x3(x) + self.cv_3x1(x)
        y = self.act(y)

        return y

    def fuseforward(self, x):
        y = self.repSConv(x)
        return self.act(y)

    def get_equivalent_kernel_bias(self):
        kernel, bias = self._fuse_bn_tensor(self.cv)
        kernel1x1, bias1x1 = self.cv_1x1.weight, self.cv_1x1.bias
        kernel1x3, bias1x3 = self.cv_1x3.weight, self.cv_1x3.bias
        kernel3x1, bias3x1 = self.cv_3x1.weight, self.cv_3x1.bias

        kernel = kernel \
                + self._pad_1x1_to_3x3_tensor(kernel1x1) \
                + self._pad_1x3_to_3x3_tensor(kernel1x3) \
                + self._pad_3x1_to_3x3_tensor(kernel3x1)
        
        bias = bias + bias1x1 + bias1x3 + bias3x1
        
        return kernel, bias

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _pad_1x3_to_3x3_tensor(self, kernel1x3):
        if kernel1x3 is None:
            return 0
        
        return torch.nn.functional.pad(kernel1x3, [0, 0, 1, 1])

    def _pad_3x1_to_3x3_tensor(self, kernel3x1):
        if kernel3x1 is None:
            return 0
        
        return torch.nn.functional.pad(kernel3x1, [1, 1, 0, 0])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0

        if hasattr(branch, "bn"):
            branch.conv = fuse_conv_and_bn(branch.conv, branch.bn)  # update conv
            delattr(branch, "bn")  # remove batchnorm
            branch.forward = branch.fuseforward  # update forward
        
        kernel, bias = branch.conv.weight, branch.conv.bias

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
        
        self.repSConv.weight.data, self.repSConv.bias.data = self.get_equivalent_kernel_bias()

        for para in self.parameters():
            para.detach_()

        delattr(self, "cv")
        delattr(self, "cv_1x1")
        delattr(self, "cv_1x3")
        delattr(self, "cv_3x1")


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
        hidden_channels = special_multiples(out_channels * expansion)

        self.conv1 = RepSConv3x3(in_channels, hidden_channels, act=act)
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
        n=2,
        act="silu",
        drop_rate=0.,
        use_rep=True,
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
        self.hidden_channels = special_multiples(out_channels//2)
        
        self.cv1 = BaseConv(in_channels, 2 * self.hidden_channels, 1, stride=1, act=act)
        
        self.m = nn.ModuleList(
            RepBottleneck(
                self.hidden_channels, self.hidden_channels, act=act, 
            )
            if use_rep else
            Bottleneck(
                self.hidden_channels, self.hidden_channels, act=act, 
            )
            for _n in range(n)
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
        out_channels=None, 
        n=2,
        act="silu",
        drop_rate=0.,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.hidden_channels = special_multiples(out_channels//2)
        
        self.cv1 = BaseConv(in_channels, 2 * self.hidden_channels, 1, stride=1, act=act)
        
        self.m = nn.ModuleList()
        for idx in range(3):
            self.m.append(
                nn.Sequential(
                    *[
                        nn.MaxPool2d(3, stride=1, padding=1)
                        for _n in range(n)
                    ],
                )
            )
        
        self.cv2 = BaseConv(2 * self.hidden_channels, out_channels, 1, stride=1, act=act)
        self.droppath = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.cv1(x)
        x_1, x_2 = x.split((self.hidden_channels, self.hidden_channels), 1)

        x_m = None
        for idx in range(3):
            x_2 = self.m[idx](x_2)
            x_m = x_2 if x_m is None else x_m + x_2
        x_2 = x_m

        if self.training:
            y = torch.cat((self.droppath(x_1), x_2), dim=1)
        else:
            y = torch.cat((x_1, x_2), dim=1)

        return self.cv2(y)


class C2MLPLayer(nn.Module):
    """
    MLP layer
    """
    def __init__(self,
        in_channels,
        out_channels=None, 
        H=16, W=16,
        expansion=2,
        act="silu",
        drop_rate=0.,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.hidden_channels = special_multiples(out_channels//2)
        
        self.cv1 = BaseConv(in_channels, 2 * self.hidden_channels, 1, stride=1, act=act)
        
        hidden_dims = special_multiples(H*W * expansion)

        self.linear1 = nn.Linear(H*W, hidden_dims, bias=True)
        self.linear1.weight.data = nn.Parameter(
            torch.cat([torch.eye(H*W), torch.zeros([hidden_dims-H*W, H*W])], dim=0),
            requires_grad=True
        )
        self.linear1.bias.data = nn.Parameter(torch.zeros(hidden_dims), requires_grad=True)
        self.linear1.requires_grad_(True)

        self.act = get_activation(act)()
        
        self.linear2 = nn.Linear(hidden_dims, H*W, bias=True)
        self.linear2.weight.data = nn.Parameter(
            torch.cat([torch.eye(H*W), torch.zeros([H*W, hidden_dims-H*W])], dim=1),
            requires_grad=True
        )
        self.linear2.bias.data = nn.Parameter(torch.zeros(H*W), requires_grad=True)
        self.linear2.requires_grad_(True)

        self.cv2 = BaseConv(2 * self.hidden_channels, out_channels, 1, stride=1, act=act)
        self.droppath = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.cv1(x)
        x_1, x_2 = x.split((self.hidden_channels, self.hidden_channels), 1)

        B, C, H, W = x_2.shape
        x_m = x_2.view(B, C, -1)
        x_m = self.linear1(x_m)
        x_m = self.act(x_m)
        x_m = self.linear2(x_m)
        x_2 = x_m.view(B, C, H, W)

        if self.training:
            y = torch.cat((self.droppath(x_1), x_2), dim=1)
        else:
            y = torch.cat((x_1, x_2), dim=1)

        return self.cv2(y)

