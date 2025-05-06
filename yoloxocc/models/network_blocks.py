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
        super(SiLU, self).__init__()

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
            groups=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act)()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
        
    def fuseforward(self, x):
        return self.act(self.conv(x))


class RepSConv(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        act="silu",
        drop_rate=0.,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cv = BaseConv(in_channels, out_channels, 3, stride=1, act="")
        
        self.cv_1x1 = BaseConv(in_channels, out_channels, 1, stride=1, act="")
        
        self.act = get_activation(act)()
        self.drop   = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        y = self.cv(x) + self.cv_1x1(x)
        y = self.act(y)

        if self.training:
            y = self.drop(y)

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


class RepBottleneck(RepSConv):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        act="silu",
        drop_rate=0.,
    ):
        super().__init__(in_channels, out_channels, act, drop_rate)
        self.use_add = in_channels == out_channels

    def forward(self, x):
        y = self.cv(x) + self.cv_1x1(x)
        y = self.act(y)

        if self.training:
            y = self.drop(y)

        if self.use_add:
            y = y + x

        return y

    def fuseforward(self, x):
        y = self.repSConv(x)
        y = self.act(y)

        if self.use_add:
            y = y + x

        return y


##### Transformer #####
class CrossAttention(nn.Module):
    def __init__(self, 
        q_dim, 
        kv_dim,
        reduction=2,
        heads=8,
        drop_rate=0.,
    ):
        super().__init__()
        assert q_dim % heads == 0

        self.heads      = heads
        self.head_dim   = q_dim // self.heads
        self.key_dim    = self.head_dim // reduction
        self.scale      = self.key_dim ** (-0.5)
        nh_key_dim      = self.key_dim * self.heads

        self.q          = BaseConv(q_dim, nh_key_dim, 1, stride=1, act="")
        self.kv         = BaseConv(kv_dim, nh_key_dim+q_dim, 1, stride=2, act="")
        self.attn_drop  = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

        self.proj       = nn.Conv2d(q_dim, q_dim, kernel_size=1, bias=True)

    def forward(self, x, y):
        B, C, H, W = x.shape

        # B,C,H,W -> B,C,H,W
        q = self.q(x)
        # B,C,H,W -> B,Heads,Kd,N
        q = q.view(B, self.heads, self.key_dim, -1)

        # B,C,H,W -> B,C,H,W
        kv = self.kv(y)
        # B,C,H,W -> B,Heads,Kd+D,N
        kv = kv.view(B, self.heads, self.key_dim+self.head_dim, -1)
        # B,Heads,Kd+D,N -> B,Heads,Kd,N, B,Heads,D,N
        k,v = kv.split((self.key_dim, self.head_dim), 2)

        # B,Heads,N,Kd @ B,Head,Kd,N -> B,Head,N,N
        attn = (q.transpose(-2, -1).contiguous() @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # B,Head,N,N @ B,Head,N,C/Head -> B,Head,N,C/Head -> B,N,Head,C/Head => B, N, C
        proj = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        # B,N,C => B,C,N => B,C,H,W
        proj = self.proj(proj)

        return proj


class SelfAttention(CrossAttention):
    def __init__(self, 
        dim, 
        reduction=2,
        heads=8, 
        drop_rate=0.,
    ):
        super().__init__(dim, dim, reduction, heads, drop_rate)

    def forward(self, x):
        return super().forward(x, x)


class Mlp(nn.Module):
    def __init__(self, 
        dim, 
        expansion=2,
        act="silu",
        drop_rate=0.
    ):
        super().__init__()
        self.fc1    = BaseConv(dim, dim*expansion, 1, stride=1, act=act)
        self.fc2    = nn.Conv2d(dim*expansion, dim, kernel_size=1, bias=True)
        self.drop1   = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        if self.training:
            x = self.drop1(x)
        x = self.fc2(x)
        return x


class CrossTransformer(nn.Module):
    def __init__(self,
        dim,
        heads=8,
        act="silu",
        drop_rate=0.,
    ):
        """
        Args:
            dim (int): input/output channels.
            heads (int): number of heads. Default value: 1.
        """
        super().__init__()
        self.attn = CrossAttention(
                dim,
                dim,
                heads=heads,
                drop_rate=drop_rate
            )

        self.mlp = Mlp(
                dim=dim, 
                act=act,
                drop_rate=drop_rate
            )

    def forward(self, q, kv):
        y = self.attn(q, kv) + q
        y = self.mlp(y) + y

        return y


class SelfTransformer(CrossTransformer):
    def __init__(self,
        dim,
        heads=8,
        act="silu",
        drop_rate=0.,
    ):
        """
        Args:
            dim (int): input/output channels.
            heads (int): number of heads. Default value: 1.
        """
        super().__init__(dim, heads, act, drop_rate)
        del self.attn
        self.attn = SelfAttention(
                dim,
                heads=heads,
                drop_rate=drop_rate
            )
        
    def forward(self, x):
        y = self.attn(x) + x
        y = self.mlp(y) + y

        return y


class SEModule(nn.Module):
    def __init__(self, 
                 channels, 
                 reduction=4,
                ):
        super().__init__()
        self.reduce = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        y = self.reduce(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        return x * self.sigmoid(y)


##### Layer #####
class C2aLayer(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
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
        self.hidden_channels = out_channels//2  # hidden channels
        
        self.cv1 = BaseConv(in_channels, 2 * self.hidden_channels, 1, stride=1, act=act)
        self.m = nn.ModuleList(
            RepSConv(
                self.hidden_channels, self.hidden_channels, act=act, drop_rate=drop_rate
            )
            for _ in range(n)
        )
        self.cv2 = BaseConv(3 * self.hidden_channels, out_channels, 1, stride=1, act=act)
        self.droppath = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.cv1(x)
        x_1, x_2 = x.split((self.hidden_channels, self.hidden_channels), 1)

        x_m = x_2
        for idx in range(len(self.m)):
            x_2 = self.m[idx](x_2)
            x_m = x_m + x_2
        x_2 = x_m

        if self.training:
            y = torch.cat((self.droppath(x), x_2), dim=1)
        else:
            y = torch.cat((x, x_2), dim=1)

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
        self.cv2 = BaseConv(3 * self.hidden_channels, out_channels, 1, stride=1, act=act)
        self.droppath = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.cv1(x)
        x_1, x_2 = x.split((self.hidden_channels, self.hidden_channels), 1)

        x_m = x_2
        for idx in range(3):
            x_2 = self.pp[idx](x_2)
            x_m = x_m + x_2
        x_2 = x_m

        if self.training:
            y = torch.cat((self.droppath(x), x_2), dim=1)
        else:
            y = torch.cat((x, x_2), dim=1)

        return self.cv2(y)


class C2kLayer(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
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
        self.hidden_channels = out_channels//2  # hidden channels
        
        self.cv1 = BaseConv(in_channels, 2 * self.hidden_channels, 1, stride=1, act=act)
        self.m = nn.ModuleList(
            C2aLayer(
                self.hidden_channels, self.hidden_channels, 1, act=act, drop_rate=drop_rate)
            for _ in range(n)
        )
        self.cv2 = BaseConv(3 * self.hidden_channels, out_channels, 1, stride=1, act=act)
        self.droppath = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.cv1(x)
        x_1, x_2 = x.split((self.hidden_channels, self.hidden_channels), 1)

        x_m = x_2
        for idx in range(len(self.m)):
            x_2 = self.m[idx](x_2)
            x_m = x_m + x_2
        x_2 = x_m

        if self.training:
            y = torch.cat((self.droppath(x), x_2), dim=1)
        else:
            y = torch.cat((x, x_2), dim=1)

        return self.cv2(y)
