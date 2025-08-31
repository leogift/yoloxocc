#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn

from yoloxocc.models.network_blocks import get_activation, BaseConv
from yoloxocc.utils import special_multiples

class OCCHead(nn.Module):
    def __init__(
        self,
        in_feature="bev_fpn3",
        in_channel=256,
        vox_y=4,
        act="silu",
        simple_reshape=False,
        aux_head=False
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
        """
        super().__init__()
        self.in_feature = in_feature
        self.aux_head = aux_head

        # 主输出 upsample
        if not self.aux_head:
            self.upsample = nn.Sequential(*[
                nn.ConvTranspose2d(
                    int(in_channel), 
                    int(in_channel), 
                    kernel_size=4, 
                    stride=2, 
                    padding=1,
                    groups=int(in_channel),
                    bias=True),
                get_activation(act)()
            ]) if simple_reshape==False else nn.Upsample(scale_factor=2, mode="bilinear")

        hidden_channel = special_multiples(in_channel // vox_y)
        self.stem = BaseConv(
            in_channel,
            int(hidden_channel * vox_y),
            1,
            stride=1,
            act=act,
        )

        # 主输出 conv
        if not self.aux_head:
            self.occ_convs = nn.ModuleList(
                nn.Sequential(*[
                    nn.Conv2d(
                        hidden_channel,
                        hidden_channel,
                        kernel_size=3,
                        stride=1,
                        bias=False,
                    ),
                    nn.InstanceNorm2d(hidden_channel),
                    get_activation(act)(),
                    nn.Conv2d(
                        hidden_channel,
                        hidden_channel,
                        kernel_size=3,
                        stride=1,
                        bias=False,
                    ),
                    nn.InstanceNorm2d(hidden_channel),
                    get_activation(act)(),
                ])
                for _ in range(vox_y)
            )
        else:
            self.occ_convs = nn.ModuleList(
                nn.Identity()
                for _ in range(vox_y)
            )

        self.occ_preds = nn.ModuleList(
            nn.Conv2d(
                hidden_channel, 
                1,
                kernel_size=1, 
                bias=True)
            for _ in range(vox_y)
        )
        import math
        prior_prob = 1e-2
        for _, m in self.occ_preds.named_modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    b = m.bias.view(1, -1)
                    b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
                    m.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        self.vox_y = vox_y
        self.hidden_channel = hidden_channel
 
    def forward(self, inputs):
        x = inputs[self.in_feature]
        occ = []

        if self.training:
            if not self.aux_head:
                x = self.upsample(x)
            
            x = self.stem(x)
            
            split_channels = [self.hidden_channel]*self.vox_y
            _xs = x.split(split_channels, 1)
            for _x, conv, pred in zip(_xs, self.occ_convs, self.occ_preds):
                _x = conv(_x)
                _x = pred(_x)
                occ.append(_x)
            
            occ = torch.cat(occ, dim=1)
            return occ
        
        else:
            if not self.aux_head:
                x = self.upsample(x)
                x = self.stem(x)
            
                split_channels = [self.hidden_channel]*self.vox_y
                _xs = x.split(split_channels, 1)
                for _x, conv, pred in zip(_xs, self.occ_convs, self.occ_preds):
                    _x = conv(_x)
                    _x = pred(_x)
                    occ.append(_x)
                
                occ = torch.cat(occ, dim=1)
                return occ
