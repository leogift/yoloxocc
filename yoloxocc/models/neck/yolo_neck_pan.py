#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.
import torch
from torch import nn

from yoloxocc.models.network_blocks import BaseConv, C2aLayer, C2PPLayer

class YOLONeckPAN(nn.Module):
    def __init__(
        self,
        in_features=["bev_trans3", "bev_trans4", "bev_trans5"],
        out_features=["bev_pan_s2", "bev_pan_s4", "bev_pan_s8"],
        channels=[256, 512, 1024],
        act="silu",
        n=2,
        simple_reshape=False,
        pp_repeats=0,
        drop_rate=0.,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # downsample and conv
        self.downsample3_4 = BaseConv(
            channels[0],
            channels[0],
            3,
            stride=2,
            groups=channels[0],
            act=act,
        ) if simple_reshape==False else nn.MaxPool2d(3, stride=2, padding=1)
        self.downsample4_5 = BaseConv(
            channels[1],
            channels[1],
            3,
            stride=2,
            groups=channels[1],
            act=act,
        ) if simple_reshape==False else nn.MaxPool2d(3, stride=2, padding=1)

        self.csp3 = C2aLayer(
            channels[0],
            channels[0],
            n,
            act=act,
            use_rep=True,
        )
        self.csp4 = C2aLayer(
            channels[1] + channels[0],
            channels[1],
            n,
            act=act,
            use_rep=True,
        )
        self.csp5 = C2aLayer(
            channels[2] + channels[1],
            channels[2],
            n,
            act=act,
            drop_rate=drop_rate,
            use_rep=True,
        )

        self.drop   = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

        # last_layer
        self.last_layer = nn.Identity() if pp_repeats==0 else C2PPLayer(
            channels[2],
            channels[2],
            n=pp_repeats,
            act=act,
            drop_rate=drop_rate,
        )


    def forward(self, inputs):
        """
        Args:
            inputs: backbone output.
        Returns:
            Tuple[Tensor]: FPN feature.
        """
        features = [inputs[f] for f in self.in_features]
        [x3, x4, x5] = features
        outputs = inputs

        x = self.csp3(x3) # s8
        outputs[self.out_features[0]] = x # s8

        x = self.downsample3_4(x) # s16
        x = torch.cat([x, x4], 1)
        x = self.csp4(x)
        outputs[self.out_features[1]] = x # s16

        x = self.downsample4_5(x) # s32
        x = torch.cat([x, x5], 1)
        x = self.csp5(x)
        if self.training:
            x = self.drop(x)
        x = self.last_layer(x)
        outputs[self.out_features[2]] = x # s32

        return outputs
