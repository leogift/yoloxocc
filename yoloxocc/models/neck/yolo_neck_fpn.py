#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn

from yoloxocc.models.network_blocks import get_activation, C2aLayer

class YOLONeckFPN(nn.Module):

    def __init__(
        self,
        in_features=("backbone3", "backbone4", "backbone5"),
        channels=[256, 512, 1024],
        out_features=("fpn3", "fpn4", "fpn5"),
        act="silu",
        layer_type=C2aLayer,
        simple_reshape=False,
        n=2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # upsample and conv
        self.upsample5_4 = nn.Sequential(*[
            nn.ConvTranspose2d(
                int(channels[2]), 
                int(channels[2]), 
                kernel_size=4, 
                stride=2, 
                padding=1,
                groups=int(channels[2]),
                bias=True
            ),
            get_activation(act)()
        ]) if simple_reshape==False else nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample4_3 = nn.Sequential(*[
            nn.ConvTranspose2d(
                int(channels[1]), 
                int(channels[1]), 
                kernel_size=4, 
                stride=2, 
                padding=1,
                groups=int(channels[1]),
                bias=True
            ),
            get_activation(act)()
        ]) if simple_reshape==False else nn.Upsample(scale_factor=2, mode="bilinear")

        self.csp4 = layer_type(
            int((channels[2] + channels[1])),
            int(channels[1]),
            n,
            act=act,
        )
        self.csp3 = layer_type(
            int((channels[1] + channels[0])),
            int(channels[0]),
            n,
            act=act,
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

        x = x5
        outputs[self.out_features[2]] = x # s32

        x = self.upsample5_4(x) # s16
        x = torch.cat([x, x4], 1)
        x = self.csp4(x)
        outputs[self.out_features[1]] = x # s16

        x = self.upsample4_3(x) # s8
        x = torch.cat([x, x3], 1)
        x = self.csp3(x)
        outputs[self.out_features[0]] = x # s8

        return outputs
