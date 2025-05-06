#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn

from yoloxocc.models.network_blocks import get_activation, C2aLayer
from yoloxocc.utils import initialize_weights

class YOLONeckFPN(nn.Module):

    def __init__(
        self,
        in_features=("backbone3", "backbone4", "backbone5"),
        in_channels=[256, 512, 1024],
        out_features=("fpn3", "fpn4", "fpn5"),
        act="silu",
        layer_type=C2aLayer,
        simple_reshape=False,
        n=2,
    ):
        super().__init__()
        assert len(in_features) == len(in_channels) and len(in_features) == 3
        self.in_features = in_features
        self.out_features = out_features

        # upsample and conv
        self.upsample5_4 = nn.Sequential(*[
            nn.ConvTranspose2d(
                int(in_channels[2]), 
                int(in_channels[2]), 
                kernel_size=4, 
                stride=2, 
                padding=1,
                groups=int(in_channels[2]),
                bias=True
            ),
            get_activation(act)()
        ]) if simple_reshape==False else nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample4_3 = nn.Sequential(*[
            nn.ConvTranspose2d(
                int(in_channels[1]), 
                int(in_channels[1]), 
                kernel_size=4, 
                stride=2, 
                padding=1,
                groups=int(in_channels[1]),
                bias=True
            ),
            get_activation(act)()
        ]) if simple_reshape==False else nn.Upsample(scale_factor=2, mode="bilinear")

        self.csp4 = layer_type(
            int((in_channels[2] + in_channels[1])),
            int(in_channels[1]),
            n,
            act=act,
        )
        self.csp3 = layer_type(
            int((in_channels[1] + in_channels[0])),
            int(in_channels[0]),
            n,
            act=act,
        )
        
        initialize_weights(self.upsample5_4)
        initialize_weights(self.upsample4_3)
        initialize_weights(self.csp4)
        initialize_weights(self.csp3)


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

        fpn_out5 = x5
        outputs[self.out_features[2]] = fpn_out5 # 1024/32

        fpn_out4 = self.upsample5_4(fpn_out5)  # 1024/16
        fpn_out4 = torch.cat([fpn_out4, x4], 1)  # 1024->1536/16
        fpn_out4 = self.csp4(fpn_out4)  # 1536->512/16
        outputs[self.out_features[1]] = fpn_out4 # 512/16

        fpn_out3 = self.upsample4_3(fpn_out4)  # 512/8
        fpn_out3 = torch.cat([fpn_out3, x3], 1)  # 512->768/8
        fpn_out3 = self.csp3(fpn_out3)  # 768->256/8
        outputs[self.out_features[0]] = fpn_out3 # 256/8

        return outputs
