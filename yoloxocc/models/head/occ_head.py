#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch.nn as nn

from yoloxocc.models.network_blocks import get_activation, RepBottleneck
from yoloxocc.utils import initialize_weights

class OCCHead(nn.Module):
    def __init__(
        self,
        in_feature="bev_fpn3",
        in_channel=256,
        vox_y=4,
        act="silu",
        simple_reshape=False,
        drop_rate=0.,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
        """
        super().__init__()
        self.in_feature = in_feature

        # 主输出头
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

        self.occ_conv = RepBottleneck(
            int(in_channel), 
            int(in_channel), 
            act=act,
            drop_rate=drop_rate
        )
        self.occ_pred = nn.Conv2d(
            int(in_channel), 
            vox_y, 
            kernel_size=1, 
            bias=True)

        initialize_weights(self)
 

    def forward(self, inputs):
        x = inputs[self.in_feature]

        y = self.upsample(x)
        y = self.occ_conv(y)
        y = self.occ_pred(y)

        return y
