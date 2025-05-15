#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch.nn as nn

from yoloxocc.models.network_blocks import get_activation, BaseConv
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
        self.conv = nn.Sequential(*[
            BaseConv(
                int(in_channel), 
                int(in_channel), 
                3,
                act=act,
            ),
            BaseConv(
                int(in_channel), 
                int(in_channel), 
                3,
                act=act,
            ),
        ])
        self.pred = nn.Conv2d(
            int(in_channel), 
            vox_y, 
            kernel_size=1, 
            bias=True)

        self.drop   = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

        initialize_weights(self)
 

    def forward(self, inputs):
        x = inputs[self.in_feature]

        y = self.upsample(x)
        y = self.conv(y)
        if self.training:
            y = self.drop(y)
        y = self.pred(y)

        return y
