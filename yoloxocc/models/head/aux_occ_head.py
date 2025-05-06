#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch.nn as nn

from yoloxocc.models.network_blocks import get_activation
from yoloxocc.utils import initialize_weights

class AUXOCCHead(nn.Module):
    def __init__(
        self,
        in_features=("bev_fpn3", "bev_fpn4", "bev_fpn5"),
        in_channels=[256, 512, 1024],
        vox_y=4,
        act="silu",
    ):
        super().__init__()
        self.in_features = in_features

        # 辅助输出头
        self.stems = nn.ModuleList()
        self.preds = nn.ModuleList()
        for i in range(len(in_channels)):
            self.stems.append(
                nn.Sequential(*[
	                nn.Conv2d(
	                    int(in_channels[i]), 
	                    int(in_channels[0]), 
	                    kernel_size=3,
	                    stride=1,
	                    padding=1,
	                    bias=True
	                ),
					get_activation(act)()
				])
            )
            self.preds.append(
                nn.Conv2d(
                    int(in_channels[0]),
                    vox_y,
                    kernel_size=1,
                    bias=True)
            )

        initialize_weights(self)
        

    def forward(self, inputs):
        xin = [inputs[f] for f in self.in_features]

        ys = []
        if self.training:
            for stem, pred, x in zip(self.stems, self.preds, xin):
                y = stem(x)
                y = pred(y)

                ys.append(y)

            # 验证时，无效
            return ys
            