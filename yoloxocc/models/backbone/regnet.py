#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.
import torch
from torch import nn

from yoloxocc.models.network_blocks import C2PPLayer

import ssl
context = ssl._create_unverified_context()
ssl._create_default_https_context = ssl._create_unverified_context
import torchvision

class Regnet(nn.Module):
    def __init__(
        self,
        model="regnet_x_800mf",
        out_features=["backbone3", "backbone4", "backbone5"],
        act="silu",
        pp_repeats=0,
        drop_rate=0.,
    ):
        super().__init__()

        # 加载预训练模型
        self.model = torchvision.models.__dict__[model](weights='IMAGENET1K_V1')

        # 丢弃分类头
        del self.model.avgpool
        del self.model.fc

        # 训练参数
        for p in self.model.parameters():
            p.requires_grad = True  # for training

        self.channels = [self.model.stem[0].out_channels]
        # 模型深度
        block_lens = []
        for idx in range(self.model.trunk_output.__len__()):
            block_lens.append(self.model.trunk_output[idx].__len__())
            self.channels.append(self.model.trunk_output[idx][-1].f[0].out_channels)
        print(f"{model}: LENS={block_lens}, CHANNELS={self.channels}")

        self.drop   = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

        # last_layer
        self.last_layer = nn.Identity() if pp_repeats==0 else C2PPLayer(
            self.channels[-1],
            self.channels[-1],
            n=pp_repeats,
            act=act,
            drop_rate=drop_rate,
        )


    def forward(self, inputs):
        x = inputs["input"]

        outputs = inputs

        x = self.model.stem(x)

        for idx in range(self.model.trunk_output.__len__()):
            x = self.model.trunk_output[idx](x)
            key = f"backbone{2+idx}"
            outputs[key] = x

        if self.training:
            x = self.drop(x)
        x = self.last_layer(x)
        outputs["backbone5"] = x

        return outputs
