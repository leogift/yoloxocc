#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.
import torch
from torch import nn

from yoloxocc.models.network_blocks import C2aLayer, C2PPLayer

import ssl
context = ssl._create_unverified_context()
ssl._create_default_https_context = ssl._create_unverified_context
import torchvision

class RegnetNeckPAN(nn.Module):
    def __init__(
        self,
        model="regnet_x_800mf",
        model_reduce=4, # 模型深度缩减
        in_features=("trans3", "trans4", "trans5"),
        channels=[256, 512, 1024],
        out_features=("bev_backbone3", "bev_backbone4", "bev_backbone5"),
        act="silu",
        layer_type=C2aLayer,
        n=2,
        pp_repeats=0,
        drop_rate=0.,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 加载预训练模型
        self.model = torchvision.models.__dict__[model](weights='IMAGENET1K_V1')

        # 丢弃分类头
        del self.model.avgpool
        del self.model.fc
        # 模型深度缩减
        new_lens = []
        for idx in range(self.model.trunk_output.__len__()):
            if model_reduce > 1:
                old_len = self.model.trunk_output[idx].__len__()
                del_len = old_len*(model_reduce-1)//model_reduce
                if del_len > 0:
                    del self.model.trunk_output[idx][-del_len:]
            new_lens.append(self.model.trunk_output[idx].__len__())
        print(f"{model} {model_reduce}: LENS={new_lens}")

        # 训练参数
        for p in self.model.parameters():
            p.requires_grad = True  # for training

        self.csp4 = layer_type(
            int(channels[1] * 2),
            int(channels[1]),
            n,
            act=act,
        )
        self.csp5 = layer_type(
            int(channels[2] * 2),
            int(channels[2]),
            n,
            act=act,
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

        x = x3 # s8
        outputs[self.out_features[0]] = x # s8

        x = self.model.trunk_output[-2](x) # s16
        x = torch.cat([x, x4], 1)
        x = self.csp4(x)
        outputs[self.out_features[1]] = x # s16

        x = self.model.trunk_output[-1](x) # s32
        x = torch.cat([x, x5], 1)
        x = self.csp5(x)
        if self.training:
            x = self.drop(x)
        x = self.last_layer(x)
        outputs[self.out_features[2]] = x # s32

        return outputs
