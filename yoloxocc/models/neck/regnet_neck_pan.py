#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.
import torch
from torch import nn

from yoloxocc.models.network_blocks import get_activation, C2PPLayer, C2aLayer
from yoloxocc.utils import initialize_weights

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
        in_channels=[256, 512, 1024],
        out_features=("bev_backbone3", "bev_backbone4", "bev_backbone5"),
        act="silu",
        pp_repeats=0,
        drop_rate=0.,
        layer_type=C2aLayer,
        n=2,
    ):
        super().__init__()
        assert len(in_features) == len(in_channels) and len(in_features) == 3
        self.in_features = in_features
        self.in_channels = in_channels
        self.out_features = out_features

        # 加载预训练模型
        self.model = torchvision.models.__dict__[model](weights='IMAGENET1K_V1', 
            activation=get_activation(act))

        # 丢弃分类头
        del self.model.avgpool
        del self.model.fc
        # 模型深度缩减
        self.output_channels = []
        new_lens = []
        for idx in range(self.model.trunk_output.__len__()):
            if model_reduce > 1:
                old_len = self.model.trunk_output[idx].__len__()
                del_len = old_len*(model_reduce-1)//model_reduce
                if del_len > 0:
                    del self.model.trunk_output[idx][-del_len:]
            new_lens.append(self.model.trunk_output[idx].__len__())
            self.output_channels.append(self.model.trunk_output[idx][-1].f[0].out_channels)
        print(f"{model} {model_reduce}: LENS={new_lens}, CHANNELS={self.output_channels}")

        # 训练参数
        for p in self.model.parameters():
            p.requires_grad = True  # for training

        self.csp4 = layer_type(
            int((self.output_channels[-2] + in_channels[1])),
            int(self.output_channels[-2]),
            n,
            act=act,
        )
        self.csp5 = layer_type(
            int((self.output_channels[-1] + in_channels[2])),
            int(self.output_channels[-1]),
            n,
            act=act,
        )

        initialize_weights(self.csp4)
        initialize_weights(self.csp5)
		
        self.drop   = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

        # last_layer
        self.last_layer = nn.Sequential(
            nn.Identity() if pp_repeats==0 else C2PPLayer(
                in_channels[2],
                in_channels[2],
                n=pp_repeats,
                act=act,
                drop_rate=drop_rate,
            ),
        )

        initialize_weights(self.last_layer)

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

        x = x3 # 256->256/8
        outputs[self.out_features[0]] = x # 256/8

        x = self.model.trunk_output[-2](x) # 256->256/16
        x = torch.cat([x, x4], 1) # 256->512/16
        x = self.csp4(x) # 512->512/16
        outputs[self.out_features[1]] = x

        x = self.model.trunk_output[-1](x) # 512->512/32
        x = torch.cat([x, x5], 1) # 512->1024/32
        x = self.csp5(x) # 1024->1024/32
        if self.training:
            x = self.drop(x)
        x = self.last_layer(x)
        outputs[self.out_features[2]] = x

        return outputs
