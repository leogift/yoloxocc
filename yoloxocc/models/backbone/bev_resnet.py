#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.
import torch
from torch import nn

from yoloxocc.models.network_blocks import C2aLayer, C2PPLayer, STNLayer

import ssl
context = ssl._create_unverified_context()
ssl._create_default_https_context = ssl._create_unverified_context
import torchvision

class BEVResnet(nn.Module):
    def __init__(
        self,
        model="resnet18",
        in_features=["bev_trans3", "bev_trans4", "bev_trans5"],
        out_features=["bev_pan_s2", "bev_pan_s4", "bev_pan_s8"],
        act="silu",
        n=2,
        pp_repeats=0,
        drop_rate=0.,
        use_stn=False,
        vox_xyz_size=[128, 4, 128],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 加载预训练模型
        self.model = torchvision.models.__dict__[model](weights='IMAGENET1K_V1')

        # 丢弃分类头
        del self.model.avgpool
        del self.model.fc

        # 训练参数
        for p in self.model.parameters():
            p.requires_grad = True  # for training

        self.channels = [self.model.conv1.out_channels]
        # 模型深度
        block_lens = [self.model.layer1.__len__(), self.model.layer2.__len__(), self.model.layer3.__len__(), self.model.layer4.__len__()]
        self.channels.append(self.model.layer1[-1].conv2.out_channels)
        self.channels.append(self.model.layer2[-1].conv2.out_channels)
        self.channels.append(self.model.layer3[-1].conv2.out_channels)
        self.channels.append(self.model.layer4[-1].conv2.out_channels)
        print(f"{model}: LENS={block_lens}, CHANNELS={self.channels}")

        self.csp1 = C2aLayer(
            self.channels[0],
            self.channels[0],
            n,
            act=act,
            use_rep=True,
        )
        self.csp2 = C2aLayer(
            int(self.channels[1] * 2),
            self.channels[1],
            n,
            act=act,
            use_rep=True,
        )
        self.csp3 = C2aLayer(
            int(self.channels[2] * 2),
            self.channels[2],
            n,
            act=act,
            drop_rate=drop_rate,
            use_rep=True,
        )

        self.drop   = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

        # last_layer
        self.last_layer = nn.Identity() if pp_repeats==0 else C2PPLayer(
            self.channels[2],
            self.channels[2],
            n=pp_repeats,
            act=act,
            drop_rate=drop_rate,
        )

        # stn
        if use_stn:
            self.stn = STNLayer(vox_xyz_size[0]//8, vox_xyz_size[2]//8, act=act)
        else:
            self.stn = nn.Identity()


    def forward(self, inputs):
        """
        Args:
            inputs: backbone output.
        Returns:
            Tuple[Tensor]: FPN feature.
        """
        features = [inputs[f] for f in self.in_features]
        [x1, x2, x3] = features
        outputs = inputs

        x = x1
        x = self.csp1(x) # s8
        outputs[self.out_features[0]] = x # s8

        x = self.model.maxpool(x) # s16
        x = self.model.layer1(x)
        x = torch.cat([x, x2], 1)
        x = self.csp2(x)
        outputs[self.out_features[1]] = x # s16

        x = self.model.layer2(x) # s32
        x = torch.cat([x, x3], 1)
        x = self.csp3(x)
        if self.training:
            x = self.drop(x)
        x = self.last_layer(x)
        x = self.stn(x)
        outputs[self.out_features[2]] = x # s32

        return outputs
