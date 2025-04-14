#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from yoloxocc.models.network_blocks import CrossTransformer

import numpy as np

import torchvision.transforms as transforms

class Temporal(nn.Module):
    def __init__(self,
                in_feature="bev_backbone5",
                in_channel=1024,
                act="silu",
                drop_rate=0.2,
                temporal_prob=0.8,
            ):
        super().__init__()
        self.in_feature = in_feature
        self.in_channel = in_channel

        self.temporal_transformer = CrossTransformer(
            in_channel,
            heads=4,
            act=act,
            drop_rate=drop_rate
        )
        self.temporal_prob = temporal_prob
    
    def forward(self, inputs):
        x = inputs[self.in_feature]
        temporal_feature = inputs["temporal_feature"]

        outputs = inputs
        
        if temporal_feature is None:
            temporal_feature = torch.zeros_like(x)
            if self.training:
                if np.random.random() < self.temporal_prob:
                    # temporal_prob的概率使用假的历史时间特征
                    with torch.no_grad():
                        # 时序多次
                        loop = np.random.randint(1, 3)
                        for _ in range(loop):
                            # 增强
                            if np.random.random() < 2**(-loop):
                                Z, X = temporal_feature.shape[-2:]
                                angle = np.random.uniform(-15, 15)
                                translate = [np.random.uniform(-0.2, 0.2)*X, np.random.uniform(-0.2, 0.2)*Z]
                                shear = np.random.uniform(-5, 5)
                                interpolation = np.random.choice([
                                    transforms.InterpolationMode.NEAREST, 
                                    transforms.InterpolationMode.BILINEAR
                                ])
                                temporal_feature = transforms.functional.affine(
                                    temporal_feature,
                                    angle=angle, # 是角度而非弧度；如果在以左下角为原点的坐标系中考虑，则逆时针旋转为正；如果在以左上角为原点的坐标系中考虑，则顺时针旋转为正；两个坐标系下图片上下相反
                                    translate=translate, # 是整数而非浮点数，允许出现负值，是一个[x, y]的列表形式， 其中x轴对应w，y轴对应h
                                    scale=1, # 浮点数，中心缩放尺度
                                    shear=shear, # 浮点数或者二维浮点数列表，切变度数，浮点数时是沿x轴切变，二维浮点数列表时先沿x轴切变，然后沿y轴切变
                                    interpolation=interpolation, # 二维线性差值，默认是最近邻差值
                                    fill=0 # 空白区域所填充的数值
                                )
                                temporal_feature = F.dropout(temporal_feature, p=0.2)

                            temporal_feature = self.temporal_transformer(x, temporal_feature)

        temporal_feature = self.temporal_transformer(x, temporal_feature)
        outputs[self.in_feature] = temporal_feature
        outputs["temporal_feature"] = temporal_feature

        return outputs
