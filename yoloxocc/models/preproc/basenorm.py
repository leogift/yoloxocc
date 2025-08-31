#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.
import torch
from torch import nn

class BaseNorm(nn.Module):
    def __init__(
        self,
        mean = [114.495,114.495,114.495],
        std = [57.63,57.63,57.63],
        trainable = False
    ):
        super().__init__()

        assert len(mean) == len(std)

        self.trainable = trainable
        N = len(mean)
        self.inversed_std = torch.true_divide(1, torch.tensor(std, dtype=torch.float32))
        self.mean = torch.tensor(mean, dtype=torch.float32)

        if self.trainable:
            self.normalize = nn.Conv2d(N, N, kernel_size=1, stride=1, padding=0)
            self.normalize.weight.data = torch.diag(self.inversed_std).view(N, N, 1, 1).requires_grad_(True)
            self.normalize.bias.data = (-self.mean*self.inversed_std).requires_grad_(True)
            self.normalize.requires_grad_(True)

    def forward(self, inputs):
        x = inputs["cameras_image_"]

        outputs = inputs
        
        if self.trainable:
            x = self.normalize(x)
        else:
            x = (x - self.mean.view(1, -1, 1, 1).to(x.device))*self.inversed_std.view(1, -1, 1, 1).to(x.device)

        outputs["input"] = x

        return outputs
