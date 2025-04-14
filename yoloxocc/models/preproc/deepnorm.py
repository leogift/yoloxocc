#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.
import torch
from torch import nn

class DeepNorm(nn.Module):
    def __init__(
        self,
        dim = 3,
    ):
        super().__init__()

        self.norm = nn.BatchNorm2d(dim)

    def forward(self, inputs):
        x = inputs["input"]

        outputs = inputs

        x = self.norm(x)

        outputs["input"] = x

        return outputs
