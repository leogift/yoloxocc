#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yoloxocc.exp import Exp as BaseExp

from exps.yoloxocc_regnet_x_200mf_w16x2x12_v64x4x48 import Exp as BaseExp

class Exp(BaseExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.train_json = "train_tailer.json"
        self.val_json = "val_tailer.json"
