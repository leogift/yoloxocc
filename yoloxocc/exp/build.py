#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import importlib
import os
import sys


def get_exp(exp_file):
    try:
        sys.path.append(os.path.dirname(exp_file))
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp()
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp

