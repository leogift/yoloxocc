#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech, Inc. and its affiliates.

from torch.utils.data.dataset import Dataset

class DatasetBase(Dataset):
    """
    Custom dataset class.
    """
    def pull_item(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.pull_item(index, False, None)
        elif isinstance(index, (tuple, list)):
            return self.pull_item(*index)
        else:
            raise NotImplementedError
