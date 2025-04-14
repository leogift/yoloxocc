#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech, Inc. and its affiliates.

import torch


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.items = []
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.items = next(self.loader)
        except StopIteration:
            for i in range(len(self.items)):
                self.items[i] = None
            return

        with torch.cuda.stream(self.stream):
            for i in range(len(self.items)):
                self.items[i] = self.items[i].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        items = self.items
        for i in range(len(items)):
            if items[i] is not None:
                items[i].record_stream(torch.cuda.current_stream())
        
        self.preload()
        return items
