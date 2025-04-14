#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler

import itertools
import random

class AugmentBatchSampler(BatchSampler):
    def __init__(self, *args, aug=True, image_size=[288,512], **kwargs):
        super().__init__(*args, **kwargs)
        self.aug = aug
        self.image_size = image_size
        self._random_image_size = image_size

    def random_image_size(self):
        ratio = self.image_size[1] * 1.0 / self.image_size[0]
        min_size = round(self.image_size[0] / 32 * 0.8)
        max_size = round(self.image_size[0] / 32 * 1.2)
        size = random.randint(min_size, max_size)
        self._random_image_size = (int(32 * size), 32 * int(size * ratio))

        return self._random_image_size

    def enable_aug(self):
        self.aug = True

    def disable_aug(self):
        self.aug = False

    def __iter__(self):
        for batch in super().__iter__():
            yield [(idx, self.aug, self._random_image_size) for idx in batch]


class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        rank=0,
        world_size=1,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._rank)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size
