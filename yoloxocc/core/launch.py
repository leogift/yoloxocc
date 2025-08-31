#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/launch.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Megvii, Inc. and its affiliates.

from datetime import timedelta
from loguru import logger

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_descriptor')

import yoloxocc.utils.dist as comm

__all__ = ["launch"]


DEFAULT_TIMEOUT = timedelta(minutes=30)


def _find_free_port():
    """
    Find an available port of current machine / node.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch(
    main_func,
    num_gpus,
    backend="nccl",
    args=(),
):
    """
    Args:
        main_func: a function that will be called by `main_func(*args)`
        args (tuple): arguments passed to main_func
    """
    if num_gpus > 1:
        port = _find_free_port()
        dist_url = f"tcp://localhost:{port}"
        start_method = "spawn"

        mp.start_processes(
            _distributed_worker,
            nprocs=num_gpus,
            args=(
                main_func,
                num_gpus,
                backend,
                dist_url,
                args,
            ),
            daemon=False,
            start_method=start_method,
        )
    else:
        main_func(*args)


def _distributed_worker(
    rank, # auto-assigned by torch.multiprocessing.spawn
    main_func,
    num_gpus,
    backend,
    dist_url,
    args,
    timeout=DEFAULT_TIMEOUT,
):
    assert (
        torch.cuda.is_available()
    ), "cuda is not available. Please check your installation."
    logger.info("Rank {} initialization finished.".format(rank))
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=num_gpus,
        rank=rank,
        timeout=timeout,
    )
    
    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    assert num_gpus <= torch.cuda.device_count()
    torch.cuda.set_device(rank)

    main_func(*args)

