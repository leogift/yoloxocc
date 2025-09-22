#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import contextlib
from copy import deepcopy
from typing import Sequence

import torch
import torch.nn as nn
from thop import profile

__all__ = [
    "fuse_conv_and_bn",
    "optimize_model",
    "get_model_info",
    "freeze_module",
    "unfreeze_module",
    "adjust_status",
]


def get_model_info(model: nn.Module, \
                   S,image_size) -> str:
    cameras_image = torch.randn(1, S, 3, *image_size).to(next(model.parameters()).device)
    cameras_extrin = torch.randn(1, S, 4, 4).to(next(model.parameters()).device)
    cameras_intrin = torch.randn(1, S, 3, 3).to(next(model.parameters()).device)

    lidars_points = torch.randn(1, 1, 1, 3).to(next(model.parameters()).device)
    lidars_extrin = torch.randn(1, 1, 4, 4).to(next(model.parameters()).device)

    cameras_annos = torch.randn(1, S, 50, 5).to(next(model.parameters()).device)
    lidars_annos = torch.randn(1, 1, 100, 8).to(next(model.parameters()).device)

    model.train()
    flops, params = profile(deepcopy(model), inputs=(cameras_image,cameras_extrin,cameras_intrin,\
                                                     lidars_points,lidars_extrin,
                                                     cameras_annos,lidars_annos), verbose=False)

    params /= 1e6
    flops /= 1e9
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info


def fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fuse convolution and batchnorm layers.
    check more info on https://tehnokv.com/posts/fusing-batchnorm-and-conv/

    Args:
        conv (nn.Conv2d): convolution to fuse.
        bn (nn.BatchNorm2d): batchnorm to fuse.

    Returns:
        nn.Conv2d: fused convolution behaves the same as the input conv and bn.
    """
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def optimize_model(model: nn.Module) -> nn.Module:
    """
    Optimize:
        fuse conv and bn in model
        repconv in model

    Args:
        model (nn.Module): model to fuse

    Returns:
        nn.Module: Optimized model
    """
    from yoloxocc.models.network_blocks import BaseConv, RepSConv3x3, RepBottleneck

    for m in model.modules():
        if type(m) is BaseConv and hasattr(m, "bn"):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, "bn")  # remove batchnorm
            m.forward = m.fuseforward  # update forward

    for m in model.modules():
        if type(m) in [RepSConv3x3] and not hasattr(m, "RepSConv3x3"):
            m.switch_to_deploy()
            m.forward = m.fuseforward  # update forward

    return model


def freeze_module(module: nn.Module, name=None) -> nn.Module:
    """freeze module inplace

    Args:
        module (nn.Module): module to freeze.
        name (str, optional): name to freeze. If not given, freeze the whole module.

    Examples:
        freeze the backbone of model
        >>> freeze_moudle(model.backbone)

        or freeze the backbone of model by name
        >>> freeze_moudle(model, name="backbone")
    """
    for param_name, parameter in module.named_parameters():
        if name is None or name in param_name:
            parameter.requires_grad = False

    # ensure module like BN and dropout are freezed
    for module_name, sub_module in module.named_modules():
        # actually there are no needs to call eval for every single sub_module
        if name is None or name in module_name:
            sub_module.eval()

    return module


def unfreeze_module(module: nn.Module, name=None) -> nn.Module:
    """unfreeze module inplace

    Args:
        module (nn.Module): module to unfreeze.
        name (str, optional): name to unfreeze. If not given, unfreeze the whole module.

    Examples:
        unfreeze the backbone of model
        >>> unfreeze_moudle(model.backbone)

        or unfreeze the backbone of model by name
        >>> unfreeze_moudle(model, name="backbone")
    """
    for param_name, parameter in module.named_parameters():
        if name is None or name in param_name:
            parameter.requires_grad = True

    # ensure module like BN and dropout are freezed
    for module_name, sub_module in module.named_modules():
        # actually there are no needs to call train for every single sub_module
        if name is None or name in module_name:
            sub_module.train()

    return module


@contextlib.contextmanager
def adjust_status(module: nn.Module, training: bool = False) -> nn.Module:
    """Adjust module to training/eval mode temporarily.

    Args:
        module (nn.Module): module to adjust status.
        training (bool): training mode to set. True for train mode, False fro eval mode.

    Examples:
        >>> with adjust_status(model, training=False):
        ...     model(data)
    """
    status = {}

    def backup_status(module):
        for m in module.modules():
            # save prev status to dict
            status[m] = m.training
            m.training = training

    def recover_status(module):
        for m in module.modules():
            # recover prev status from dict
            m.training = status.pop(m)

    backup_status(module)
    yield module
    recover_status(module)
