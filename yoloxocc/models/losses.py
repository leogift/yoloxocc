#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

# bce loss
class MaskedBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.13), reduction="none")

    def forward(self, logits, target, mask=None):
        assert logits.shape == target.shape, \
            f"expect {logits.shape} == {target.shape}"

        target = target.type_as(logits)
        if mask is None:
            mask = torch.ones_like(logits)
        mask = mask.type_as(logits)

        _loss = self.loss_fn(logits, target)

        loss = (_loss * mask).sum() / mask.sum().clamp(1e-7)

        return loss

# dice loss
class DiceLoss(nn.Module):
    def forward(self, pred, target, mask=None):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"

        target = target.type_as(pred)

        loss = 1 - 2 * (pred * target * mask).sum() / ((pred + target) * mask).sum().clamp(1e-7)

        return loss

# heatmap loss
class HeatmapLoss(nn.Module):
    def __init__(self, loss_fn=MaskedBCEWithLogitsLoss()):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, target, mask=None, channel_weight=None):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"

        target = target.type_as(pred)
        if mask is None:
            mask = torch.ones_like(pred)
        mask = mask.type_as(pred)

        B, C, H, W = pred.shape
        if channel_weight is None:
            channel_weight = [1.0]*C
        channel_weight = torch.tensor(channel_weight).to(pred.device)

        loss = 0
        for c in range(C):
            loss += self.loss_fn(pred[:, c], target[:, c], mask[:, c]) * channel_weight[c]

        return loss / C

# gaussian l1 loss
class GaussianL1Loss(nn.Module):
    def __init__(self, reduction="none"):
        super().__init__()
        assert reduction in [ None, 'none', 'mean', 'sum']
        self.reduction = reduction

    def forward(self, pred, target):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"

        target = target.type_as(pred)

        l1 = torch.abs(pred - target)
        loss = - (1 - l1 + 1e-7).log() * l1

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        
        return loss

# balance loss
class BalanceLoss(nn.Module):
    def __init__(self, loss_fn=nn.L1Loss()):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, target, mask):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"

        target = target.type_as(pred)
        if mask is None:
            mask = torch.ones_like(pred)
        mask = mask.type_as(pred)

        loss = self.loss_fn(pred, target)

        g_mask = target.ge(0.5).float() * mask
        l_mask = target.lt(0.5).float() * mask

        g_loss = (loss * g_mask).sum() / g_mask.sum().clamp(1e-7)
        l_loss = (loss * l_mask).sum() / l_mask.sum().clamp(1e-7)

        loss = (g_loss + l_loss) * 0.5

        return loss

# uncertainty loss
class UncertaintyLoss(nn.Module):
    def __init__(self, factor=1):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.factor = factor

    def forward(self, loss):
        loss = self.factor*torch.exp(-self.weight)*loss + self.weight
        return (F.relu(loss) + F.sigmoid(loss)) * 0.5

# aiou/giou/diou
class IOULoss(nn.Module):
    def __init__(self, reduction="none"):
        super().__init__()
        assert reduction in [ None, 'none', 'mean', 'sum']
        self.reduction = reduction

    def forward(self, pred, target, eps=1e-7):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"

        target = target.type_as(pred)

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        # base iou [0, 1]
        area_i = torch.prod(br - tl, 1) * (tl < br).type(tl.type()).prod(dim=1)
        area_u = torch.prod(pred[:, 2:], 1)+torch.prod(target[:, 2:], 1) - area_i
        iou = area_i / area_u.clamp(eps)

        # aiou [0, 1]
        iou = iou.pow(2)

        # combine bound boxes
        c_tl = torch.min(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        c_br = torch.max(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        # giou [-1, 1]
        area_c = torch.prod(c_br - c_tl, 1)
        iou = iou - (area_c - area_u) / area_c.clamp(eps)

        # diou [-2. 1]
        w_c = (c_br - c_tl)[:, 0]
        h_c = (c_br - c_tl)[:, 1]
        w_d = (pred[:, :2] - target[:, :2])[:, 0]
        h_d = (pred[:, :2] - target[:, :2])[:, 1]
        iou = iou - (w_d ** 2 + h_d ** 2) / (w_c ** 2 + h_c ** 2).clamp(eps)

        # iou loss [0, 3]
        loss = 1.0 - iou

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
