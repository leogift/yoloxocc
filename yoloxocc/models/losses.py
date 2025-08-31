#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss

# focal loss
class BinarySegmentationLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        assert reduction in [ None, 'none', 'mean', 'sum']
        self.reduction = reduction
        self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, pred, target):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"

        target = target.type_as(pred)

        loss = self.loss_fn(pred, target)
        return loss + sigmoid_focal_loss(pred, target, self.alpha, self.gamma, self.reduction)

# uncertainty loss
class UncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, loss, factor=1):
        loss = factor*torch.exp(-self.weight)*loss + self.weight
        return (F.relu(loss) + F.sigmoid(loss)) * 0.5

# heatmap loss
class HeatmapLoss(nn.Module):
    def __init__(self, loss_fn=BinarySegmentationLoss(reduction="mean")):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, target, debug=False):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"
        
        target = target.type_as(pred)

        B, C, H, W = pred.shape
        loss = 0
        for c in range(C):
            loss += self.loss_fn(pred[:, c], target[:, c])

            if debug: 
                import cv2
                import numpy as np
                import os
                if not os.path.exists("debug"):
                    os.mkdir("debug")
                heatmap_pred = (pred[0, c].detach().cpu().numpy()*255).astype(np.uint8)
                cv2.imwrite(f"debug/heatmap_pred_{c}.png", heatmap_pred)
                heatmap_target = (target[0, c].cpu().numpy()*255).astype(np.uint8)
                cv2.imwrite(f"debug/heatmap_target_{c}.png", heatmap_target)
                
        return loss / C

# diff loss
class DiffLoss(nn.Module):
    def __init__(self, reduction="none"):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(reduction=reduction)

    def forward(self, pred, target):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"

        target = target.type_as(pred)

        return torch.exp(self.loss_fn(pred, target)) - 0.996

# balance loss
class BalanceLoss(nn.Module):
    def __init__(self, loss_fn=DiffLoss(reduction="none")):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, target):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"

        target = target.type_as(pred)

        loss = self.loss_fn(pred, target)

        g_mask = target.gt(0.5).float()
        l_mask = target.lt(0.5).float()

        g_loss = (loss * g_mask).sum() / g_mask.sum().clamp(1e-7)
        l_loss = (loss * l_mask).sum() / l_mask.sum().clamp(1e-7)

        loss = (g_loss + l_loss) * 0.5

        return loss

# dice loss
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    '''
    Args:
        pred: tensor in Integer
        target: tensor in Integer
    '''
    def forward(self, pred, target):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"

        target = target.type_as(pred)

        loss = 1 - 2 * (pred * target).sum() / (pred + target).sum().clamp(1e-7)

        return loss
