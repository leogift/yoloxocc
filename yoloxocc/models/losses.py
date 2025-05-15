#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="none"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        assert reduction in [ None, 'none', 'mean', 'sum']
        self.reduction = reduction

    '''
    Args:
        pred: tensor without sigmoid
        target: tensor
    '''
    def forward(self, pred, target):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"
        if pred.shape[0] == 0:
            loss = torch.ones([1, target.shape[1:]], device=pred.device) * target.shape[1:]

        else:
            pred_sigmoid = pred.sigmoid()
            target = target.type_as(pred)
            pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
            weight = pt.pow(self.gamma)
            alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
            weight = alpha_factor * weight
            loss = weight * F.binary_cross_entropy_with_logits(
                pred, target, 
                reduction="none")

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss

# uncertainty loss
class UncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, loss, factor=1):
        return factor*torch.exp(-self.weight)*loss + F.relu(self.weight)

# heatmap loss
class HeatmapLoss(nn.Module):
    def __init__(self, loss_fn=nn.BCEWithLogitsLoss(reduction="mean")):
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

# SmoothL1Loss with beta
class SmoothL1Loss(nn.Module):
    def __init__(self, reduction="none", beta=1):
        super().__init__()
        assert reduction in [ None, 'none', 'mean', 'sum']
        self.reduction = reduction
        self.beta = beta

    def forward(self, pred, target):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"
        
        target = target.type_as(pred)

        loss = F.smooth_l1_loss(pred, target, reduction=self.reduction, beta=self.beta) / (1 - 0.5*self.beta)

        return loss

# balance loss
class BalanceLoss(nn.Module):
    def __init__(self, loss_fn=SmoothL1Loss(reduction="none", beta=0.5)):
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

# iou/giou/ciou/diou
class IOULoss(nn.Module):
    def __init__(self, loss_type="fusion", reduction="none"):
        super().__init__()
        assert reduction in [ None, 'none', 'mean', 'sum']
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target, eps=1e-7):
        assert pred.shape == target.shape

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        if target.shape[0]==0:
            loss = torch.ones([1], device=pred.device) * 2

        else:
            tl = torch.max(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            br = torch.min(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )

            area_i = torch.prod(br - tl, 1) * (tl < br).type(tl.type()).prod(dim=1)
            area_u = torch.prod(pred[:, 2:], 1)+torch.prod(target[:, 2:], 1) - area_i
            iou = area_i / area_u.clamp(eps)

            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )

            if self.loss_type in ['giou', 'fusion']:
                area_c = torch.prod(c_br - c_tl, 1)
                iou = iou - (area_c - area_u) / area_c.clamp(eps)
            
            if self.loss_type in ['ciou', 'fusion']:
                c2 = torch.sum((c_tl - c_br)**2, 1) # convex (smallest enclosing box) diagonal squared
                rho2 = torch.sum((pred[:, :2] - target[:, :2])**2, 1) # center distance squared
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(target[:,2]/ (target[:,3].clamp(eps))) - torch.atan(pred[:,2] / (pred[:,3].clamp(eps))), 2)

                alpha = v / (v - iou + 1).clamp(eps)
                iou = iou - (rho2 / c2 + v * alpha)  # CIoU

            if self.loss_type == ['diou', 'fusion']:
                w_c = (c_br - c_tl)[:, 0]
                h_c = (c_br - c_tl)[:, 1]
                w_d = (pred[:, :2] - target[:, :2])[:, 0]
                h_d = (pred[:, :2] - target[:, :2])[:, 1]
                iou = iou - (w_d ** 2 + h_d ** 2) / (w_c ** 2 + h_c ** 2).clamp(eps)

            loss = 1.0 - iou

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

