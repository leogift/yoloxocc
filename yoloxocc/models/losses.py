#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction="none", pos_weight=torch.Tensor([1])):
        super().__init__()
        assert reduction in [ None, 'none', 'mean', 'sum']
        self.reduction = reduction
        self.pos_weight = pos_weight

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
            loss = F.binary_cross_entropy_with_logits(
                pred, target, 
                pos_weight=self.pos_weight.to(pred.device),
                reduction="none")

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss


# focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="none", pos_weight=torch.Tensor([1])):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        assert reduction in [ None, 'none', 'mean', 'sum']
        self.reduction = reduction
        self.pos_weight = pos_weight

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
                pos_weight=self.pos_weight.to(pred.device),
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
    def __init__(self, loss_fn=nn.BCEWithLogitsLoss()):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, target, debug=False):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"
        
        target = target.type_as(pred)

        B, C, H, W = pred.shape
        loss = 0
        for c in range(C):
            _loss = self.loss_fn(pred[:, c], target[:, c])
            loss += _loss.mean()

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

# balance loss
class BalanceLoss(nn.Module):
    def __init__(self, loss_fn=nn.L1Loss):
        super().__init__()
        self.loss_fn = loss_fn(reduction="none")

    def forward(self, pred, target):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"
        
        target = target.type_as(pred)

        loss = self.loss_fn(pred, target)

        g_mask = target.gt(0.5).float()
        l_mask = target.lt(0.5).float()

        g_loss = (loss * g_mask).sum() / torch.clamp(g_mask.sum(), 1e-8) + (loss * g_mask).max() * 0.1
        l_loss = (loss * l_mask).sum() / torch.clamp(l_mask.sum(), 1e-8) + (loss * l_mask).max() * 0.1

        loss = (g_loss + l_loss) * 0.5

        return loss

# dice loss
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"

        target = target.type_as(pred)

        loss = 1 - 2 * (pred * target).sum() / ((pred + target).sum() + 1e-8)

        return loss

# aiou/giou/ciou/diou
class IOULoss(nn.Module):
    def __init__(self, loss_type="fusion", reduction="none"):
        super().__init__()
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
            area_u = torch.prod(pred[:, 2:], 1)+torch.prod(target[:, 2:], 1)-area_i
            iou = area_i / area_u.clamp(eps)

            if self.loss_type in ['iou', 'fusion']:
                iou = iou**2

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

# oksloss: l2loss+bceloss
class OKSLoss(nn.Module):
    def __init__(self, num_kpts, kpts_weight=None, reduction="none"):
        super().__init__()
        self.num_kpts = num_kpts
        kpts_weight = torch.tensor(kpts_weight) if kpts_weight is not None and len(kpts_weight)==num_kpts else torch.tensor([1]*num_kpts)
        kpts_weight = torch.clip(kpts_weight, 0.5, 4.0)
        self.sigmas = torch.tensor([1/num_kpts]*num_kpts) / kpts_weight
        self.reduction = reduction
        self.dist_loss = nn.MSELoss(reduction="none")
        self.conf_loss = FocalLoss(reduction="none")

    def forward(self, kpts_pred, kpts_conf_pred, \
                kpts_target, kpts_conf_target, bbox_targets):
        sigmas = self.sigmas.to(device=kpts_pred.device)

        # OKS based loss
        if kpts_target.shape[0]==0:
            loss_kpts = torch.ones([1, kpts_target.shape[1]], device=kpts_target.device)
        else:
            dist = self.dist_loss(kpts_pred[:, 0::2], kpts_target[:, 0::2]) + self.dist_loss(kpts_pred[:, 1::2], kpts_target[:, 1::2])
            bbox_area = torch.prod(bbox_targets[:, -2:], dim=1, keepdim=True)  # scale derived from bbox gt: w*h
            kpts_loss_factor = (torch.sum(kpts_conf_target != 0) + torch.sum(kpts_conf_target == 0)) / torch.clip(torch.sum(kpts_conf_target != 0), 1e-9)
            oks = torch.exp(-dist / torch.clip(2 * bbox_area * (sigmas**2), 1e-9))
            loss_kpts = kpts_loss_factor * ((1 - oks) * kpts_conf_target)
        loss_kpts = loss_kpts.mean(axis=1)

        # confidence loss
        if kpts_conf_target.shape[0]==0:
            loss_kpts_conf = torch.ones([1, kpts_conf_target.shape[1]], device=kpts_conf_target.device) * kpts_conf_target.shape[1]
        else:
            loss_kpts_conf = self.conf_loss(kpts_conf_pred, kpts_conf_target)
        loss_kpts_conf = loss_kpts_conf.mean(axis=1)

        if self.reduction == "mean":
            loss_kpts = loss_kpts.mean()
            loss_kpts_conf = loss_kpts_conf.mean()
        elif self.reduction == "sum":
            loss_kpts = loss_kpts.sum()
            loss_kpts_conf = loss_kpts_conf.sum()

        return loss_kpts, loss_kpts_conf

