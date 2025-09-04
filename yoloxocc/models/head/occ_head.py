#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from yoloxocc.models.network_blocks import get_activation, BaseConv
from yoloxocc.utils import special_multiples

from yoloxocc.models.losses import HeatmapLoss, BalanceLoss, UncertaintyLoss
from yoloxocc.models.metrics import HeatmapMetric, SimilarityMetric, IOUMetric

class OCCHead(nn.Module):
    def __init__(
        self,
        in_feature="bev_fpn1",
        in_channel=64,
        vox_y=4,
        act="silu",
        simple_reshape=False,
        aux_head=False
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
        """
        super().__init__()
        self.in_feature = in_feature
        self.aux_head = aux_head

        assert vox_y > 0, "vox_y should be greater than 0."
        self.vox_y = vox_y

        # 主输出 upsample
        if not self.aux_head:
            self.upsample = nn.Sequential(*[
                nn.ConvTranspose2d(
                    in_channel, 
                    in_channel, 
                    kernel_size=4, 
                    stride=2, 
                    padding=1,
                    groups=in_channel,
                    bias=True),
                get_activation(act)()
            ]) if simple_reshape==False else nn.Upsample(scale_factor=2, mode="bilinear")

        hidden_channel = special_multiples(in_channel//2)
 
        self.stem = BaseConv(
            in_channel,
            hidden_channel,
            1,
            stride=1,
            act=act,
        )

        # 主输出 conv
        self.occ_convs = nn.Identity() if self.aux_head else nn.Sequential(*[
            BaseConv(
                hidden_channel,
                hidden_channel,
                3,
                stride=1,
                groups=hidden_channel,
                act=act,
            ),
            BaseConv(
                hidden_channel,
                hidden_channel,
                3,
                stride=1,
                groups=hidden_channel,
                act=act,
            ),
        ])

        self.occ_preds = nn.Conv2d(
                hidden_channel, 
                vox_y,
                kernel_size=1, 
                bias=True)
        
        import math
        prior_prob = 1e-1
        for _, m in self.occ_preds.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=1)
                if m.bias is not None:
                    b = m.bias.view(1, -1)
                    b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
                    m.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        # loss functions
        self.bce_loss = HeatmapLoss()
        self.diff_loss = HeatmapLoss(BalanceLoss())

        # uncertainty loss
        self.bce_uncertainty_loss = UncertaintyLoss()
        self.diff_uncertainty_loss = UncertaintyLoss()
		
		# metric
        self.similarity_metric = HeatmapMetric(SimilarityMetric())
        self.iou_metric = HeatmapMetric(IOUMetric())
		
    def forward(self, inputs):
        x = inputs[self.in_feature]
        occ = []

        if self.training:
            if not self.aux_head:
                x = self.upsample(x)
            
            x = self.stem(x)

            if not self.aux_head:
                x = self.occ_convs(x)
    
            occ = self.occ_preds(x)
            
            return occ
        
        else:
            if not self.aux_head:
                x = self.upsample(x)
                x = self.stem(x)
                x = self.occ_convs(x)
                occ = self.occ_preds(x)
                return occ

    # losses
    def get_losses(
        self,
        pred,
        target,
        valid_mask,
        uncertainty=True
    ):
        # B,Y,Z,X
        assert pred.shape[:2] == target.shape[:2] == valid_mask.shape[:2], \
            f"expect {pred.shape[:2]} == {target.shape[:2]} == {valid_mask.shape[:2]}"

        # 对齐
        Zp,Xp = pred.shape[2:]
        _target = F.interpolate(target, (Zp,Xp), mode="bilinear", align_corners=True)
        _valid_mask = F.interpolate(valid_mask, (Zp,Xp), mode="nearest")
        _pred = pred

        # losses
        bce_loss = self.bce_loss(_pred, _target.round(), _valid_mask)
        diff_loss  = self.diff_loss(_pred.sigmoid(), _target, _valid_mask)

        if uncertainty:
            # uncertainty weight
            bce_loss = self.bce_uncertainty_loss(bce_loss, 2)
            diff_loss = self.diff_uncertainty_loss(diff_loss)

        total_loss = bce_loss + diff_loss
        
        losses = {
            "total_loss": total_loss,
            "bce_loss": bce_loss,
            "diff_loss": diff_loss,
        }

        return losses

    # metrics
    def get_metrics(
        self,
        pred,
        target,
        valid_mask,
    ):
        # B,Y,Z,X
        assert pred.shape[:2] == target.shape[:2] == valid_mask.shape[:2], \
            f"expect {pred.shape[:2]} == {target.shape[:2]} == {valid_mask.shape[:2]}"

        # 对齐
        Zp,Xp = pred.shape[2:]
        _target = F.interpolate(target, (Zp,Xp), mode="nearest")
        _valid_mask = F.interpolate(valid_mask, (Zp,Xp), mode="nearest")
        _pred = pred

        similarity = self.similarity_metric(_pred.sigmoid(), _target, _valid_mask, debug=True)
        iou = self.iou_metric(_pred.sigmoid().round(), _target.round(), _valid_mask)

        metrics = {
            "similarity": similarity,
            "iou": iou,
        }

        return metrics
