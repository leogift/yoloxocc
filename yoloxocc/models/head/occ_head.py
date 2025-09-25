#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from yoloxocc.models.network_blocks import get_activation, BaseConv, SeparableConv
from yoloxocc.utils import special_multiples, initialize_regression_weights

from yoloxocc.models.losses import MaskedBCEWithLogitsLoss, DiceLoss, HeatmapLoss, GaussianL1Loss, BalanceLoss, UncertaintyLoss
from yoloxocc.models.metrics import HeatmapMetric, SimilarityMetric, IOUMetric

class OCCHead(nn.Module):
    def __init__(
        self,
        in_feature="bev_fpn1",
        in_channel=64,
        act="silu",
        n=2,
        vox_y=4,
        vox_y_weight=None,
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
        if vox_y_weight is not None:
            assert len(vox_y_weight) == vox_y, "vox_y_weight should be equal to vox_y."
        self.vox_y_weight = vox_y_weight

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
            SeparableConv(
                hidden_channel,
                hidden_channel,
                3,
                stride=1,
                act=act,
            )
            for _n in range(n)
        ])
        # 占用格子
        self.occ_pred = nn.Conv2d(
            hidden_channel, 
            vox_y,
            kernel_size=1, 
            bias=True)
        
        initialize_regression_weights(self.occ_pred)

        # loss functions
        self.bce_loss = HeatmapLoss(MaskedBCEWithLogitsLoss())
        self.dice_loss = DiceLoss()
        self.center_loss = BalanceLoss(loss_fn=GaussianL1Loss(reduction="none"))
        # uncertainty loss
        if not self.aux_head:
            self.bce_uncertainty_loss = UncertaintyLoss(factor=2.0)
            self.dice_uncertainty_loss = UncertaintyLoss(factor=1.0)
            self.center_uncertainty_loss = UncertaintyLoss(factor=1.0)
		
		# metric
        self.similarity_metric = HeatmapMetric(SimilarityMetric())
        self.iou_metric = HeatmapMetric(IOUMetric())

    def forward(self, inputs):
        x = inputs[self.in_feature]
        occ = []

        if not self.aux_head:
            x = self.upsample(x)
        
        x = self.stem(x)

        if not self.aux_head:
            x = self.occ_convs(x)

        occ = self.occ_pred(x)
        
        if self.training:
            return occ
        
        elif not self.aux_head:
            return occ.sigmoid()

    # losses
    def get_losses(
        self,
        preds,
        targets,
        valid_mask,
        uncertainty=True
    ):
        # B,Y,Z,X
        assert preds.shape[:2] == targets.shape[:2] == valid_mask.shape[:2], \
            f"expect {preds.shape[:2]} == {targets.shape[:2]} == {valid_mask.shape[:2]}"

        # 对齐
        Zp,Xp = preds.shape[2:]
        _target = F.interpolate(targets, (Zp,Xp), mode="bilinear", align_corners=True)
        _valid_mask = F.interpolate(valid_mask, (Zp,Xp), mode="nearest")
        _pred = preds

        # losses
        with torch.cuda.amp.autocast(enabled=False):
            bce_loss = self.bce_loss(_pred, _target.round(), mask=_valid_mask, channel_weight=self.vox_y_weight)
            dice_loss = self.dice_loss(_pred.sigmoid().round(), _target.round(), mask=_valid_mask)
            center_loss  = self.center_loss(_pred.sigmoid(), _target, mask=_valid_mask)

            if uncertainty:
                bce_loss = self.bce_uncertainty_loss(bce_loss)
                dice_loss = self.dice_uncertainty_loss(dice_loss)
                center_loss = self.center_uncertainty_loss(center_loss)

            total_loss = bce_loss + dice_loss + center_loss

        assert total_loss==total_loss, f"NaN total_loss {total_loss}={bce_loss}+{dice_loss}+{center_loss}"

        losses = {
            "total_loss": total_loss,
            "bce_loss": bce_loss,
            "dice_loss": dice_loss,
            "center_loss": center_loss,
        }

        return losses

    # metrics
    @torch.no_grad()
    def get_metrics(
        self,
        preds,
        targets,
        valid_mask,
    ):
        # B,Y,Z,X
        assert preds.shape[:2] == targets.shape[:2] == valid_mask.shape[:2], \
            f"expect {preds.shape[:2]} == {targets.shape[:2]} == {valid_mask.shape[:2]}"

        # 对齐
        Zp,Xp = preds.shape[2:]
        _target = F.interpolate(targets, (Zp,Xp), mode="bilinear", align_corners=True)
        _valid_mask = F.interpolate(valid_mask, (Zp,Xp), mode="nearest")
        _pred = preds

        with torch.cuda.amp.autocast(enabled=False):
            similarity = self.similarity_metric(_pred, _target, _valid_mask, debug=True)
            iou = self.iou_metric(_pred.round(), _target.round(), _valid_mask)

        assert similarity==similarity, f"NaN similarity {similarity}"
        assert iou==iou, f"NaN iou {iou}"

        metrics = {
            "similarity": similarity,
            "iou": iou,
        }

        return metrics
