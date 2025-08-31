#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn

# similarity metric
class SimilarityMetric(nn.Module):
    def forward(self, pred, target):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"

        target = target.type_as(pred)

        metric = 1 - (torch.abs(pred - target)).mean()

        return metric

# iou metric
class IOUMetric(nn.Module):
    def forward(self, pred, target):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"

        target = target.type_as(pred)

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection

        metric = intersection / union.clamp(1e-7)

        return metric

# heatmap metric
class HeatmapMetric(nn.Module):
    def __init__(self, metric_fn=SimilarityMetric):
        super().__init__()
        self.metric_fn = metric_fn

    '''
    Args:
        pred: tensor in Integer
        target: tensor in Integer
    '''
    def forward(self, pred, target, debug=False):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"
        
        target = target.type_as(pred)

        B, C, H, W = pred.shape
        metric = 0
        for c in range(C):
            metric += self.metric_fn(pred[:, c], target[:, c])

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
                
        return metric / C
