#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn

from yoloxocc.utils import VoxUtil, basic
from yoloxocc.models.network_blocks import C2aLayer

import random

class IPMTrans(nn.Module):
    def __init__(self,
        in_features=["backbone3", "backbone4", "backbone5"],
        channels=[256, 512, 1024],
        out_features=["trans3", "trans4", "trans5"],
        act="silu",
        layer_type=C2aLayer,
        n=2,
        vox_xyz_size=[128, 4, 128],
        world_xyz_bounds=[-32, 32, -2, 2, -32, 32],
    ):
        super().__init__()
        self.in_features = in_features
        self.channels = channels
        self.out_features = out_features

        # voxel
        self.vox_xyz_size = vox_xyz_size

        # strided vox
        self.vox_s2 = VoxUtil(
            [vox_xyz_size[0]//2, vox_xyz_size[1], vox_xyz_size[2]//2],
            world_xyz_bounds=world_xyz_bounds,
        )
        self.vox_s4 = VoxUtil(
            [vox_xyz_size[0]//4, vox_xyz_size[1], vox_xyz_size[2]//4],
            world_xyz_bounds=world_xyz_bounds,
        )
        self.vox_s8 = VoxUtil(
            [vox_xyz_size[0]//8, vox_xyz_size[1], vox_xyz_size[2]//8],
            world_xyz_bounds=world_xyz_bounds,
        )

        # gridsample 在gpu上速度快，但是需要额外op支持
        # remapping 在npu上速度快，但是需要较大的交换矩阵
        self.perspective_mode = None # in ["gridsample", "remapping"]

        # bev compress
        self.csp_s2 = layer_type(
            channels[0]*vox_xyz_size[1],
            channels[0],
            n,
            act=act,
        )
        self.csp_s4 = layer_type(
            channels[1]*vox_xyz_size[1],
            channels[1],
            n,
            act=act,
        )
        self.csp_s8 = layer_type(
            channels[2]*vox_xyz_size[1],
            channels[2],
            n,
            act=act,
        )


    def forward(self, inputs):
        """
        Args:
            inputs: frontview output.
        Returns:
            Tuple[Tensor]: bev feature.
        """
        features = [inputs[f] for f in self.in_features]
        [x3, x4, x5] = features

        # BS, C, H, W
        BS  = x3.shape[0]
        B = inputs["B"]

        outputs = inputs

        # 若未指定 perspective mode，则随机选择
        if self.perspective_mode is None:
            self.perspective_mode = random.choice(["gridsample", "remapping"])

        assert self.perspective_mode in ["gridsample", "remapping"]
        if self.perspective_mode == "gridsample":
            perspective_Pix2Vox_s2 = self.vox_s2.gridsample_Pix2Vox
            perspective_Pix2Vox_s4 = self.vox_s4.gridsample_Pix2Vox
            perspective_Pix2Vox_s8 = self.vox_s8.gridsample_Pix2Vox
        elif self.perspective_mode == "remapping":
            perspective_Pix2Vox_s2 = self.vox_s2.remapping_Pix2Vox
            perspective_Pix2Vox_s4 = self.vox_s4.remapping_Pix2Vox
            perspective_Pix2Vox_s8 = self.vox_s8.remapping_Pix2Vox

        # 从前视图到voxel, BS,C*Y,Z*X
        valid_vox_s2, vox_s2 = perspective_Pix2Vox_s2(x3)
        valid_vox_s4, vox_s4 = perspective_Pix2Vox_s4(x4)
        valid_vox_s8, vox_s8 = perspective_Pix2Vox_s8(x5)

        # unpack camera dim
        __u = lambda x: basic.unpack_seqdim(x, B)

        # BS x CxY x ZxX -> B x S x CxY x ZxX -> B x C*Y x ZxX
        valid_vox_mask, _ = __u(valid_vox_s2).max(dim=1)
        outputs["valid_vox_mask"] = valid_vox_mask.float()

        # 从前视图到voxel
        mask_vox_s2 = valid_vox_s2.reshape(BS, 1, self.vox_xyz_size[1], -1).float() # BS, 1, Y, Z*X
        mask_vox_s2 = mask_vox_s2.repeat(1, self.channels[0], 1, 1) # BS, C, Y, Z*X
        mask_vox_s2 = mask_vox_s2.reshape(BS, self.channels[0]*self.vox_xyz_size[1], -1) # BS, C*Y, Z*X
        vox_s2 = vox_s2 * mask_vox_s2        
        vox_s2, _ = __u(vox_s2).max(dim=1) # B, C*Y, Z*X
        vox_s2 = vox_s2.reshape(B, -1, self.vox_xyz_size[2]//2, self.vox_xyz_size[0]//2) # B, C*Y, Z, X

        mask_vox_s4 = valid_vox_s4.reshape(BS, 1, self.vox_xyz_size[1], -1).float() # BS, 1, Y, Z*X
        mask_vox_s4 = mask_vox_s4.repeat(1, self.channels[1], 1, 1)
        mask_vox_s4 = mask_vox_s4.reshape(BS, self.channels[1]*self.vox_xyz_size[1], -1)
        vox_s4 = vox_s4 * mask_vox_s4
        vox_s4, _ = __u(vox_s4).max(dim=1)
        vox_s4 = vox_s4.reshape(B, -1, self.vox_xyz_size[2]//4, self.vox_xyz_size[0]//4)

        mask_vox_s8 = valid_vox_s8.reshape(BS, 1, self.vox_xyz_size[1], -1).float() # BS, 1, Y, Z*X
        mask_vox_s8 = mask_vox_s8.repeat(1, self.channels[2], 1, 1)
        mask_vox_s8 = mask_vox_s8.reshape(BS, self.channels[2]*self.vox_xyz_size[1], -1)
        vox_s8 = vox_s8 * mask_vox_s8
        vox_s8, _ = __u(vox_s8).max(dim=1)
        vox_s8 = vox_s8.reshape(B, -1, self.vox_xyz_size[2]//8, self.vox_xyz_size[0]//8)

        # 从voxel到bev特征
        bev_s2 = self.csp_s2(vox_s2)
        bev_s4 = self.csp_s4(vox_s4)
        bev_s8 = self.csp_s8(vox_s8)

        outputs[self.out_features[0]] = bev_s2
        outputs[self.out_features[1]] = bev_s4
        outputs[self.out_features[2]] = bev_s8

        return outputs
