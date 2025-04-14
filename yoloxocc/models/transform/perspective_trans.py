#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn

from yoloxocc.utils import VoxUtil, basic
from yoloxocc.models.network_blocks import C2aLayer

class PerspectiveTrans(nn.Module):
    def __init__(self,
        in_features=("fpn3", "fpn4", "fpn5"),
        in_channels=[256, 512, 1024],
        out_features=("trans3", "trans4", "trans5"),
        act="silu",
        layer_type=C2aLayer,
        vox_xyz_size=[128, 4, 128],
        world_xyz_bounds=[-32, 32, -2, 2, -32, 32],
        n=2,
    ):
        super().__init__()
        assert len(in_features) == len(in_channels) and len(in_features) == 3
        self.in_features = in_features
        self.in_channels = in_channels
        self.out_features = out_features

        self.csp3 = layer_type(
            in_channels[0]*vox_xyz_size[1],
            in_channels[0],
            n,
            act=act,
        )
        self.csp4 = layer_type(
            in_channels[1]*vox_xyz_size[1],
            in_channels[1],
            n,
            act=act,
        )
        self.csp5 = layer_type(
            in_channels[2]*vox_xyz_size[1],
            in_channels[2],
            n,
            act=act,
        )

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

        self.perspective_mode = "gridsample"
        # self.perspective_mode = "remapping"
        
        # gridsample 在gpu上速度快，但是需要额外op支持
        # remapping 在npu上速度快，但是需要较大的交换矩阵
        assert self.perspective_mode in ["gridsample", "remapping"]
    
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

        outputs = inputs

        if self.perspective_mode == "gridsample":
            perspective_Pix2Vox_s2 = self.vox_s2.gridsample_Pix2Vox
            perspective_Pix2Vox_s4 = self.vox_s4.gridsample_Pix2Vox
            perspective_Pix2Vox_s8 = self.vox_s8.gridsample_Pix2Vox
        elif self.perspective_mode == "remapping":
            perspective_Pix2Vox_s2 = self.vox_s2.remapping_Pix2Vox
            perspective_Pix2Vox_s4 = self.vox_s4.remapping_Pix2Vox
            perspective_Pix2Vox_s8 = self.vox_s8.remapping_Pix2Vox
        else:
            raise ValueError("Unknown perspective method.")

        # 从前视图到voxel, BS,C*Y,Z*X
        valid_vox_s2, vox_s2 = perspective_Pix2Vox_s2(
            x3
        )
        valid_vox_s4, vox_s4 = perspective_Pix2Vox_s4(
            x4
        )
        valid_vox_s8, vox_s8 = perspective_Pix2Vox_s8(
            x5
        )

        # unpack camera dim
        __u = lambda x: basic.unpack_seqdim(x, inputs["B"])

        # B,S,C*Y,Z*X
        mask_vox_s2 = valid_vox_s2.reshape(BS, 1, self.vox_xyz_size[1], -1).float() # BS, 1, Y, Z*X
        mask_vox_s2 = mask_vox_s2.repeat(1, self.in_channels[0], 1, 1) # BS, C, Y, Z*X
        mask_vox_s2 = mask_vox_s2.reshape(BS, self.in_channels[0]*self.vox_xyz_size[1], -1) # BS, C*Y, Z*X
        vox_s2 = vox_s2 * mask_vox_s2
        bev_s2 = torch.sum(__u(vox_s2), dim=1) # B, C*Y, Z*X
        mask_bev_s2 = torch.sum(__u(mask_vox_s2), dim=1) # B, C*Y, Z*X
        mask_bev_s2 = torch.relu(mask_bev_s2-1)+1
        bev_s2 = bev_s2 / mask_bev_s2
        bev_s2 = bev_s2.reshape(inputs["B"], -1, self.vox_xyz_size[2]//2, self.vox_xyz_size[0]//2) # B, C*Y, Z, X

        mask_vox_s4 = valid_vox_s4.reshape(BS, 1, self.vox_xyz_size[1], -1).float() # BS, 1, Y, Z*X
        mask_vox_s4 = mask_vox_s4.repeat(1, self.in_channels[1], 1, 1)
        mask_vox_s4 = mask_vox_s4.reshape(BS, self.in_channels[1]*self.vox_xyz_size[1], -1)
        vox_s4 = vox_s4 * mask_vox_s4
        bev_s4 = torch.sum(__u(vox_s4), dim=1)
        mask_bev_s4 = torch.sum(__u(mask_vox_s4), dim=1)
        mask_bev_s4 = torch.relu(mask_bev_s4-1)+1
        bev_s4 = bev_s4 / mask_bev_s4
        bev_s4 = bev_s4.reshape(inputs["B"], -1, self.vox_xyz_size[2]//4, self.vox_xyz_size[0]//4)
        
        mask_vox_s8 = valid_vox_s8.reshape(BS, 1, self.vox_xyz_size[1], -1).float() # BS, 1, Y, Z*X
        mask_vox_s8 = mask_vox_s8.repeat(1, self.in_channels[2], 1, 1)
        mask_vox_s8 = mask_vox_s8.reshape(BS, self.in_channels[2]*self.vox_xyz_size[1], -1)
        vox_s8 = vox_s8 * mask_vox_s8
        bev_s8 = torch.sum(__u(vox_s8), dim=1)
        mask_bev_s8 = torch.sum(__u(mask_vox_s8), dim=1)
        mask_bev_s8 = torch.relu(mask_bev_s8-1)+1
        bev_s8 = bev_s8 / mask_bev_s8
        bev_s8 = bev_s8.reshape(inputs["B"], -1, self.vox_xyz_size[2]//8, self.vox_xyz_size[0]//8)

        # 从voxel到bev特征
        bev_s2 = self.csp3(bev_s2)
        bev_s4 = self.csp4(bev_s4)
        bev_s8 = self.csp5(bev_s8)
        outputs[self.out_features[0]] = bev_s2
        outputs[self.out_features[1]] = bev_s4
        outputs[self.out_features[2]] = bev_s8

        return outputs
