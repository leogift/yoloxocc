#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from yoloxocc.utils import basic, geom, VoxUtil

import numpy as np
import cv2

# 模型框架
class YOLOXOCC(nn.Module):
    """
    YOLOXOCC model module.
    The network returns loss values during training
    and output results during test.
    """

    def __init__(self, 
                preproc=None, # 数据预处理 basenorm/deepnorm
                backbone=None, # backbone
                neck=None, # fpn
                transform=None, # remapping/perspective
                bev_augment=None,
                bev_backbone=None,
                bev_neck=None, # bev fpn
                occ_head=None, # occ head
                aux_occ_head_list=[], # 辅助occ head
                world_xyz_bounds = [-32, 32, -2, 2, -32, 32], # 世界坐标范围 单位m
                vox_xyz_size=[128, 4, 128], # 体素坐标大小 单位格子
                use_gaussian_mask=False, # 是否使用gaussian mask
                ego_dimention=[1.4, 0.8, 1.5], # ego长宽高 单位m
                dimentions=[[1.4, 0.8, 1.5]], # 目标长宽高 单位m
            ):
        super().__init__()

        # preproc
        self.preproc = nn.Identity() if preproc is None else preproc
        # backbone
        self.backbone = nn.Identity() if backbone is None else backbone
        # neck: fpn
        self.neck = nn.Identity() if neck is None else neck
        # front view transform to bird's eye view
        self.transform = nn.Identity() if transform is None else transform
        self.bev_augment = bev_augment
        
        # bev backbone
        self.bev_backbone = nn.Identity() if bev_backbone is None else bev_backbone
        # bev neck: fpn
        self.bev_neck = nn.Identity() if transform is None else bev_neck

        # occ head
        self.occ_head = nn.Identity() if transform is None else occ_head
        # aux occ head
        self.aux_occ_head_list = nn.ModuleList()
        for aux_occ_head in aux_occ_head_list:
            self.aux_occ_head_list.append(aux_occ_head)

        # 预制grid
        self.grid3d_s2 = basic.cloudgrid3d(1, vox_xyz_size[0]//2, vox_xyz_size[1], vox_xyz_size[2]//2)
        self.grid3d_s4 = basic.cloudgrid3d(1, vox_xyz_size[0]//4, vox_xyz_size[1], vox_xyz_size[2]//4)
        self.grid3d_s8 = basic.cloudgrid3d(1, vox_xyz_size[0]//8, vox_xyz_size[1], vox_xyz_size[2]//8)

        # strided vox
        self.vox = VoxUtil(
                [vox_xyz_size[0], vox_xyz_size[1], vox_xyz_size[2]],
                world_xyz_bounds=world_xyz_bounds,
            )

        # gaussian mask
        if use_gaussian_mask:
            self.gaussian_mask = self.get_gaussian_mask(world_xyz_bounds, vox_xyz_size,
                                        ego_dimention=ego_dimention)
        else:
            self.gaussian_mask = torch.ones((1, 1, vox_xyz_size[2], vox_xyz_size[0]), dtype=torch.float32)

        self.dimentions = np.array(dimentions)

    # gaussian mask for voxel
    def get_gaussian_mask(self,
                   world_xyz_bounds,
                   vox_xyz_size,
                   ego_dimention=[1.0,1.0,1.0],
            ):
        world_xmin, world_xmax, world_ymin, world_ymax, world_zmin, world_zmax = world_xyz_bounds
        vox_x, vox_y, vox_z = vox_xyz_size
        max_vox_zx = max(vox_z, vox_x)
        vox_xmin = vox_x/(world_xmax-world_xmin)*world_xmin
        vox_zmin = vox_z/(world_zmax-world_zmin)*world_zmin
        ksize = max_vox_zx*2+1
        sigma = max_vox_zx
        _gaussian_mask = cv2.getGaussianKernel(ksize, sigma) # 边缘 0.5
        _gaussian_mask = (_gaussian_mask.dot(_gaussian_mask.T)) # 2d gaussian
        _gaussian_mask = (_gaussian_mask-_gaussian_mask.min()) / (_gaussian_mask.max()-_gaussian_mask.min())
        mask_z, mask_x = _gaussian_mask.shape
        # 去掉ego灯下黑
        ego_mask_vox_x = vox_x * ego_dimention[1]/(world_xmax-world_xmin) # ego X size
        ego_mask_vox_z = vox_z * ego_dimention[0]/(world_zmax-world_zmin) # ego Z size
        _gaussian_mask[round(mask_z/2-ego_mask_vox_z/2):round(mask_z/2+ego_mask_vox_z/2), 
                        round(mask_x/2-ego_mask_vox_x/2):round(mask_x/2+ego_mask_vox_x/2)] = 0
        offset_vox_z = mask_z//2 + int(vox_zmin)
        offset_vox_x = mask_x//2 + int(vox_xmin)
        gaussian_mask = _gaussian_mask[offset_vox_z:offset_vox_z+vox_z, offset_vox_x:offset_vox_x+vox_x]
        return torch.from_numpy(gaussian_mask[None,None,:,:]).float()

    def prepare_forward(self, cameras_image, cameras_extrin, cameras_intrin):
        B, S, C, H, W = cameras_image.shape

        cameras_extrin_ = basic.pack_seqdim(cameras_extrin, B)
        ref_T_cameras_ = geom.safe_inverse(cameras_extrin_)
        cameras_intrin_ = basic.pack_seqdim(cameras_intrin, B)

        # 准备gridsample
        if hasattr(self.transform, "vox_s2"):
            image_stride = 8
             # 将内参缩小到和特征图一样的尺寸
            cameras_intrin_stride_ = geom.scale_intrinsics(cameras_intrin_, 1/image_stride, 1/image_stride)
            self.transform.vox_s2.prepare_Pix2Vox(
                cameras_intrin_stride_,
                ref_T_cameras_,
                self.grid3d_s2.repeat(B*S, 1, 1).to(cameras_intrin_stride_.device),
                W//image_stride, H//image_stride
            )
        if hasattr(self.transform, "vox_s4"):
            image_stride = 16
            cameras_intrin_stride_ = geom.scale_intrinsics(cameras_intrin_, 1/image_stride, 1/image_stride)
            self.transform.vox_s4.prepare_Pix2Vox(
                cameras_intrin_stride_,
                ref_T_cameras_,
                self.grid3d_s4.repeat(B*S, 1, 1).to(cameras_intrin_stride_.device),
                W//image_stride, H//image_stride
            )
        if hasattr(self.transform, "vox_s8"):
            image_stride = 32
            cameras_intrin_stride_ = geom.scale_intrinsics(cameras_intrin_, 1/image_stride, 1/image_stride)
            self.transform.vox_s8.prepare_Pix2Vox(
                cameras_intrin_stride_,
                ref_T_cameras_,
                self.grid3d_s8.repeat(B*S, 1, 1).to(cameras_intrin_stride_.device),
                W//image_stride, H//image_stride
            )

    def _forward_fv_(self, cameras_image, temporal_feature=None):
        B, S, C, H, W = cameras_image.shape
        cameras_image_ = basic.pack_seqdim(cameras_image, B)
        datas = {
            "B": B,
            "S": S,
            "cameras_image_": cameras_image_,
            "temporal_feature": temporal_feature,
        }

        datas = self.preproc(datas)
        datas = self.backbone(datas)
        datas = self.neck(datas)

        return datas

    def _forward_bev_(self, datas):
        datas = self.bev_backbone(datas)
        datas = self.bev_neck(datas)

        return datas

    # 生成点云目标
    def make_occ_targets(self, B, lidars_points, lidars_extrin):
        # 点云聚合
        lidars_points_ = basic.pack_seqdim(lidars_points, B)
        lidars_extrin_ = basic.pack_seqdim(lidars_extrin, B)
        lidars_points_ref_ = geom.apply_4x4(lidars_extrin_, lidars_points_)

        # 将lidar点云合并到voxel坐标系
        lidars_points_ref = lidars_points_ref_.reshape(B, -1, 3)
        # 平均目标直径
        target_world_diameter = (self.dimentions.mean(0)[0]**2 + self.dimentions.mean(0)[1]**2)**0.5
        vox_per_world = (self.vox.vox_x_size/(self.vox.world_xmax-self.vox.world_xmin), \
                            self.vox.vox_z_size/(self.vox.world_zmax-self.vox.world_zmin))
        target_vox_diameter = (target_world_diameter*vox_per_world[0], 
                                target_world_diameter*vox_per_world[1])
        radius = basic.gaussian_radius(target_vox_diameter)
        return self.vox.occ_centermask(lidars_points_ref, radius=radius) # B, Y, Z, X

    def forward_trainval(self, cameras_image, cameras_extrin, cameras_intrin,
            lidars_points, lidars_extrin,
            cameras_annos=None, lidars_annos=None
        ):
        # 准备centermask的GT    
        assert lidars_points is not None \
            and lidars_extrin is not None, "lidars_points and lidars_extrin must be provided for training"
        # 点云聚合
        B, S, C, H, W = cameras_image.shape

        occ_targets = self.make_occ_targets(B, lidars_points, lidars_extrin)

        self.prepare_forward(cameras_image, cameras_extrin, cameras_intrin)

        # fv forward
        datas = self._forward_fv_(cameras_image, None)

        # transform
        datas = self.transform(datas)
        # bev augment
        if self.training and self.bev_augment is not None:
            # bev数据增强
            occ_targets, datas \
                = self.bev_augment(occ_targets, datas)

        # bev forward
        datas = self._forward_bev_(datas)

        # occ
        occ_pred = self.occ_head(datas)
        if self.training:
            aux_occ_preds_list = []
            for aux_occ_head in self.aux_occ_head_list:
                aux_occ_pred = aux_occ_head(datas)
                aux_occ_preds_list.append(aux_occ_pred)

        # gaussian mask repeat as batch size
        if self.gaussian_mask.device != occ_pred.device:
            self.gaussian_mask = self.gaussian_mask.to(occ_pred.device)

        Zm,Xm = self.gaussian_mask.shape[2:]
        valid_vox_mask = datas["valid_vox_mask"]
        valid_vox_mask = F.interpolate(valid_vox_mask, (Zm,Xm), mode="nearest")
        valid_vox_mask = valid_vox_mask * self.gaussian_mask

        if self.training:
            outputs = {"total_loss":0}
            occ_total_loss = 0
            # occ main loss
            occ_losses \
                = self.occ_head.get_losses(occ_pred, occ_targets, valid_vox_mask)
            occ_total_loss += occ_losses["total_loss"]
            for key in occ_losses:
                new_key = "occ_" + key
                outputs[new_key] = occ_losses[key]

            # aux occ loss
            aux_occ_total_loss = 0
            for aux_occ_preds in aux_occ_preds_list:
                aux_occ_losses \
                    = self.occ_head.get_losses(aux_occ_preds, occ_targets, valid_vox_mask, uncertainty=False)
                aux_occ_total_loss += 0.1 * aux_occ_losses["total_loss"]

            if len(aux_occ_preds_list) > 0:
                aux_occ_total_loss = aux_occ_total_loss/len(aux_occ_preds_list)
                occ_total_loss += aux_occ_total_loss
                
                outputs["aux_occ_loss"] = aux_occ_total_loss
            
            total_loss = occ_total_loss

            # train outputs
            outputs["total_loss"] = total_loss

        else:
            outputs = {}
            occ_metrics = self.occ_head.get_metrics(occ_pred, occ_targets, valid_vox_mask)
            
            for key in occ_metrics:
                new_key = "occ_" + key
                outputs[new_key] = occ_metrics[key]

        return outputs

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.forward = self.forward_trainval
        return self

    # ------------------------------------------------
    # BEGIN: export forward
    def prepare_export(self, cameras_image, cameras_extrin, cameras_intrin, perspective_mode="gridsample"):
        self.forward = self.forward_export
        self.transform.perspective_mode = perspective_mode
        self.prepare_forward(cameras_image, cameras_extrin, cameras_intrin)

    def forward_export(self, cameras_image, temporal_feature=None):
        datas = self._forward_fv_(cameras_image, temporal_feature)
        datas = self.transform(datas)
        datas = self._forward_bev_(datas)
        
        occ = self.occ_head(datas)
        
        if temporal_feature is not None:
            return occ, datas["temporal_feature"]
        else:
            return occ
    
    def prepare_export_images(self, cameras_image_0, cameras_image_1, cameras_image_2, cameras_extrin, cameras_intrin, perspective_mode="gridsample"):
        self.forward = self.forward_export_images
        self.transform.perspective_mode = perspective_mode
        cameras_image = torch.stack([cameras_image_0, cameras_image_1, cameras_image_2], dim=1)
        self.prepare_forward(cameras_image, cameras_extrin, cameras_intrin)
    
    def forward_export_images(self, cameras_image_0, cameras_image_1, cameras_image_2, temporal_feature=None):
        cameras_image = torch.stack([cameras_image_0, cameras_image_1, cameras_image_2], dim=1)
        return self.forward_export(cameras_image, temporal_feature)

    # END: export forward
    # ------------------------------------------------
