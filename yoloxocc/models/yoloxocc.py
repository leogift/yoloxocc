#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from yoloxocc.models.losses import HeatmapLoss, BalanceLoss, UncertaintyLoss, DiceLoss
from yoloxocc.models.metrics import HeatmapMetric, SimilarityMetric, IOUMetric
from yoloxocc.utils import basic, geom, VoxUtil

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
                bev_neck=None, # bev fpn
                occ_head=None, # occ head
                aux_occ_head_list=[], # 辅助occ head
                world_xyz_bounds = [-32, 32, -2, 2, -32, 32], # 世界坐标范围 单位m
                vox_xyz_size=[128, 4, 128], # 体素坐标大小 单位格子
                use_gaussian_mask=False, # 是否使用gaussian mask
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
        
        # bev neck: fpn
        self.bev_neck = nn.Identity() if transform is None else bev_neck

        # occ head
        self.occ_head = nn.Identity() if transform is None else occ_head
        # aux occ head
        self.aux_occ_head_list = nn.ModuleList()
        for aux_occ_head in aux_occ_head_list:
            self.aux_occ_head_list.append(aux_occ_head)

        # loss functions
        self.bce_loss = HeatmapLoss()
        self.diff_loss = HeatmapLoss(BalanceLoss())
        self.dice_loss = HeatmapLoss(DiceLoss())

        # uncertainty loss
        self.bce_uncertainty_loss = UncertaintyLoss()
        self.diff_uncertainty_loss = UncertaintyLoss()
        self.dice_uncertainty_loss = UncertaintyLoss()
		
		# metric
        self.similarity_metric = HeatmapMetric(SimilarityMetric())
        self.iou_metric = HeatmapMetric(IOUMetric())
		
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
            self.gaussian_mask = self.get_gaussian_mask(world_xyz_bounds, vox_xyz_size)
        else:
            self.gaussian_mask = torch.ones((1, 1, vox_xyz_size[2], vox_xyz_size[0]), dtype=torch.float32)

    # gaussian mask for voxel
    def get_gaussian_mask(self,
                   world_xyz_bounds,
                   vox_xyz_size):
        import cv2
        world_xmin, world_xmax, world_ymin, world_ymax, world_zmin, world_zmax = world_xyz_bounds
        vox_x, vox_y, vox_z = vox_xyz_size
        max_vox_zx = max(vox_z, vox_x)
        vox_xmin = vox_x/(world_xmax-world_xmin)*world_xmin
        vox_zmin = vox_z/(world_zmax-world_zmin)*world_zmin
        ksize = max_vox_zx*2+1
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        gaussian_mask = cv2.getGaussianKernel(ksize, sigma)
        gaussian_mask = (gaussian_mask.dot(gaussian_mask.T))**0.5 # 2d gaussian
        mask_z, mask_x = gaussian_mask.shape
        offset_vox_z = mask_z//2 + int(vox_zmin)
        offset_vox_x = mask_x//2 + int(vox_xmin)
        gaussian_mask = gaussian_mask[offset_vox_z:offset_vox_z+vox_z, offset_vox_x:offset_vox_x+vox_x]
        gaussian_mask = (gaussian_mask-gaussian_mask.min()) / (gaussian_mask.max()-gaussian_mask.min())
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
        datas = self.bev_neck(datas)

        return datas

    def forward_trainval(self, cameras_image, cameras_extrin, cameras_intrin,
            lidars_points, lidars_extrin,
            cameras_annos=None, lidars_annos=None
        ):
        # 准备centermask的GT    
        assert lidars_points is not None \
            and lidars_extrin is not None, "lidars_points and lidars_extrin must be provided for training"
        # 点云聚合
        B, S, C, H, W = cameras_image.shape
        lidars_points_ = basic.pack_seqdim(lidars_points, B)
        lidars_extrin_ = basic.pack_seqdim(lidars_extrin, B)
        lidars_points_ref_ = geom.apply_4x4(lidars_extrin_, lidars_points_)

        # 将lidar点云合并到voxel坐标系
        lidars_points_ref = lidars_points_ref_.reshape(B, -1, 3)
        radius = basic.gaussian_radius((self.vox.vox_z_size,self.vox.vox_x_size), stride=8)
        occ_centermask_target = self.vox.occ_centermask(lidars_points_ref, radius=radius) # B, Y, Z, X

        self.prepare_forward(cameras_image, cameras_extrin, cameras_intrin)

        # fv forward
        datas = self._forward_fv_(cameras_image, None)

        # transform
        datas = self.transform(datas)
        # bev augment
        if self.training and self.bev_augment is not None:
            # bev数据增强
            occ_centermask_target, datas \
                = self.bev_augment(occ_centermask_target, datas)

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
            # occ main loss
            occ_total_loss, occ_bce_loss, occ_diff_loss, occ_dice_loss \
                = self.get_occ_losses(occ_pred, occ_centermask_target, valid_vox_mask)

            # aux occ loss
            aux_occ_total_loss = 0
            for aux_occ_preds in aux_occ_preds_list:
                _aux_occ_total_loss, _, _, _ \
                    = self.get_occ_losses(aux_occ_preds, occ_centermask_target, valid_vox_mask, uncertainty=False)
                aux_occ_total_loss += 0.1 * _aux_occ_total_loss

            if len(aux_occ_preds_list) > 0:
                aux_occ_total_loss = aux_occ_total_loss/len(aux_occ_preds_list)
                occ_total_loss += aux_occ_total_loss

            total_loss = occ_total_loss

            # train outputs
            outputs = {
                "total_loss": total_loss,
                "occ_total_loss": occ_total_loss,
                "occ_bce_loss": occ_bce_loss,
                "occ_diff_loss": occ_diff_loss,
                "occ_dice_loss": occ_dice_loss,
                "aux_occ_total_loss": aux_occ_total_loss,
            }

        else:
            with torch.no_grad():
                occ_similarity = self.get_occ_similarity(occ_pred, occ_centermask_target, valid_vox_mask)
                occ_iou = self.get_occ_iou(occ_pred, occ_centermask_target, valid_vox_mask)

            # eval outputs
            outputs = {
                "occ_similarity": occ_similarity,
                "occ_iou": occ_iou,
            }

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
            return occ.sigmoid(), datas["temporal_feature"]
        else:
            return occ.sigmoid()
    
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

    # losses
    def get_occ_losses(
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
        bce_loss = self.bce_loss(_pred*_valid_mask, (_target*_valid_mask).round())
        diff_loss  = self.diff_loss((_pred*_valid_mask).sigmoid(), _target*_valid_mask)
        dice_loss = self.dice_loss((_pred*_valid_mask).sigmoid(), (_target*_valid_mask).round())

        if uncertainty:
            # uncertainty weight
            bce_loss = self.bce_uncertainty_loss(bce_loss)
            diff_loss = self.diff_uncertainty_loss(diff_loss, 5)
            dice_loss = self.dice_uncertainty_loss(dice_loss, 2)

        total_loss = bce_loss + diff_loss + dice_loss

        return total_loss, bce_loss, diff_loss, dice_loss

    # similarity
    def get_occ_similarity(
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

        similarity = self.similarity_metric((_pred*_valid_mask).sigmoid(), _target*_valid_mask, debug=True)

        return similarity

    # dice
    def get_occ_iou(
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
        _target = F.interpolate(target, (Zp,Xp), mode="bilinear", align_corners=True)
        _valid_mask = F.interpolate(valid_mask, (Zp,Xp), mode="nearest")
        _pred = pred

        iou = self.iou_metric((_pred*_valid_mask).sigmoid(), (_target*_valid_mask).round())

        return iou
