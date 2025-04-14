#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yoloxocc.models import BEVAugment
from yoloxocc.models.losses import FocalLoss, CenterLoss, DiceLoss, UncertaintyLoss
from yoloxocc.utils import basic, geom, VoxUtil, get_occ_iou

import random

# 模型框架
class YOLOXOCC(nn.Module):
    """
    YOLOXOCC model module.
    The network returns loss values during training
    and output results during test.
    """

    def __init__(self, 
                preproc=None, # 数据预处理 basenorm/deepnorm
                backbone=None, # regnet
                neck=None, # fpn
                transform=None, # remapping/perspective
                bev_backbone=None, # regnet_neck_pan
                bev_temporal=None, # 假时序
                bev_neck=None, # fpn
                bev_head=None, # bev head
                aux_bev_head_list=[], # 辅助bev head
                occ_head=None, # occ head
                aux_occ_head_list=[], # 辅助occ head
                world_xyz_bounds = [-32, 32, -2, 2, -32, 32], # 世界坐标范围 单位m
                vox_xyz_size=[128, 4, 128], # 体素坐标大小 单位格子
                precise_meter=0.5, # 高斯图精度 单位m
                bev_erase_prob=0.5, # 擦除概率
                bev_flip_prob=0.5, # 翻转概率
                bev_mixup_prob=0.5, # 混合概率
                bev_mosaic_prob=0.5 # 马赛克概率
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
        self.bev_augment = BEVAugment(
            max_labels=200, # max number of labels
            bev_erase_prob=bev_erase_prob, # probability of erase
            bev_flip_prob=bev_flip_prob, # probability of flip
            bev_mixup_prob=bev_mixup_prob, # probability of mixup
            bev_mosaic_prob=bev_mosaic_prob, # probability of mosaic
        )

        # bev backbone
        self.bev_backbone = nn.Identity() if bev_backbone is None else bev_backbone
        # bev temporal
        self.bev_temporal = nn.Identity() if bev_temporal is None else bev_temporal
        # bev neck: fpn
        self.bev_neck = nn.Identity() if bev_neck is None else bev_neck

        # bev head
        self.bev_head = nn.Identity() if bev_head is None else bev_head
        # aux bev head
        self.aux_bev_head_list = nn.ModuleList()
        for aux_bev_head in aux_bev_head_list:
            self.aux_bev_head_list.append(aux_bev_head)

        # occupancy head
        self.occ_head = nn.Identity() if occ_head is None else occ_head
        # aux occupancy head
        self.aux_occ_head_list = nn.ModuleList()
        for aux_occ_head in aux_occ_head_list:
            self.aux_occ_head_list.append(aux_occ_head)
        
        # loss functions
        # bev preds vs targets 
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([2.13]), reduction="mean")
        self.focal_loss = FocalLoss(reduction="mean")
        # bev preds vs centermask targets
        self.center_loss = CenterLoss()
        # bev preds vs targets
        self.dice_loss = DiceLoss()

        # uncertainty loss
        self.bce_uncertainty_loss = UncertaintyLoss()
        self.focal_uncertainty_loss = UncertaintyLoss()
        self.center_uncertainty_loss = UncertaintyLoss()
        self.dice_uncertainty_loss = UncertaintyLoss()

        # 预制grid
        self.grid3d_s2 = basic.cloudgrid3d(1, vox_xyz_size[0]//2, vox_xyz_size[1], vox_xyz_size[2]//2)
        self.grid3d_s4 = basic.cloudgrid3d(1, vox_xyz_size[0]//4, vox_xyz_size[1], vox_xyz_size[2]//4)
        self.grid3d_s8 = basic.cloudgrid3d(1, vox_xyz_size[0]//8, vox_xyz_size[1], vox_xyz_size[2]//8)

        # strided vox
        self.vox = VoxUtil(
                [vox_xyz_size[0], vox_xyz_size[1], vox_xyz_size[2]],
                world_xyz_bounds=world_xyz_bounds,
                precise_meter=precise_meter
            )
        
        self.valid_mask = self.get_valid_mask(world_xyz_bounds, vox_xyz_size)
        
        
    # 全图给出高斯圆mask
    def get_valid_mask(self,
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
        valid_mask = cv2.getGaussianKernel(ksize, sigma)
        valid_mask = valid_mask.dot(valid_mask.T) # 2d gaussian
        mask_z, mask_x = valid_mask.shape
        offset_vox_z = mask_z//2 + int(vox_zmin)
        offset_vox_x = mask_x//2 + int(vox_xmin)
        valid_mask = valid_mask[offset_vox_z:offset_vox_z+vox_z, offset_vox_x:offset_vox_x+vox_x]
        valid_mask = (valid_mask-valid_mask.min()) / (valid_mask.max()-valid_mask.min())
        return torch.from_numpy(valid_mask[None,None,:,:]).float()


    def prepare_forward(self, cameras_image, cameras_extrin, cameras_intrin):
        B, S, C, H, W = cameras_image.shape

        cameras_extrin_ = basic.pack_seqdim(cameras_extrin, B)
        ref_T_cameras_ = geom.safe_inverse(cameras_extrin_)
        cameras_intrin_ = basic.pack_seqdim(cameras_intrin, B)
        cameras_intrin_s8_ = geom.scale_intrinsics(cameras_intrin_, 1/8, 1/8) # 将内参缩小到和特征图一样的尺寸
        cameras_intrin_s16_ = geom.scale_intrinsics(cameras_intrin_, 1/16, 1/16)
        cameras_intrin_s32_ = geom.scale_intrinsics(cameras_intrin_, 1/32, 1/32)

        # 准备gridsample
        self.transform.vox_s2.prepare_Pix2Vox(
            cameras_intrin_s8_,
            ref_T_cameras_,
            self.grid3d_s2.repeat(B*S, 1, 1).to(cameras_intrin_s8_.device),
            W//8, H//8
        )
        self.transform.vox_s4.prepare_Pix2Vox(
            cameras_intrin_s16_,
            ref_T_cameras_,
            self.grid3d_s4.repeat(B*S, 1, 1).to(cameras_intrin_s16_.device),
            W//16, H//16
        )
        self.transform.vox_s8.prepare_Pix2Vox(
            cameras_intrin_s32_,
            ref_T_cameras_,
            self.grid3d_s8.repeat(B*S, 1, 1).to(cameras_intrin_s32_.device),
            W//32, H//32
        )


    def _forward_fv_(self, cameras_image, temporal_feature=None):
        B, S, C, H, W = cameras_image.shape
        cameras_image_ = basic.pack_seqdim(cameras_image, B)
        datas = {
            "B": B, 
            "cameras_image_": cameras_image_,
            "temporal_feature": temporal_feature,
        }

        datas = self.preproc(datas)

        datas = self.backbone(datas)
        datas = self.neck(datas)

        return datas

    def _forward_bev_(self, datas):
        datas = self.bev_backbone(datas)
        datas = self.bev_temporal(datas)
        datas = self.bev_neck(datas)

        return datas


    def forward_trainval(self, cameras_image, cameras_extrin, cameras_intrin,
                lidars_points, lidars_extrin,
                cameras_annos=None, lidars_annos=None
                ):
        # 准备occ_centermask的GT    
        assert lidars_points is not None \
            and lidars_extrin is not None, "lidars_points and lidars_extrin must be provided for training"
        # 点云聚合
        B, S, C, H, W = cameras_image.shape
        lidars_points_ = basic.pack_seqdim(lidars_points, B)
        lidars_extrin_ = basic.pack_seqdim(lidars_extrin, B)
        lidars_points_ref_ = geom.apply_4x4(lidars_extrin_, lidars_points_)
        # BEV instance 聚合
        lidars_annos_xyz = lidars_annos[..., :3]
        lidars_annos_xyz_ = basic.pack_seqdim(lidars_annos_xyz, B)
        lidars_annos_xyz_ref_ = geom.apply_4x4(lidars_extrin_, lidars_annos_xyz_)
        lidars_annos_xz_ref_ = lidars_annos_xyz_ref_[..., [0,2]]
        # BEV instance bbox
        lidars_annos_rlw = lidars_annos[..., [3,5,6]]
        lidars_annos_rlw_ = basic.pack_seqdim(lidars_annos_rlw, B)
        lidars_annos_r_ = lidars_annos_rlw_[..., 0]
        lidars_annos_l_ = lidars_annos_rlw_[..., 1]
        lidars_annos_w_ = lidars_annos_rlw_[..., 2]
        lidars_annos_Z_ = lidars_annos_l_*torch.abs(torch.cos(lidars_annos_r_)) + lidars_annos_w_*torch.abs(torch.sin(lidars_annos_r_))
        lidars_annos_X_ = lidars_annos_l_*torch.abs(torch.sin(lidars_annos_r_)) + lidars_annos_w_*torch.abs(torch.cos(lidars_annos_r_))
        lidars_annos_x0z0_ref_ = lidars_annos_xz_ref_ - torch.stack([lidars_annos_X_, lidars_annos_Z_], dim=-1)/2
        lidars_annos_x1z1_ref_ = lidars_annos_xz_ref_ + torch.stack([lidars_annos_X_, lidars_annos_Z_], dim=-1)/2
        lidars_annos_x0z0x1z1_ref_ = torch.cat([lidars_annos_x0z0_ref_, lidars_annos_x1z1_ref_], dim=-1) # B, N, 4
        # BEV instance cls
        lidars_annos_cls = lidars_annos[..., 4]
        lidars_annos_cls_ = basic.pack_seqdim(lidars_annos_cls, B)
        # 拼接 x0,z0,x1,z1,cls
        lidars_annos_x0z0x1z1cls_ref_ = torch.cat([lidars_annos_x0z0x1z1_ref_, lidars_annos_cls_[..., None]], dim=-1) # B, N, 5

        # 将lidar点云合并到voxel坐标系
        lidars_points_ref = lidars_points_ref_.reshape(B, -1, 3)
        occ_centermask_target = self.vox.occ_centermask(lidars_points_ref) # B, Y, Z, X

        self.prepare_forward(cameras_image, cameras_extrin, cameras_intrin)
        
        # 前视forward
        datas = self._forward_fv_(cameras_image, None)
        # 随机映射方式
        perspective_modes = ["gridsample", "remapping"]
        self.transform.perspective_mode = random.choice(perspective_modes)
        datas = self.transform(datas)

        if self.training:
            # bev数据增强
            datas["trans3"],datas["trans4"],datas["trans5"],\
                occ_centermask_target, lidars_annos = self.bev_augment(
                    datas["trans3"],datas["trans4"],datas["trans5"],\
                    occ_centermask_target, lidars_annos_x0z0x1z1cls_ref_)

        # 鸟瞰forward
        datas = self._forward_bev_(datas)

        # bev
        bev_preds = self.bev_head(datas)
        aux_bev_preds_list = []
        for aux_bev_head in self.aux_bev_head_list:
            bev_preds_list = aux_bev_head(datas)
            aux_bev_preds_list.extend(bev_preds_list)

        # occ
        occ_pred = self.occ_head(datas)
        aux_occ_preds_list = []
        for aux_occ_head in self.aux_occ_head_list:
            occ_preds_list = aux_occ_head(datas)
            aux_occ_preds_list.extend(occ_preds_list)

        # valid mask repeat as batch size
        if self.valid_mask.device != occ_pred.device:
            self.valid_mask = self.valid_mask.to(occ_pred.device)

        if self.training:
            total_loss = 0
            # occ main loss
            occ_total_loss, occ_bce_loss, occ_focal_loss, occ_center_loss, occ_dice_loss \
                = self.get_occ_losses(occ_pred, occ_centermask_target)

            # aux occ loss
            aux_occ_loss = 0
            for aux_occ_preds in aux_occ_preds_list:
                aux_occ_total_loss, _, _, _, _ \
                    = self.get_occ_losses(aux_occ_preds, occ_centermask_target, uncertainty=False)
                aux_occ_loss += 0.1 * aux_occ_total_loss

            if len(aux_occ_preds_list) > 0:
                aux_occ_loss = aux_occ_loss/len(aux_occ_preds_list)
                occ_total_loss += aux_occ_loss

            total_loss += occ_total_loss

            # train outputs
            outputs = {
                "total_loss": total_loss,
                "occ_total_loss": occ_total_loss,
                "occ_bce_loss": occ_bce_loss,
                "occ_focal_loss": occ_focal_loss,
                "occ_center_loss": occ_center_loss,
                "occ_dice_loss": occ_dice_loss,
                "aux_occ_loss": aux_occ_loss
            }

        else:
            occ_iou = get_occ_iou(occ_pred, occ_centermask_target)

            # eval outputs
            outputs = {
                "occ_iou": occ_iou,
            }

        return outputs
    
    def forward(self, cameras_image, *args):
        self.forward_trainval(cameras_image, *args)

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
        occ_pred,
        occ_centermask_target,
        uncertainty=True
    ):
        # occ_pred B,Y,Z,X
        # occ_centermask_target B,Y,Z,X
        # valid_mask B,Z,X

        assert occ_pred.shape[0] == occ_centermask_target.shape[0] \
            and occ_pred.shape[1] == occ_centermask_target.shape[1], \
            f"expect {occ_pred.shape[:2]} == {occ_centermask_target.shape[:2]}"

        # 对齐
        Zp,Xp = occ_pred.shape[2:]
        Z,X = occ_centermask_target.shape[2:]
        if Zp!=Z or Xp!=X:
            occ_pred = F.adaptive_max_pool2d(occ_pred, (Z,X))

        # occ_pred
        bce_loss = self.bce_loss(occ_pred, occ_centermask_target.round())
        focal_loss = self.focal_loss(occ_pred, occ_centermask_target.round())
        center_loss = self.center_loss(occ_pred.sigmoid()*self.valid_mask, occ_centermask_target*self.valid_mask)
        dice_loss = self.dice_loss((occ_pred.sigmoid()*self.valid_mask).round(), (occ_centermask_target*self.valid_mask).round())

        if uncertainty:
            # uncertainty weight
            bce_loss = self.bce_uncertainty_loss(bce_loss)
            focal_loss = self.focal_uncertainty_loss(focal_loss)
            center_loss = self.center_uncertainty_loss(center_loss)
            dice_loss = self.dice_uncertainty_loss(dice_loss)

        total_loss = bce_loss + focal_loss + center_loss + dice_loss

        return total_loss, bce_loss, focal_loss, center_loss, dice_loss

