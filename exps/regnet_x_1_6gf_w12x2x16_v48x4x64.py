#!/usr/bin/env python3

import os
import torch
from torch import nn

from yoloxocc.exp import Exp as BaseExp

from loguru import logger

from yoloxocc.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

_CKPT_FULL_PATH = "pretrain/yoloxocc_regnet_x_1_6gf_y4.pth"

class Exp(BaseExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_dir = "custom"

        self.train_json = "train.json"
        self.val_json = "val.json"

        self.image_size = (288, 512)  # (height, width)
        self.camera_list = ['front', 'left', 'right']
        self.lidar_list = ['top']
        self.category_list = ['ugv', 'pedestrian', 'tree']

        self.world_xyz_bounds = [-6, 6, -1.5, 0.5, -2, 14] # [xmin, xmax, ymin, ymax, zmin, zmax]
        self.vox_xyz_size = [48, 4, 64] # voxel size in [x, y, z]
        self.vox_y_weight = [1.0, 2.0, 2.0 ,0.5]
        self.use_gaussian_mask = True
        self.ego_dimention = [1.4, 0.8, 1.5]
        self.dimentions = [
            [1.4, 0.8, 1.5], # ugv
            [0.7, 0.7, 1.75], # pedestrian
            [0.25, 0.25, 15], # tree
        ]

        self.act = "relu"
        self.max_epoch = 30

        self.model_name = "regnet_x_1_6gf"
        self.bev_model_name = "resnet18"

        self.warmup_epochs = 5
        self.no_aug_epochs = 10
        self.data_num_workers = 2
        self.eval_epoch_interval = 5


    def get_model(self):

        if "model" not in self.__dict__:
            from yoloxocc.models import YOLOXOCC, \
            BaseNorm, Regnet, YOLONeckFPN, \
            IPMTrans, BEVAugment, \
            BEVResnet, \
            OCCHead
            
            preproc = BaseNorm(trainable=True)

            pp_repeats = 0 if min(self.image_size[0], self.image_size[1])//32 < 7 \
                else (min(self.image_size[0], self.image_size[1])//32 - 7)//6 + 1
            backbone = Regnet(
                self.model_name,
                act=self.act, 
                pp_repeats=pp_repeats,
                drop_rate=0.1,
            )
            channels = backbone.channels[-3:]
            neck = YOLONeckFPN(
                in_features=("backbone3", "backbone4", "backbone5"),
                out_features=("fpn3", "fpn4", "fpn5"),
                channels=channels,
                act=self.act,
                n=2,
                simple_reshape=False
            )

            bev_backbone = BEVResnet(
                self.bev_model_name,
                in_features=["bev_trans3", "bev_trans4", "bev_trans5"],
                out_features=["bev_backbone3", "bev_backbone4", "bev_backbone5"],
                act=self.act,
                n=2,
                drop_rate=0.1,
                vox_xyz_size=self.vox_xyz_size,
            )
            bev_channels = bev_backbone.channels[:3]

            transform = IPMTrans(
                in_features=["fpn3", "fpn4", "fpn5"],
                in_channels=channels,
                out_features=["bev_trans3", "bev_trans4", "bev_trans5"],
                out_channels=bev_channels,
                act=self.act, 
                n=2,
                vox_xyz_size=self.vox_xyz_size,
                world_xyz_bounds=self.world_xyz_bounds,
            )
            bev_augment = BEVAugment(
                bev_erase_prob=self.bev_erase_prob, # probability of erase
                bev_flip_prob=self.bev_flip_prob, # probability of flip
                bev_mosaic_prob=self.bev_mosaic_prob, # probability of mosaic
            )

            bev_neck = nn.Sequential(*[
                YOLONeckFPN(
                    in_features=["bev_backbone3", "bev_backbone4", "bev_backbone5"],
                    out_features=["bev_fpn3", "bev_fpn4", "bev_fpn5"],
                    channels=bev_channels,
                    act=self.act, 
                    n=2,
                    simple_reshape=False
                ),
            ])
            
            occ_head = OCCHead(
                in_feature="bev_fpn3",
                in_channel=bev_channels[0],
                act=self.act, 
                n=2,
                vox_y=self.vox_xyz_size[1],
                vox_y_weight=self.vox_y_weight,
                simple_reshape=False
            )
            aux_occ_head_list = [
                OCCHead(
                    in_feature="bev_fpn3",
                    in_channel=bev_channels[0],
                    vox_y=self.vox_xyz_size[1],
                    act=self.act,
                    aux_head=True
                ),
                OCCHead(
                    in_feature="bev_fpn4",
                    in_channel=bev_channels[1],
                    vox_y=self.vox_xyz_size[1],
                    act=self.act,
                    aux_head=True
                ),
                OCCHead(
                    in_feature="bev_fpn5",
                    in_channel=bev_channels[2],
                    vox_y=self.vox_xyz_size[1],
                    act=self.act,
                    aux_head=True
                ),
            ]

            self.model = YOLOXOCC(
                preproc=preproc,
                backbone=backbone,
                neck=neck, 
                transform=transform,
                bev_augment=bev_augment,
                bev_backbone=bev_backbone,
                bev_neck=bev_neck,
                occ_head=occ_head, 
                aux_occ_head_list=aux_occ_head_list,
                world_xyz_bounds=self.world_xyz_bounds,
                vox_xyz_size=self.vox_xyz_size,
                use_gaussian_mask=self.use_gaussian_mask,
                ego_dimention=self.ego_dimention,
                dimentions=self.dimentions,
            )
        
        ckpt = torch.load(_CKPT_FULL_PATH, map_location="cpu", weights_only=True)
        if "model" in ckpt:
            ckpt = ckpt["model"]

        for k in list(ckpt.keys()):
            if "loss" in k or "Loss" in k:
                del ckpt[k]

        incompatible = self.model.load_state_dict(ckpt, strict=False)
        logger.info("missing_keys:")
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )

        logger.info("unexpected_keys:")
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )
        
        return self.model
