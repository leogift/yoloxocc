#!/usr/bin/env python3

import os
import torch

from yoloxocc.exp import Exp as BaseExp

from loguru import logger

from yoloxocc.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

_CKPT_FULL_PATH = "pretrain/yoloxocc_regnet_x_400mf_v64x4x48.pth"

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
        self.category_list = ['car', 'pedestrian']
        self.world_xyz_bounds = [-8, 8, -0.5, 1.5, -4, 8] # [xmin, xmax+1unit, ymin, ymax+1unit, zmin, zmax+1unit]
        # voxel size in [x, y, z]
        self.vox_xyz_size = [64, 4, 48]

        self.act = "relu"
        self.max_epoch = 120

        self.model_name = "regnet_x_400mf"

        self.warmup_epochs = 10
        self.no_aug_epochs = 20
        self.data_num_workers = 4
        self.eval_epoch_interval = 5


    def get_model(self):

        if "model" not in self.__dict__:
            from yoloxocc.models import YOLOXOCC, \
            BaseNorm, Regnet, YOLONeckFPN, \
            PerspectiveTrans, \
            RegnetNeckPAN, Temporal, OCCHead, AUXOCCHead, \
            C2kLayer
            
            preproc = BaseNorm(trainable=True)

            pp_repeats = 0 if min(self.image_size[0], self.image_size[1])//32 <= 4 \
                else (min(self.image_size[0], self.image_size[1])//32 - 4)//6 + 1
            backbone = Regnet(
                    self.model_name,
                    act=self.act, 
                    pp_repeats=pp_repeats,
                    drop_rate=0.1,
                )
            self.channels = backbone.output_channels[-3:]
            neck = YOLONeckFPN(
                    in_features=("backbone3", "backbone4", "backbone5"),
                    in_channels=self.channels,
                    out_features=("fpn3", "fpn4", "fpn5"),
                    act=self.act, 
                    layer_type=C2kLayer,
                    simple_reshape=True,
                    n=1
                )
            transform = PerspectiveTrans(
                    in_channels=self.channels,
                    act=self.act, 
                    layer_type=C2kLayer,
                    vox_xyz_size=self.vox_xyz_size,
                    world_xyz_bounds=self.world_xyz_bounds,
                    n=1
                )
            
            pp_repeats = 0 if min(self.vox_xyz_size[0], self.vox_xyz_size[2])//8 <= 4 \
                else (min(self.vox_xyz_size[0], self.vox_xyz_size[2])//8 - 4)//6 + 1
            bev_backbone = RegnetNeckPAN(
                    self.model_name,
                    in_channels=self.channels,
                    act=self.act,
                    pp_repeats=pp_repeats,
                    transformer=True,
                    drop_rate=0.1,
                    layer_type=C2kLayer,
                    n=1
                )
            self.bev_channels = bev_backbone.output_channels[-3:]
            bev_temporal = None
            bev_neck = YOLONeckFPN(
                    in_features=("bev_backbone3", "bev_backbone4", "bev_backbone5"),
                    in_channels=self.bev_channels,
                    out_features=("bev_fpn3", "bev_fpn4", "bev_fpn5"),
                    act=self.act, 
                    layer_type=C2kLayer,
                    simple_reshape=True,
                    n=1
                )
            occ_head = OCCHead(
                    in_feature="bev_fpn3",
                    in_channel=self.bev_channels[0],
                    vox_y=self.vox_xyz_size[1],
                    act=self.act, 
                    drop_rate=0.1,
                    simple_reshape=True
                )
            aux_occ_head_list = [
                    AUXOCCHead(
                        in_features=("bev_backbone3", "bev_backbone4", "bev_backbone5"),
                        in_channels=self.bev_channels,
                        vox_y=self.vox_xyz_size[1],
                    ),
                    AUXOCCHead(
                        in_features=("bev_fpn3", "bev_fpn4", "bev_fpn5"),
                        in_channels=self.bev_channels,
                        vox_y=self.vox_xyz_size[1],
                    ),
                ]
            
            self.model = YOLOXOCC(
                    preproc=preproc,
                    backbone=backbone,  
                    neck=neck, 
                    transform=transform,
                    bev_backbone=bev_backbone,
                    bev_temporal=bev_temporal,
                    bev_neck=bev_neck,
                    occ_head=occ_head, 
                    aux_occ_head_list=aux_occ_head_list,
                    vox_xyz_size=self.vox_xyz_size,
                    world_xyz_bounds=self.world_xyz_bounds,
                    bev_erase_prob=self.bev_erase_prob,
                    bev_flip_prob=self.bev_flip_prob,
                    bev_mixup_prob=self.bev_mixup_prob,
                    bev_mosaic_prob=self.bev_mosaic_prob,
                )
        
        ckpt = torch.load(_CKPT_FULL_PATH, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]

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
