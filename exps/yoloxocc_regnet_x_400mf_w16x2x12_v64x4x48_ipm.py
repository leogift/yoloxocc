#!/usr/bin/env python3

import os
import torch
from torch import nn

from yoloxocc.exp import Exp as BaseExp

from loguru import logger

from yoloxocc.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

_CKPT_FULL_PATH = "pretrain/yoloxocc_regnet_x_400mf_y4_ipm.pth"

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
        self.world_xyz_bounds = [-16, 16, -1.5, 0.5, -8, 16] # [xmin, xmax, ymin, ymax, zmin, zmax]
        # voxel size in [x, y, z]
        self.vox_xyz_size = [64, 4, 48]
        self.use_gaussian_mask = True

        self.act = "relu"
        self.max_epoch = 30

        self.model_name = "regnet_x_400mf"

        self.warmup_epochs = 10
        self.no_aug_epochs = 10
        self.data_num_workers = 4
        self.eval_epoch_interval = 5


    def get_model(self):

        if "model" not in self.__dict__:
            from yoloxocc.models import YOLOXOCC, \
            BaseNorm, Regnet, YOLONeckFPN, \
            IPMTrans, BEVAugment, \
            RegnetNeckPAN, \
            OCCHead, \
            C2aLayer
            
            preproc = BaseNorm(trainable=True)

            pp_repeats = 0 if min(self.image_size[0], self.image_size[1])//32 < 7 \
                else (min(self.image_size[0], self.image_size[1])//32 - 7)//6 + 1
            backbone = Regnet(
                self.model_name,
                model_reduce=1,
                act=self.act, 
                pp_repeats=pp_repeats,
                drop_rate=0.1,
            )
            self.channels = backbone.output_channels[-3:]
            neck = YOLONeckFPN(
                in_features=("backbone3", "backbone4", "backbone5"),
                channels=self.channels,
                out_features=("fpn3", "fpn4", "fpn5"),
                act=self.act, 
                layer_type=C2aLayer,
                simple_reshape=True,
                n=1
            )

            transform = IPMTrans(
                in_features=["fpn3", "fpn4", "fpn5"],
                channels=self.channels,
                out_features=["trans3", "trans4", "trans5"],
                act=self.act, 
                layer_type=C2aLayer,
                n=1,
                vox_xyz_size=self.vox_xyz_size,
                world_xyz_bounds=self.world_xyz_bounds,
            )
            bev_augment = BEVAugment(
                bev_erase_prob=self.bev_erase_prob, # probability of erase
                bev_flip_prob=self.bev_flip_prob, # probability of flip
                bev_mosaic_prob=self.bev_mosaic_prob, # probability of mosaic
            )

            bev_pp_repeats = 0 if min(self.vox_xyz_size[0], self.vox_xyz_size[2])//8 < 7 \
                else (min(self.vox_xyz_size[0], self.vox_xyz_size[2])//8 - 7)//6 + 1
            bev_neck = nn.Sequential(*[
                RegnetNeckPAN(
                    self.model_name,
                    model_reduce=4,
                    in_features=["trans3", "trans4", "trans5"],
                    channels=self.channels,
                    out_features=("bev_pan3", "bev_pan4", "bev_pan5"),
                    act=self.act, 
                    layer_type=C2aLayer,
                    n=1,
                    pp_repeats=bev_pp_repeats,
                    drop_rate=0.1,
                ),
                YOLONeckFPN(
                    in_features=("bev_pan3", "bev_pan4", "bev_pan5"),
                    channels=self.channels,
                    out_features=("bev_feat3", "bev_feat4", "bev_feat5"),
                    act=self.act, 
                    layer_type=C2aLayer,
                    simple_reshape=True,
                    n=1
                ),
            ])
            
            occ_head = OCCHead(
                in_feature="bev_feat3",
                in_channel=self.channels[0],
                vox_y=self.vox_xyz_size[1],
                act=self.act, 
                simple_reshape=True
            )
            aux_occ_head_list = [
                OCCHead(
                    in_feature="bev_feat4",
                    in_channel=self.channels[1],
                    vox_y=self.vox_xyz_size[1],
                    act=self.act,
                    aux_head=True
                ),
                OCCHead(
                    in_feature="bev_feat5",
                    in_channel=self.channels[2],
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
                bev_neck=bev_neck,
                occ_head=occ_head, 
                aux_occ_head_list=aux_occ_head_list,
                vox_xyz_size=self.vox_xyz_size,
                world_xyz_bounds=self.world_xyz_bounds,
                use_gaussian_mask=self.use_gaussian_mask,
            )
        
        ckpt = torch.load(_CKPT_FULL_PATH, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]

        for k in list(ckpt.keys()):
            if "loss" in k or "Loss" in k:
                del ckpt[k]
            elif "occ_head.stem.conv.weight" in k \
                    or "aux_occ_head_list.0.stem.conv" in k \
                    or "aux_occ_head_list.1.stem.conv" in k:
                ckpt[k] = ckpt[k].sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)

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
