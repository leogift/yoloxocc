#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import random

import numpy as np
import torch


# 随机镜像
def augment_mirror(
        bev_numpy_list
    ):
    # B,C,Z,X
    # 左右翻转
    if random.random() < 0.5:
        for idx in range(len(bev_numpy_list)):
            bev_numpy_list[idx] = bev_numpy_list[idx][:, :, :, ::-1].copy()

    # 上下翻转
    if random.random() < 0.5:
        for idx in range(len(bev_numpy_list)):
            bev_numpy_list[idx] = bev_numpy_list[idx][:, :, ::-1, :].copy()

    return bev_numpy_list


# 随机擦除
def augment_erase(
        bev_numpy_list,
        ratio=0.2
    ):
    # B,C,Z,X
    ratio = abs(ratio)

    h_ratio = np.random.uniform(0.05, ratio)
    w_ratio = np.random.uniform(0.05, ratio)
    t_ratio = np.random.uniform(0, 1 - h_ratio)
    l_ratio = np.random.uniform(0, 1 - t_ratio)

    for idx in range(len(bev_numpy_list)):
        Z, X = bev_numpy_list[idx].shape[-2:]
        H = round(h_ratio * Z)
        W = round(w_ratio * X)
        T = round(t_ratio * Z)
        L = round(l_ratio * X)
        bev_numpy_list[idx][:, :, T:T+H, L:L+W] = 0

    return bev_numpy_list

# 将同batch数据任意四张按马赛克方式拼接
def augment_mosaic(
        bev_numpy_list,
    ):
    # 中心点在9宫格中心
    center_x_ratio = np.random.uniform(1/3, 2/3)
    center_y_ratio = np.random.uniform(1/3, 2/3)

    B = bev_numpy_list[0].shape[0]
    mosaic_bev_numpy_list = []
    for idx in range(len(bev_numpy_list)):
        mosaic_bev_numpy_list.append(bev_numpy_list[idx].copy())

    for b in range(B):
        lt_index = b
        # 随机选择1张图片去右上角
        rt_index = np.random.randint(0, B)
        # 随机选择1张图片去左下角
        lb_index = np.random.randint(0, B)
        # 随机选择1张图片去右下角
        rb_index = np.random.randint(0, B)

        # 拼接
        lt_w_ratio = center_x_ratio
        lt_h_ratio = center_y_ratio
        lt_l_ratio = np.random.uniform(0, 1-lt_w_ratio)
        lt_t_ratio = np.random.uniform(0, 1-lt_h_ratio)

        rt_w_ratio = 1 - center_x_ratio
        rt_h_ratio = center_y_ratio
        rt_l_ratio = np.random.uniform(0, 1-rt_w_ratio)
        rt_t_ratio = np.random.uniform(0, 1-rt_h_ratio)

        lb_w_ratio = center_x_ratio
        lb_h_ratio = 1 - center_y_ratio
        lb_l_ratio = np.random.uniform(0, 1-lb_w_ratio)
        lb_t_ratio = np.random.uniform(0, 1-lb_h_ratio)

        rb_w_ratio = 1 - center_x_ratio
        rb_h_ratio = 1 - center_y_ratio
        rb_l_ratio = np.random.uniform(0, 1-rb_w_ratio)
        rb_t_ratio = np.random.uniform(0, 1-rb_h_ratio)

        for idx in range(len(bev_numpy_list)):
            image = bev_numpy_list[idx]
            _, _, H, W = image.shape
            
            # 拼接 mask
            lt_w = round(lt_w_ratio * W)
            lt_h = round(lt_h_ratio * H)
            lt_l = round(lt_l_ratio * W)
            lt_t = round(lt_t_ratio * H)

            rt_w = round(rt_w_ratio * W)
            rt_h = round(rt_h_ratio * H)
            rt_l = round(rt_l_ratio * W)
            rt_t = round(rt_t_ratio * H)

            lb_w = round(lb_w_ratio * W)
            lb_h = round(lb_h_ratio * H)
            lb_l = round(lb_l_ratio * W)
            lb_t = round(lb_t_ratio * H)

            rb_w = round(rb_w_ratio * W)
            rb_h = round(rb_h_ratio * H)
            rb_l = round(rb_l_ratio * W)
            rb_t = round(rb_t_ratio * H)

            # 原图留左上角
            lt_image = image[lt_index]
            rt_image = image[rt_index]
            lb_image = image[lb_index]
            rb_image = image[rb_index]

            # image -> mosaic_image
            mosaic_image = mosaic_bev_numpy_list[idx]
            mosaic_image[b, :,  :lt_h,  :lt_w] = lt_image[:, lt_t:lt_t+lt_h, lt_l:lt_l+lt_w]
            mosaic_image[b, :,  :rt_h, -rt_w:] = rt_image[:, rt_t:rt_t+rt_h, rt_l:rt_l+rt_w]
            mosaic_image[b, :, -lb_h:,  :lb_w] = lb_image[:, lb_t:lb_t+lb_h, lb_l:lb_l+lb_w]
            mosaic_image[b, :, -rb_h:, -rb_w:] = rb_image[:, rb_t:rb_t+rb_h, rb_l:rb_l+rb_w]

    return mosaic_bev_numpy_list


class BEVAugment():
    def __init__(self,
            bev_erase_prob=0.5, # probability of erase
            bev_flip_prob=0.5, # probability of flip
            bev_mosaic_prob=0.5, # probability of mosaic
        ):
        self._bev_erase_prob = bev_erase_prob
        self._bev_flip_prob = bev_flip_prob
        self._bev_mosaic_prob = bev_mosaic_prob

        self.enable_aug()

    def enable_aug(self):
        self.bev_erase_prob = self._bev_erase_prob
        self.bev_flip_prob = self._bev_flip_prob
        self.bev_mosaic_prob = self._bev_mosaic_prob

    def disable_aug(self):
        self.bev_erase_prob = 0
        self.bev_flip_prob = 0
        self.bev_mosaic_prob = 0

    def __call__(self,
            occ_centermask_tensor,
            datas = [],
        ):
        bev_tensor_list = [occ_centermask_tensor]
        key_list = ["occ_centermask"]
        if "valid_vox_mask" in datas:
            bev_tensor_list.append(datas["valid_vox_mask"])
            key_list.append("valid_vox_mask")
        if "trans3" in datas:
            bev_tensor_list.append(datas["trans3"])
            key_list.append("trans3")
        if "trans4" in datas:
            bev_tensor_list.append(datas["trans4"])
            key_list.append("trans4")
        if "trans5" in datas:
            bev_tensor_list.append(datas["trans5"])
            key_list.append("trans5")
        
        device = bev_tensor_list[0].device
        with torch.no_grad():
            bev_numpy_list = []
            for bev_tensor in bev_tensor_list:
                bev_numpy_list.append(bev_tensor.detach().cpu().numpy())

            # B,C,Z,X
            # 随机翻转
            if np.random.random() < self.bev_flip_prob:
                bev_numpy_list = augment_mirror(
                        bev_numpy_list
                    )
            # 马赛克
            if np.random.random() < self.bev_mosaic_prob:
                bev_numpy_list = augment_mosaic(
                            bev_numpy_list
                        )
            # 随机擦除
            if np.random.random() < self.bev_erase_prob:
                bev_numpy_list = augment_erase(
                            bev_numpy_list
                        )

            bev_tensor_list = []
            for bev_numpy in bev_numpy_list:
                bev_tensor_list.append(torch.from_numpy(bev_numpy).to(device))

            for idx, (key, bev_tensor) in enumerate(zip(key_list, bev_tensor_list)):
                if idx == 0:
                    occ_centermask_tensor = bev_tensor
                else:
                    datas[key] = bev_tensor
            
            return occ_centermask_tensor, datas


if __name__ == "__main__":
    # Test the BEVAugment class
    bev_augment = BEVAugment(
            bev_erase_prob=1, # probability of erase
            bev_flip_prob=1, # probability of flip
            bev_mosaic_prob=1, # probability of mosaic
    )
    white = torch.ones(1, 1, 64, 64)
    gray1 = torch.ones(1, 1, 64, 64)*0.7
    gray2 = torch.ones(1, 1, 64, 64)*0.3
    black = torch.zeros(1, 1, 64, 64)
    bev_pred_s2_tensor = torch.cat([white, gray1, gray2, black], dim=0)  # Example tensor with two images
    bev_centermask_target_tensor = torch.nn.functional.interpolate(bev_pred_s2_tensor, size=(128, 128), mode='nearest')  # Example target tensor

    bev_tensor_list = [bev_centermask_target_tensor, bev_pred_s2_tensor]

    import cv2
    bev_s2_tensor_np = bev_tensor_list[0].squeeze()[0].numpy()
    cv2.imwrite("before_aug.png", (bev_s2_tensor_np*255).astype(np.uint8))

    bev_tensor_list = bev_augment(
        bev_tensor_list
    )
    bev_s2_tensor_np = bev_tensor_list[0].squeeze()[0].numpy()
    cv2.imwrite("after_aug.png", (bev_s2_tensor_np*255).astype(np.uint8))

