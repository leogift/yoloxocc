#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np
import torch


# 随机镜像
def augment_mirror(
        image_s2, image_s4, image_s8, 
        mask, 
        instances, 
        prob=0.5
    ):
    height, width = mask.shape[:2]
    # 左右翻转
    if random.random() < prob:
        image_s2 = image_s2[:, ::-1].copy()
        image_s4 = image_s4[:, ::-1].copy()
        image_s8 = image_s8[:, ::-1].copy()
        mask = mask[:, ::-1].copy()
        if len(instances) > 0:
            tmp = width - instances[:, 0] - 1
            instances[:, 0] = width - instances[:, 2] - 1
            instances[:, 2] = tmp

    # 上下翻转
    if random.random() < prob:
        image_s2 = image_s2[::-1, :].copy()
        image_s4 = image_s4[::-1, :].copy()
        image_s8 = image_s8[::-1, :].copy()
        mask = mask[::-1, :].copy()
        if len(instances) > 0:
            tmp = height - instances[:, 1] - 1
            instances[:, 3] = height - instances[:, 3] - 1
            instances[:, 1] = tmp

    return image_s2, image_s4, image_s8, \
        mask, instances


# 随机擦除
def augment_erase(
        image_s2, image_s4, image_s8, 
        ratio=0.2
    ):
    ratio = abs(ratio)

    h_ratio = np.random.uniform(0.05, ratio)
    w_ratio = np.random.uniform(0.05, ratio)
    t_ratio = np.random.uniform(0, 1 - h_ratio)
    l_ratio = np.random.uniform(0, 1 - t_ratio)

    Z, X = image_s2.shape[:2]
    H = round(h_ratio * Z)
    W = round(w_ratio * X)
    T = round(t_ratio * Z)
    L = round(l_ratio * X)
    image_s2[T:T+H, L:L+W] = 0

    Z, X = image_s4.shape[:2]
    H = round(h_ratio * Z)
    W = round(w_ratio * X)
    T = round(t_ratio * Z)
    L = round(l_ratio * X)
    image_s4[T:T+H, L:L+W] = 0

    Z, X = image_s8.shape[:2]
    H = round(h_ratio * Z)
    W = round(w_ratio * X)
    T = round(t_ratio * Z)
    L = round(l_ratio * X)
    image_s8[T:T+H, L:L+W] = 0

    return image_s2, image_s4, image_s8


# 将同组数据任意两张按照一定比例混合
def augment_mixup(
        image_s2, image_s4, image_s8, 
        mask, 
        instances,
    ):
    B = mask.shape[0]
    # 随机选择B张图片
    mixup_index = np.random.randint(0, B, B)
    for i in range(B):
        if i == mixup_index[i]:
            mixup_index[i] = (mixup_index[i] + 1) % B # 避免自己和自己混合

    mixup_image_s2 = image_s2[mixup_index]
    mixup_image_s4 = image_s4[mixup_index]
    mixup_image_s8 = image_s8[mixup_index]
    mixup_mask = mask[mixup_index]
    mixup_instances = instances[mixup_index]

    # 混合
    mix_ratio = np.random.uniform(0.2, 0.8)
    mix_ratio_inv = 1 - mix_ratio
    new_image_s2 = mix_ratio * image_s2 + mix_ratio_inv * mixup_image_s2
    new_image_s4 = mix_ratio * image_s4 + mix_ratio_inv * mixup_image_s4
    new_image_s8 = mix_ratio * image_s8 + mix_ratio_inv * mixup_image_s8
    new_mask = mix_ratio * mask + mix_ratio_inv * mixup_mask
    new_instances = np.concatenate((instances, mixup_instances), axis=1)
    
    return new_image_s2, new_image_s4, new_image_s8,\
        new_mask, new_instances


# 将同batch数据任意四张按马赛克方式拼接
def augment_mosaic(
        image_s2, image_s4, image_s8, 
        mask, 
        instances, 
        max_instances=200,
        prob=0.5
    ):
    if random.random() < prob:
        B, _, H, W = mask.shape
        # 中心点在9宫格中心
        center_x_ratio = np.random.uniform(1/3, 2/3)
        center_y_ratio = np.random.uniform(1/3, 2/3)
        center_x = round(center_x_ratio * W)
        center_y = round(center_y_ratio * H)
        mosaic_mask = np.zeros_like(mask)
        mosaic_instances = [None]*B

        C_s2 = image_s2.shape[1]
        mosaic_image_s2 = np.zeros((B, C_s2, H//2, W//2), dtype=np.float32)
        C_s4 = image_s4.shape[1]
        mosaic_image_s4 = np.zeros((B, C_s4, H//4, W//4), dtype=np.float32)
        C_s8 = image_s8.shape[1]
        mosaic_image_s8 = np.zeros((B, C_s8, H//8, W//8), dtype=np.float32)

        for i in range(B):
            # 原图去左上角
            lt_w_ratio = center_x_ratio
            lt_h_ratio = center_y_ratio
            lt_l_ratio = np.random.uniform(0, 1-lt_w_ratio)
            lt_t_ratio = np.random.uniform(0, 1-lt_h_ratio)

            # 左上角处理标注
            lt_w = round(lt_w_ratio * W)
            lt_h = round(lt_h_ratio * H)
            lt_l = round(lt_l_ratio * W)
            lt_t = round(lt_t_ratio * H)
            lt_instances = instances[i].copy()
            lt_instances[:, 0] = lt_instances[:, 0] - lt_l
            lt_instances[:, 1] = lt_instances[:, 1] - lt_t
            lt_instances[:, 2] = lt_instances[:, 2] - lt_l
            lt_instances[:, 3] = lt_instances[:, 3] - lt_t
            lt_instances_center = ((lt_instances[:, 0] + lt_instances[:, 2]) / 2,
                                    (lt_instances[:, 1] + lt_instances[:, 3]) / 2)
            lt_instances = lt_instances[(lt_instances_center[0]>=0) & (lt_instances_center[0]<=lt_w) & (lt_instances_center[1]>=0) & (lt_instances_center[1]<=lt_h)]
            lt_instances[:, 0] = np.clip(lt_instances[:, 0], 0, lt_w)
            lt_instances[:, 1] = np.clip(lt_instances[:, 1], 0, lt_h)
            lt_instances[:, 2] = np.clip(lt_instances[:, 2], 0, lt_w)
            lt_instances[:, 3] = np.clip(lt_instances[:, 3], 0, lt_h)

            lt_image_s2= image_s2[i].copy()
            lt_image_s4= image_s4[i].copy()
            lt_image_s8= image_s8[i].copy()
            lt_mask = mask[i].copy()
            
            # 随机选择1张图片去右上角
            rt_index = np.random.randint(0, B)
            rt_w_ratio = 1 - center_x_ratio
            rt_h_ratio = center_y_ratio
            rt_l_ratio = np.random.uniform(0, 1-rt_w_ratio)
            rt_t_ratio = np.random.uniform(0, 1-rt_h_ratio)

            # 右上角处理标注
            rt_w = round(rt_w_ratio * W)
            rt_h = round(rt_h_ratio * H)
            rt_l = round(rt_l_ratio * W)
            rt_t = round(rt_t_ratio * H)
            rt_instances = instances[rt_index].copy()
            rt_instances[:, 0] = rt_instances[:, 0] - rt_l + center_x
            rt_instances[:, 1] = rt_instances[:, 1] - rt_t
            rt_instances[:, 2] = rt_instances[:, 2] - rt_l + center_x
            rt_instances[:, 3] = rt_instances[:, 3] - rt_t
            rt_instances_center = ((rt_instances[:, 0] + rt_instances[:, 2]) / 2,
                                    (rt_instances[:, 1] + rt_instances[:, 3]) / 2)
            rt_instances = rt_instances[(rt_instances_center[0]>=center_x) & (rt_instances_center[0]<=W) & (rt_instances_center[1]>=0) & (rt_instances_center[1]<=rt_h)]
            rt_instances[:, 0] = np.clip(rt_instances[:, 0], center_x, W)
            rt_instances[:, 1] = np.clip(rt_instances[:, 1], 0, rt_h)
            rt_instances[:, 2] = np.clip(rt_instances[:, 2], center_x, W)
            rt_instances[:, 3] = np.clip(rt_instances[:, 3], 0, rt_h)

            rt_image_s2 = image_s2[rt_index].copy()
            rt_image_s4 = image_s4[rt_index].copy()
            rt_image_s8 = image_s8[rt_index].copy()
            rt_mask = mask[rt_index].copy()

            # 随机选择1张图片去左下角
            lb_index = np.random.randint(0, B)
            lb_w_ratio = center_x_ratio
            lb_h_ratio = 1 - center_y_ratio
            lb_l_ratio = np.random.uniform(0, 1-lb_w_ratio)
            lb_t_ratio = np.random.uniform(0, 1-lb_h_ratio)

            # 左下角处理标注
            lb_w = round(lb_w_ratio * W)
            lb_h = round(lb_h_ratio * H)
            lb_l = round(lb_l_ratio * W)
            lb_t = round(lb_t_ratio * H)
            lb_instances = instances[lb_index].copy()
            lb_instances[:, 0] = lb_instances[:, 0] - lb_l
            lb_instances[:, 1] = lb_instances[:, 1] - lb_t + center_y
            lb_instances[:, 2] = lb_instances[:, 2] - lb_l
            lb_instances[:, 3] = lb_instances[:, 3] - lb_t + center_y
            lb_instances_center = ((lb_instances[:, 0] + lb_instances[:, 2]) / 2,
                                    (lb_instances[:, 1] + lb_instances[:, 3]) / 2)
            lb_instances = lb_instances[(lb_instances_center[0]>=0) & (lb_instances_center[0]<=lb_w) & (lb_instances_center[1]>=center_y) & (lb_instances_center[1]<=H)]
            lb_instances[:, 0] = np.clip(lb_instances[:, 0], 0, lb_w)
            lb_instances[:, 1] = np.clip(lb_instances[:, 1], center_y, H)
            lb_instances[:, 2] = np.clip(lb_instances[:, 2], 0, lb_w)
            lb_instances[:, 3] = np.clip(lb_instances[:, 3], center_y, H)

            lb_image_s2 = image_s2[lb_index].copy()
            lb_image_s4 = image_s4[lb_index].copy()
            lb_image_s8 = image_s8[lb_index].copy()
            lb_mask = mask[lb_index].copy()

            # 随机选择1张图片去右下角
            rb_index = np.random.randint(0, B)
            rb_w_ratio = 1 - center_x_ratio
            rb_h_ratio = 1 - center_y_ratio
            rb_l_ratio = np.random.uniform(0, 1-rb_w_ratio)
            rb_t_ratio = np.random.uniform(0, 1-rb_h_ratio)

            # 右下角处理标注
            rb_w = round(rb_w_ratio * W)
            rb_h = round(rb_h_ratio * H)
            rb_l = round(rb_l_ratio * W)
            rb_t = round(rb_t_ratio * H)
            rb_instances = instances[rb_index].copy()
            rb_instances[:, 0] = rb_instances[:, 0] - rb_l + center_x
            rb_instances[:, 1] = rb_instances[:, 1] - rb_t + center_y
            rb_instances[:, 2] = rb_instances[:, 2] - rb_l + center_x
            rb_instances[:, 3] = rb_instances[:, 3] - rb_t + center_y
            rb_instances_center = ((rb_instances[:, 0] + rb_instances[:, 2]) / 2,
                                    (rb_instances[:, 1] + rb_instances[:, 3]) / 2)
            rb_instances = rb_instances[(rb_instances_center[0]>=center_x) & (rb_instances_center[0]<=W) & (rb_instances_center[1]>=center_y) & (rb_instances_center[1]<=H)]
            rb_instances[:, 0] = np.clip(rb_instances[:, 0], center_x, W)
            rb_instances[:, 1] = np.clip(rb_instances[:, 1], center_y, H)
            rb_instances[:, 2] = np.clip(rb_instances[:, 2], center_x, W)
            rb_instances[:, 3] = np.clip(rb_instances[:, 3], center_y, H)

            rb_image_s2 = image_s2[rb_index].copy()
            rb_image_s4 = image_s4[rb_index].copy()
            rb_image_s8 = image_s8[rb_index].copy()
            rb_mask = mask[rb_index].copy()

            # 拼接 mask
            mosaic_mask[i, :, 0:lt_h, 0:lt_w] = lt_mask[:, lt_t:lt_t+lt_h, lt_l:lt_l+lt_w]
            mosaic_mask[i, :, 0:rt_h, lt_w:] = rt_mask[:, rt_t:rt_t+rt_h, rt_l:rt_l+rt_w]
            mosaic_mask[i, :, lt_h:, 0:lb_w] = lb_mask[:, lb_t:lb_t+lb_h, lb_l:lb_l+lb_w]
            mosaic_mask[i, :, lt_h:, lt_w:] = rb_mask[:, rb_t:rb_t+rb_h, rb_l:rb_l+rb_w]
            
            # 拼接 stride 2
            lt_w = round(lt_w_ratio * (W//2))
            lt_h = round(lt_h_ratio * (H//2))
            lt_l = round(lt_l_ratio * (W//2))
            lt_t = round(lt_t_ratio * (H//2))
            rt_w = round(rt_w_ratio * (W//2))
            rt_h = round(rt_h_ratio * (H//2))
            rt_l = round(rt_l_ratio * (W//2))
            rt_t = round(rt_t_ratio * (H//2))
            lb_w = round(lb_w_ratio * (W//2))
            lb_h = round(lb_h_ratio * (H//2))
            lb_l = round(lb_l_ratio * (W//2))
            lb_t = round(lb_t_ratio * (H//2))
            rb_w = round(rb_w_ratio * (W//2))
            rb_h = round(rb_h_ratio * (H//2))
            rb_l = round(rb_l_ratio * (W//2))
            rb_t = round(rb_t_ratio * (H//2))
            mosaic_image_s2[i, :, 0:lt_h, 0:lt_w] = lt_image_s2[:, lt_t:lt_t+lt_h, lt_l:lt_l+lt_w]
            mosaic_image_s2[i, :, 0:rt_h, lt_w:] = rt_image_s2[:, rt_t:rt_t+rt_h, rt_l:rt_l+rt_w]
            mosaic_image_s2[i, :, lt_h:, 0:lb_w] = lb_image_s2[:, lb_t:lb_t+lb_h, lb_l:lb_l+lb_w]
            mosaic_image_s2[i, :, lt_h:, lt_w:] = rb_image_s2[:, rb_t:rb_t+rb_h, rb_l:rb_l+rb_w]

            # 拼接 stride 4
            lt_w = round(lt_w_ratio * (W//4))
            lt_h = round(lt_h_ratio * (H//4))
            lt_l = round(lt_l_ratio * (W//4))
            lt_t = round(lt_t_ratio * (H//4))
            rt_w = round(rt_w_ratio * (W//4))
            rt_h = round(rt_h_ratio * (H//4))
            rt_l = round(rt_l_ratio * (W//4))
            rt_t = round(rt_t_ratio * (H//4))
            lb_w = round(lb_w_ratio * (W//4))
            lb_h = round(lb_h_ratio * (H//4))
            lb_l = round(lb_l_ratio * (W//4))
            lb_t = round(lb_t_ratio * (H//4))
            rb_w = round(rb_w_ratio * (W//4))
            rb_h = round(rb_h_ratio * (H//4))
            rb_l = round(rb_l_ratio * (W//4))
            rb_t = round(rb_t_ratio * (H//4))
            mosaic_image_s4[i, :, 0:lt_h, 0:lt_w] = lt_image_s4[:, lt_t:lt_t+lt_h, lt_l:lt_l+lt_w]
            mosaic_image_s4[i, :, 0:rt_h, lt_w:] = rt_image_s4[:, rt_t:rt_t+rt_h, rt_l:rt_l+rt_w]
            mosaic_image_s4[i, :, lt_h:, 0:lb_w] = lb_image_s4[:, lb_t:lb_t+lb_h, lb_l:lb_l+lb_w]
            mosaic_image_s4[i, :, lt_h:, lt_w:] = rb_image_s4[:, rb_t:rb_t+rb_h, rb_l:rb_l+rb_w]

            # 拼接 stride 8
            lt_w = round(lt_w_ratio * (W//8))
            lt_h = round(lt_h_ratio * (H//8))
            lt_l = round(lt_l_ratio * (W//8))
            lt_t = round(lt_t_ratio * (H//8))
            rt_w = round(rt_w_ratio * (W//8))
            rt_h = round(rt_h_ratio * (H//8))
            rt_l = round(rt_l_ratio * (W//8))
            rt_t = round(rt_t_ratio * (H//8))
            lb_w = round(lb_w_ratio * (W//8))
            lb_h = round(lb_h_ratio * (H//8))
            lb_l = round(lb_l_ratio * (W//8))
            lb_t = round(lb_t_ratio * (H//8))
            rb_w = round(rb_w_ratio * (W//8))
            rb_h = round(rb_h_ratio * (H//8))
            rb_l = round(rb_l_ratio * (W//8))
            rb_t = round(rb_t_ratio * (H//8))
            mosaic_image_s8[i, :, 0:lt_h, 0:lt_w] = lt_image_s8[:, lt_t:lt_t+lt_h, lt_l:lt_l+lt_w]
            mosaic_image_s8[i, :, 0:rt_h, lt_w:] = rt_image_s8[:, rt_t:rt_t+rt_h, rt_l:rt_l+rt_w]
            mosaic_image_s8[i, :, lt_h:, 0:lb_w] = lb_image_s8[:, lb_t:lb_t+lb_h, lb_l:lb_l+lb_w]
            mosaic_image_s8[i, :, lt_h:, lt_w:] = rb_image_s8[:, rb_t:rb_t+rb_h, rb_l:rb_l+rb_w]

            tmp_instances_list = [lt_instances, rt_instances, lb_instances, rb_instances]
            for tmp_instances in tmp_instances_list:
                if tmp_instances.shape[1] > 0:
                    mosaic_instances[i] = np.concatenate([mosaic_instances[i], tmp_instances], axis=0) if mosaic_instances[i] is not None else tmp_instances

            # 限制最大标签数
            if mosaic_instances[i].shape[0] > max_instances:
                inds = np.random.choice(mosaic_instances[i].shape[0], max_instances, replace=False)
                mosaic_instances[i] = mosaic_instances[i][inds]
            elif mosaic_instances[i].shape[0] > 0:
                inds = np.random.choice(mosaic_instances[i].shape[0], max_instances-mosaic_instances[i].shape[0], replace=True)
                mosaic_instances[i] = np.concatenate((mosaic_instances[i], mosaic_instances[i][inds]), axis=0)
            else:
                mosaic_instances[i] = np.ones((max_instances, 5))*(-1)

        mosaic_instances = np.stack(mosaic_instances, axis=0)
        return mosaic_image_s2, mosaic_image_s4, mosaic_image_s8, \
            mosaic_mask, mosaic_instances

    return image_s2, image_s4, image_s8, \
        mask, instances


class BEVAugment:
    def __init__(self,
            max_labels=200, # max number of labels
            bev_erase_prob=0.5, # probability of erase
            bev_flip_prob=0.5, # probability of flip
            bev_mixup_prob=0.5, # probability of mixup
            bev_mosaic_prob=0.5, # probability of mosaic
        ):
        self.max_labels = max_labels

        self._bev_erase_prob = bev_erase_prob
        self._bev_flip_prob = bev_flip_prob
        self._bev_mixup_prob = bev_mixup_prob
        self._bev_mosaic_prob = bev_mosaic_prob

        self.enable_aug()

    def enable_aug(self):
        self.bev_erase_prob = self._bev_erase_prob
        self.bev_flip_prob = self._bev_flip_prob
        self.bev_mixup_prob = self._bev_mixup_prob
        self.bev_mosaic_prob = self._bev_mosaic_prob

    def disable_aug(self):
        self.bev_erase_prob = 0
        self.bev_flip_prob = 0
        self.bev_mixup_prob = 0
        self.bev_mosaic_prob = 0

    def __call__(self, 
            bev_pred_s2_tensor, bev_pred_s4_tensor, bev_pred_s8_tensor, 
            bev_centermask_target_tensor, 
            bev_instances_target_tensor, 
        ):
        with torch.no_grad():
            bev_pred_s2 = bev_pred_s2_tensor.detach().cpu().numpy()
            bev_pred_s4 = bev_pred_s4_tensor.detach().cpu().numpy()
            bev_pred_s8 = bev_pred_s8_tensor.detach().cpu().numpy()
            bev_centermask_target = bev_centermask_target_tensor.detach().cpu().numpy()
            bev_instances_target = bev_instances_target_tensor.detach().cpu().numpy()

            B = bev_pred_s2.shape[0]

            bev_pred_s2_list = []
            bev_pred_s4_list = []
            bev_pred_s8_list = []
            bev_centermask_target_list = []
            bev_instances_target_list = []
            for b in range(B):
                # B,C,Z,X -> C,Z,X -> Z,X,C
                bev_image_s2 = bev_pred_s2[b].transpose(1,2,0).copy()
                bev_image_s4 = bev_pred_s4[b].transpose(1,2,0).copy()
                bev_image_s8 = bev_pred_s8[b].transpose(1,2,0).copy()
                bev_centermask_image = bev_centermask_target[b].transpose(1,2,0).copy()
                bev_instance = bev_instances_target[b]

                # 随机擦除
                if np.random.random() < self.bev_erase_prob:
                    bev_image_s2, bev_image_s4, bev_image_s8 = augment_erase(
                            bev_image_s2, bev_image_s4, bev_image_s8,
                        )
                # 随机翻转
                if np.random.random() < self.bev_flip_prob:
                    bev_image_s2, bev_image_s4, bev_image_s8, \
                        bev_centermask_image, bev_instance = augment_mirror(
                            bev_image_s2, bev_image_s4, bev_image_s8,
                            bev_centermask_image, 
                            bev_instance, 
                            self.bev_flip_prob
                        )

                bev_pred_s2_list.append(bev_image_s2.transpose(2,0,1).copy())
                bev_pred_s4_list.append(bev_image_s4.transpose(2,0,1).copy())
                bev_pred_s8_list.append(bev_image_s8.transpose(2,0,1).copy())
                bev_centermask_target_list.append(bev_centermask_image.transpose(2,0,1).copy())
                bev_instances_target_list.append(bev_instance)

            new_bev_pred_s2 = np.stack(bev_pred_s2_list, axis=0)
            new_bev_pred_s4 = np.stack(bev_pred_s4_list, axis=0)
            new_bev_pred_s8 = np.stack(bev_pred_s8_list, axis=0)
            new_bev_centermask_target = np.stack(bev_centermask_target_list, axis=0)
            new_bev_instances_target = np.stack(bev_instances_target_list, axis=0)

            # 混合
            if np.random.random() < self.bev_mixup_prob:
                new_bev_pred_s2, new_bev_pred_s4, new_bev_pred_s8, \
                    new_bev_centermask_target, new_bev_instances_target = augment_mixup(
                            new_bev_pred_s2, new_bev_pred_s4, new_bev_pred_s8,
                            new_bev_centermask_target, 
                            new_bev_instances_target, 
                        )
            # 马赛克
            if np.random.random() < self.bev_mosaic_prob:
                new_bev_pred_s2, new_bev_pred_s4, new_bev_pred_s8, \
                    new_bev_centermask_target, new_bev_instances_target = augment_mosaic(
                            new_bev_pred_s2, new_bev_pred_s4, new_bev_pred_s8, 
                            new_bev_centermask_target, 
                            new_bev_instances_target, 
                            prob=self.bev_mosaic_prob
                        )
            
            new_bev_pred_s2_tensor = torch.from_numpy(new_bev_pred_s2).to(bev_pred_s2_tensor.device)
            new_bev_pred_s4_tensor = torch.from_numpy(new_bev_pred_s4).to(bev_pred_s4_tensor.device)
            new_bev_pred_s8_tensor = torch.from_numpy(new_bev_pred_s8).to(bev_pred_s8_tensor.device)
            new_bev_centermask_target_tensor = torch.from_numpy(new_bev_centermask_target).to(bev_centermask_target_tensor.device)
            new_bev_instances_target_tensor = torch.from_numpy(new_bev_instances_target).to(bev_instances_target_tensor.device)

            return new_bev_pred_s2_tensor, new_bev_pred_s4_tensor, new_bev_pred_s8_tensor, \
                new_bev_centermask_target_tensor, new_bev_instances_target_tensor
