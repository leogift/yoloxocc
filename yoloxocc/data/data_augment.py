#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech, Inc. and its affiliates.

import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from yoloxocc.utils import geom

# 色彩抖动
# Hrange 42, Srange 212, Vrange 209
def augment_hsv(img, hgain=42/2, sgain=212/2, vgain=209/2):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    return cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR)


# 运动模糊
def augment_blur(img, kernel=15, angle=180):
    kernel = abs(kernel)
    angle = abs(angle)

    # be sure the kernel size is odd
    kernel = round(np.random.randint(3, kernel))//2*2+1
    angle = np.random.uniform(-angle, angle)

    M = cv2.getRotationMatrix2D((kernel / 2, kernel / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(kernel))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (kernel, kernel))
 
    motion_blur_kernel = motion_blur_kernel / kernel
    blurred = cv2.filter2D(img, -1, motion_blur_kernel)
 
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    # gaussian blur
    blurred = cv2.GaussianBlur(blurred, ksize=(kernel, kernel), sigmaX=0, sigmaY=0)

    return blurred


# 随机擦除
def augment_erase(img, ratio=0.2):
    ratio = abs(ratio)
    
    H,W = img.shape[:2]

    w = np.random.randint(3, round(W*ratio))
    h = np.random.randint(3, round(H*ratio))
    x = np.random.randint(0, W - w)
    y = np.random.randint(0, H - h)

    img[y:y+h, x:x+w] = 114
    return img


# 随机缩放裁剪
def augment_crop(img, ratio=0.2):
    ratio = abs(ratio)

    H,W = img.shape[:2]
    new_H = round(H + np.random.uniform(-H*ratio, H*ratio))
    new_W = round(W + np.random.uniform(-W*ratio, W*ratio))
    ratio_h = new_H / H
    ratio_w = new_W / W

    shift_h = round(np.random.uniform(0, new_H - H))
    shift_w = round(np.random.uniform(0, new_W - W))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    resample_type = [
            Image.NEAREST, 
            Image.BILINEAR,
        ]
    img = img.resize((new_W, new_H), resample=np.random.choice(resample_type))
    
    # 解决PIL.Image.crop()的全0填充问题
    img = img.convert('RGBA')
    mask = Image.new(img.mode, img.size, (255,)*3)

    img = img.crop((shift_w, shift_h, shift_w + W, shift_h + H))
    mask = mask.crop((shift_w, shift_h, shift_w + W, shift_h + H))

    padding = Image.new(img.mode, img.size, (114,)*3)
    img = Image.composite(img, padding, mask)
    img = img.convert('RGB')

    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    return img, [ratio_w, ratio_h], [shift_w, shift_h]


# 数据增强
class DataAugment:
    def __init__(self,
                 hsv_prob=0,
                 blur_prob=0,
                 erase_prob=0,
                 crop_prob=0,
            ):

        self.hsv_prob = hsv_prob
        self.blur_prob = blur_prob
        self.erase_prob = erase_prob
        self.crop_prob = crop_prob

    def __call__(self, 
                 input_image, 
                 input_annos,
                 input_intrinsic):

        image = input_image.copy()
        annos = input_annos.copy()
        intrin = input_intrinsic.copy()

        # 随机色彩抖动
        if np.random.random() < self.hsv_prob:
            image = augment_hsv(image)
        # 随机运动模糊
        if np.random.random() < self.blur_prob:
            image = augment_blur(image)
        # 随机擦除
        if np.random.random() < self.erase_prob:
            image = augment_erase(image)
        # 随机缩放裁剪
        if np.random.random() < self.crop_prob:
            image, ratio_wh, shift_wh = augment_crop(image)
            annos[:, 0] = annos[:, 0] * ratio_wh[0] - shift_wh[0]
            annos[:, 1] = annos[:, 1] * ratio_wh[1] - shift_wh[1]
            annos[:, 2] = annos[:, 2] * ratio_wh[0] - shift_wh[0]
            annos[:, 3] = annos[:, 3] * ratio_wh[1] - shift_wh[1]
            intrin = geom.scale_intrinsics_single(intrin, ratio_wh[0], ratio_wh[1])
            intrin = geom.translate_intrinsics_single(intrin, shift_wh[0], shift_wh[1])

        return image, annos, intrin

