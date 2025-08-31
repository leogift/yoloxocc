#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech, Inc. and its affiliates.

import cv2
import numpy as np

from yoloxocc.utils import geom

# static resize
def static_resize(img, image_size, gray=False, bgcolor=114):
    if gray:
        padded_img = np.ones((image_size[0], image_size[1]), dtype=np.uint8) * bgcolor
    else:
        padded_img = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * bgcolor
    ratio = min(image_size[0] / img.shape[0], image_size[1] / img.shape[1])
    # 随机缩放模式
    interpolation_types = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
        ]
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
        interpolation=np.random.choice(interpolation_types),
    ).astype(np.uint8)

    padded_img[: int(img.shape[0] * ratio), : int(img.shape[1] * ratio)] = resized_img.astype(np.uint8)

    return padded_img, ratio


def data_preprocess(
        input_image, 
        input_annos,
        input_distort, 
        input_intrin,
        image_size=(288, 512)
    ):
    
    image = input_image.copy()
    annos = input_annos.copy()
    intrin = input_intrin.copy()

    if input_distort is not None:
        # 去畸变
        image = cv2.undistort(image, intrin, input_distort, newCameraMatrix=intrin)

    # 静态缩放
    image, ratio = static_resize(image, image_size)
    # annos: list[category, xmin, ymin, xmax, ymax]
    annos[:, 1:5] = annos[:, 1:5] * ratio
    
    intrin = geom.scale_intrinsics_single(intrin, ratio, ratio)

    return image, annos, intrin
