#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Hanqtech, Inc. and its affiliates.

import argparse

import cv2
import numpy as np
from rknn.api import RKNN

def make_parser():
    parser = argparse.ArgumentParser("rknn inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yoloxocc_regnet_x_800mf.onnx",
        help="Input onnx model.",
    )
    parser.add_argument(
        "-c",
        "--cameras",
        type=int,
        default=3,
        help="Number of cameras.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='tools/wesine.png',
        help="Path to your test input image.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=str,
        default="512,288",
        help="Specify an input resolution(WH) for inference.",
    )
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="rk3588",
        help="platform for inference.",
    )
    
    parser.add_argument(
        "-q",
        "--quantization",
        type=str,
        default=None,
        help="Specify an input shape for inference.",
    )
    
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()

    # Create RKNN object
    rknn = RKNN()
    # RKNN config
    print('--> Config model')
    rknn.config(quant_img_RGB2BGR=True, target_platform=args.platform, optimization_level=3, \
                quantized_algorithm="kl_divergence", \
                model_pruning=True, \
                enable_flash_attention=True, \
                disable_rules=['fuse_mul_into_matmul']
            )
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=args.model)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=args.quantization!=None, dataset=args.quantization)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    rknn_model_path = '.'.join(args.model.split(".")[:-1])
    rknn_model_path += "_"+args.platform+".rknn"
    ret = rknn.export_rknn(rknn_model_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # 精度测量
    origin_img = cv2.imread(args.image_path)
    input_shape = tuple(map(int, args.resolution.split(',')))
    resize_img = cv2.resize(origin_img, input_shape)
    resize_img = np.ascontiguousarray(resize_img, dtype=np.float32)
    resize_img = resize_img.transpose((2,0,1))

    rknn.accuracy_analysis(inputs=[resize_img[None,:,:,:], resize_img[None,:,:,:], resize_img[None,:,:,:]], output_dir='./snapshot')

    rknn.release()
