#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Hanqtech, Inc. and its affiliates.

import argparse

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
        "-i",
        "--image_path",
        type=str,
        default='tools/wesine.png',
        help="Path to your test input image.",
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
    rknn.config(quant_img_RGB2BGR=True, target_platform=args.platform, optimization_level=3,
                quantized_algorithm="kl_divergence",
                enable_flash_attention=True,
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
    ret = rknn.build(do_quantization=args.quantization!=None, 
                    dataset=args.quantization)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    rknn_model_path = '.'.join(args.model.split(".")[:-1])
    quant_type = "int8" if args.quantization is not None else "fp16"
    rknn_model_path += "_"+args.platform+f"_{quant_type}.rknn"
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
    rknn.accuracy_analysis(inputs=[args.image_path, args.image_path, args.image_path], output_dir='./snapshot')

    rknn.release()
