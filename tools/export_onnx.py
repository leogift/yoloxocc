#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech, Inc. and its affiliates.

import argparse
from loguru import logger

import torch
from torch import nn

from yoloxocc.exp import get_exp
from yoloxocc.data import CustomDataset
from yoloxocc.utils import optimize_model

def make_parser():
    parser = argparse.ArgumentParser("YOLOXOCC onnx deploy")
    parser.add_argument(
        "-o", "--output-name", type=str, default="yoloxocc.onnx", help="output name of models"
    )
    parser.add_argument(
        "-s", "--opset", default=12, type=int, help="onnx opset version"
    )
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument("-p", "--perspective-mode", default="remapping", type=str, help="'gridsample' or 'remapping'")

    parser.add_argument(
        "--images",
        action="store_true", 
        default=False,
        help="Use images for inference.",
    )
    
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file)

    valdataset = CustomDataset(
            data_dir=exp.data_dir,
            sequence_json=exp.val_json,
            camera_list=exp.camera_list,
            image_size=exp.image_size,
            lidar_list=exp.lidar_list,
            category_list=exp.category_list,
            augment=None
        )
    cameras_image, cameras_extrin, cameras_intrin, \
            lidars_points, lidars_extrin, \
            cameras_annos, lidars_annos = valdataset.pull_item(0)
    X,Y,Z = exp.vox_xyz_size
    cameras_image = torch.tensor(cameras_image).unsqueeze(0).type(torch.float32)
    cameras_extrin = torch.tensor(cameras_extrin).unsqueeze(0).type(torch.float32)
    cameras_intrin = torch.tensor(cameras_intrin).unsqueeze(0).type(torch.float32)

    if args.batch_size > 1:
        cameras_image = cameras_image.repeat(args.batch_size, 1, 1, 1)
        cameras_extrin = cameras_extrin.repeat(args.batch_size, 1, 1)
        cameras_intrin = cameras_intrin.repeat(args.batch_size, 1, 1)

    logger.info("load dataset done.")

    model = exp.get_model()
    model = model.cpu()
    ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)

    if "model" in ckpt:
        ckpt = ckpt["model"]

    model.load_state_dict(ckpt, strict=True)

    if args.images:
        model.prepare_export_images(cameras_image[:,0], cameras_image[:,1], cameras_image[:,2], \
                                    cameras_extrin, cameras_intrin, perspective_mode=args.perspective_mode)
        input_args = [cameras_image[:,0], cameras_image[:,1], cameras_image[:,2]]
        input_names = ["cameras_image_0", "cameras_image_1", "cameras_image_2"]
    else:
        model.prepare_export(cameras_image, cameras_extrin, cameras_intrin, perspective_mode=args.perspective_mode)
        input_args = [cameras_image]
        input_names = ["cameras_image"]

    model = optimize_model(model)

    model.eval()

    logger.info("load checkpoint done.")

    output_names = ["occ_pred"]

    output_onnx_name = args.output_name.split(".onnx")[0]
    if args.images:
        output_onnx_name += "_images"
    output_onnx_name += f"_{args.perspective_mode}" + ".onnx"
    torch.onnx.export(
        model,
        tuple(input_args),
        output_onnx_name,
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset,
    )
    logger.info("generated onnx model named {}".format(output_onnx_name))

    import onnx
    from onnxsim import simplify

    # use onnx-simplifier to reduce reduent model.
    onnx_model = onnx.load(output_onnx_name)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_onnx_name)
    logger.info("generated simplified onnx model named {}".format(output_onnx_name))

if __name__ == "__main__":
    main()
