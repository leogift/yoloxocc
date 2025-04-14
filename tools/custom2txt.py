#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Hanqtech, Inc. and its affiliates.

import argparse
import json
import os
import random

def make_parser():
    parser = argparse.ArgumentParser("rknn inference sample")
    parser.add_argument(
        "-v",
        "--val_json",
        type=str,
        default="custom val.json",
        help="Input val json.",
    )
    parser.add_argument(
        "-p",
        "--prob",
        type=float,
        default=0.2,
        help="quant prob.",
    )
    return parser

if __name__ == '__main__':
    random.seed(10927)
    args = make_parser().parse_args()
    with open(args.val_json) as f:
        val_json = json.load(f)
        f.close()

    root = os.path.dirname(args.val_json)
    json_list = []
    for k in val_json.keys():
        json_list.extend(val_json[k])

    quant_str = "wesine.png wesine.png wesine.png\n"
    for j in json_list:
        if random.random() > args.prob:
            continue
        with open(os.path.join(root, j)) as f:
            json_data = json.load(f)
            f.close()
        quant_str += f'{os.path.join(root, json_data["cameras"]["front"]["path"])} \
            {os.path.join(root, json_data["cameras"]["left"]["path"])} \
            {os.path.join(root, json_data["cameras"]["right"]["path"])}\n'
    
    with open("quant.txt", "w") as f:
        f.write(quant_str)
        f.close()
    print(quant_str)