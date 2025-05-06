import onnxruntime
import numpy as np
import time
import cv2
import sys
import argparse
import os

def make_parser():
    parser = argparse.ArgumentParser("YOLOOCC onnx test")

    parser.add_argument(
        "--onnx", type=str, default="20250222_yoloxocc_regnet_y_200mf_w16x2x12_v64x4x48_best54.38_images_remapping.onnx", help="onnx models"
    )
    parser.add_argument(
        "--model", type=str, default="regnet_x_400mf", help="model name"
    )
    parser.add_argument(
        "--test", type=int, default=0, help="test index [0,1,2]"
    )
    parser.add_argument(
        "--images",
        action="store_true",
        default=False,
        help="Use images for inference",
    )

    return parser

# main
parser = make_parser()
args = parser.parse_args(sys.argv[1:])

# 载入模型
print(args.onnx)
sess = onnxruntime.InferenceSession(args.onnx)

# 载入图像
# 三个相机的图像：FRONT -> LEFT -> RIGHT
img_front = cv2.imread(f"dataset/CAMERA_FRONT/{args.test}.png")
img_left = cv2.imread(f"dataset/CAMERA_LEFT/{args.test}.png")
img_right = cv2.imread(f"dataset/CAMERA_RIGHT/{args.test}.png")

H,W = 288, 512
Z,X = 48, 64

img_front = cv2.resize(img_front, (W, H), interpolation=cv2.INTER_NEAREST)
img_left = cv2.resize(img_left, (W, H), interpolation=cv2.INTER_NEAREST)
img_right = cv2.resize(img_right, (W, H), interpolation=cv2.INTER_NEAREST)

# 转换为NCHW格式
img_front = np.transpose(img_front[None,...], (0, 3, 1, 2)).astype(np.float32)
img_left = np.transpose(img_left[None,...], (0, 3, 1, 2)).astype(np.float32)
img_right = np.transpose(img_right[None,...], (0, 3, 1, 2)).astype(np.float32)

if args.images:
    inputs = {
        "cameras_image_0": img_front,
        "cameras_image_1": img_left,
        "cameras_image_2": img_right,
    }
else:
    cameras_image = np.stack([img_front, img_left, img_right], axis=1)
    inputs = {
        "cameras_image": cameras_image
    }

print("load data done")

# onnxruntime warmup

if args.model in ["regnet_x_800mf", "regnet_x_1_6gf"]:
    if args.model == "regnet_x_800mf":
        D = 672
    elif args.model == "regnet_x_1_6gf":
        D = 912
    temporal_feature = np.zeros((1, D, Z//8, X//8), dtype=np.float32)
    inputs["temporal_feature_in"] = temporal_feature

_ = sess.run(None, inputs)

last_occ_pred = None
start = time.time()
loop = 3
for i in range(loop):
    # inference
    if args.model in ["regnet_x_800mf", "regnet_x_1_6gf"]:
        inputs["temporal_feature_in"] = temporal_feature

    outputs = sess.run(None, inputs)

    occ_pred = outputs[0]
    if args.model in ["regnet_x_800mf", "regnet_x_1_6gf"]:
        temporal_feature = outputs[1]

    last_occ_pred = occ_pred

print("[Inference Time]", (time.time() - start)/loop)

_,Y,Z,X = occ_pred.shape

# 保存结果
os.makedirs(args.model, exist_ok=True)
occ_all = np.zeros((occ_pred.shape[2], occ_pred.shape[3]))

for y in range(Y):
    bev = occ_pred[0, y] #cv2.threshold(occ_pred[0, y], 0.5, 1, cv2.THRESH_TOZERO)[1]
    cv2.imwrite(os.path.join(args.model, f"occ_pred_{y}.png"), (bev * 255).astype(np.uint8))
    occ_all += bev * (128 // (2**y))

cv2.imwrite(os.path.join(args.model, f"occ_pred_all.png"), occ_all.astype(np.uint8))
