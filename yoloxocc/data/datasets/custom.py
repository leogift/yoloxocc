#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech, Inc. and its affiliates.

# customDataset format:
'''
sequence dict file format:
{
    "000000": [
        "single.json",
        ...
    ],
    ...
}

single dict file format:
{
    "token": "0000-0000-0000-0000",
    "next": "0000-0000-0000-0001",
    "prev": null,
    "lidars": {
        "top": {
            "path": "20241121/lidars/top/1732154776500.ply",
            "instances": [
                {
                    "location": [0, 1, 2],
                    "rotation": 3,
                    "dimensions": [4, 5, 6],
                    "category": "car"
                },
                {
                    "location": [4, 5, 6],
                    "rotation": 7,
                    "dimensions": [8, 9, 10],
                    "category": "pedestrian"
                }
            ],
            "extrinsics": [0, 1, 0, 0, 0, 0, 1, -1355, 1, 0, 0, 0]
        }
    },
    "cameras":{
        "front": {
            "path": "20241121/cameras/front/1732154776703.jpg",
            "width": 1280,
            "height": 720,
            "instances": [
                {
                    "bbox": [1, 2, 3, 4],
                    "category": "car"
                }
            ],
            "distortion": [-3.910e-03, -3.638e-02, 2.603e-03, -1.265e-03, 1.115e-02],
            "intrinsics": [4.960e+02, 4.957e+02, 6.142e+02, 3.151e+02],
            "extrinsics": [1, 0, 0, 0, 0, 1, 0, -1200, 0, 0, 1, 35]
        },
        "left": {
            "path": "20241121/cameras/left/1732154776703.jpg",
            "width": 1280,
            "height": 720,
            "instances": {
                {
                    "bbox": [1, 2, 3, 4],
                    "category": "pedestrian"
                }
            },
            "distortion": [-3.910e-03, -3.638e-02, 2.603e-03, -1.265e-03, 1.115e-02],
            "intrinsics": [4.960e+02, 4.957e+02, 6.142e+02, 3.151e+02],
            "extrinsics": [0.5, 0, -0.866, -82.5, 0, 1, 0, -1200, 0.866, 0, 0.5, 2.5]
        }
    },
}
'''

import os
from loguru import logger

import cv2
import numpy as np

from yoloxocc.data.datasets import DatasetBase

from yoloxocc.data import get_dataset_root, data_preprocess, DataAugment
from yoloxocc.utils import geom

import json
import open3d as o3d


class CustomDataset(DatasetBase):
    """
    Custom dataset class.
    """
    def __init__(
        self,
        data_dir="custom",
        sequence_json="train.json",
        camera_list=['front','left','right'],
        image_size=(288, 512),
        lidar_list=['top'],
        category_list=['tree', 'pedestrian'],
        augment=None,
        max_lidar_points=40000,
        max_instances=200,
    ):
        """
        Args:
            data_dir (str): dataset root directory
            sequence_json (str): sequence dict file name
            camera_list (list[str]): list of camera names
            image_size (tuple[int]): image size
            lidar_list (list[str]): list of lidar names
            category_list (list[str]): list of category names
            augment: data augmentation strategy
            max_lidar_points (int): max number of lidar points
            max_instances (int): max number of instances
        """
        super().__init__()
        self.data_dir = os.path.join(get_dataset_root(), data_dir)
        
        self.annotations_dict = self._load_annotations(sequence_json)

        self.camera_list = camera_list
        self.image_size = image_size
        self.lidar_list = lidar_list
        self.category_list = category_list

        self.augment = augment

        self.max_lidar_points = max_lidar_points
        self.max_instances = max_instances

        self.datasets = self._preload_all()

    def __len__(self):
        return len(self.datasets)

    def _load_annotations(self, sequence_json):
        # 读取sequence文件
        sequence_dict = {}
        with open(os.path.join(self.data_dir, sequence_json), "r") as f:
            sequence_dict = json.load(f)
            f.close()

        # 读取单个标注文件
        annotations_dict = {}
        for sequence in sequence_dict.keys():
            for single_json in sequence_dict[sequence]:
                with open(os.path.join(self.data_dir, single_json), "r") as f:
                    single_dict = json.load(f)
                    token = single_dict["token"]
                    annotations_dict[token] = single_dict
                    f.close()

        return annotations_dict

    # 除图片，全载入内存
    def _load_camera_data(self, annotation):
        cameras_imagefile_list = []
        cameras_annos_list = []
        cameras_extrin_list = []
        cameras_distort_list = []
        cameras_intrin_list = []

        camera_list = self.camera_list.copy()
        
        # 遍历摄像头
        for camera_name in camera_list:
            camera = annotation["cameras"][camera_name]

            imagefile = os.path.join(self.data_dir, camera["path"])
            cameras_imagefile_list.append(imagefile)
            annos = []

            for instance in camera["instances"]:
                category = instance["category"]
                if category not in self.category_list:
                    continue
                category_id = self.category_list.index(category)
                bbox = instance["bbox"]
                annos.append(np.array([*bbox, category_id]))

            annos = np.stack(annos) if len(annos) > 0 else np.ones((0, 5))*(-1)
            cameras_annos_list.append(annos)

            extrin = geom.merge_extrinsics_single(*camera["extrinsics"])
            intrin = geom.merge_intrinsics_single(*camera["intrinsics"])
            distort = np.array(camera["distortion"])

            cameras_extrin_list.append(extrin)
            cameras_intrin_list.append(intrin)
            cameras_distort_list.append(distort                    )

        return cameras_imagefile_list, cameras_annos_list, cameras_extrin_list, cameras_intrin_list, cameras_distort_list


    # 除点云，全载入内存
    def _load_lidar_data(self, annotation):
        lidars_pointcloudfile_list = []
        lidars_annos_list = []
        lidars_extrin_list = []

        lidar_list = self.lidar_list.copy()
        
        # 遍历激光雷达
        for lidar_name in lidar_list:
            lidar = annotation["lidars"][lidar_name]

            pointcloudfile = os.path.join(self.data_dir, lidar["path"])
            lidars_pointcloudfile_list.append(pointcloudfile)
            annos = []

            for instance in lidar["instances"]:
                category = instance["category"]
                if category not in self.category_list:
                    continue
                category_id = self.category_list.index(category)
                dimensions = instance["dimensions"]
                location = instance["location"]
                rotation = instance["rotation"]
                annos.append(np.array([*location, *dimensions, rotation, category_id]))

            annos = np.stack(annos) if len(annos) > 0 else np.ones((0, 8))*(-1)
            lidars_annos_list.append(annos)

            extrin = geom.merge_extrinsics_single(*lidar["extrinsics"])

            lidars_extrin_list.append(extrin)

        return lidars_pointcloudfile_list, lidars_annos_list, lidars_extrin_list


    def _load_by_token(self, token):
        annotation = self.annotations_dict[token]

        # cameras_imagefile_list: list[image_file] # BGR
        # cameras_annos_list: list[max_instance, 5] # x1,y1,x2,y2,category
        # cameras_extrin_list: list[4, 4] # R,t
        # cameras_intrin_list: list[3, 3] # fx,fy,cx,cy
        # cameras_distort_list: list[5] # k1,k2,p1,p2,k3
        cameras_imagefile_list, cameras_annos_list, cameras_extrin_list, cameras_intrin_list, cameras_distort_list \
            = self._load_camera_data(annotation)

        # lidars_pointcloudfile_list: list[max_lidar_points, 3] # X,Y,Z
        # lidars_annos_list: list[max_instance, 8] # X,Y,Z,R,category,L,W,H
        # lidars_extrin_list: list[4, 4] # R,t
        lidars_pointcloudfile_list, lidars_annos_list, lidars_extrin_list \
            = self._load_lidar_data(annotation)

        return \
            cameras_imagefile_list, cameras_annos_list, cameras_extrin_list, cameras_intrin_list, cameras_distort_list, \
            lidars_pointcloudfile_list, lidars_annos_list, lidars_extrin_list


    def _load_by_index(self, index):
        token = self.annotations_dict.keys()[index]
        return self._load_by_token(token)


    def _preload_all(self):
        datasets = []
        for token in self.annotations_dict.keys():
            dataset = self._load_by_token(token)
            datasets.append(dataset)
    
        return datasets


    # 拉取单个标注 读文件
    def pull_item(self, index, aug=False, image_size=None):
        cameras_imagefile_list, cameras_annos_list, cameras_extrin_list, cameras_intrin_list, cameras_distort_list, \
        lidars_pointcloudfile_list, lidars_annos_list, lidars_extrin_list \
            = self.datasets[index]

        # 读取图片
        cameras_image_list = []
        new_cameras_annos_list = []
        new_cameras_intrin_list = []
        for imagefile, annos, intrin, distort in zip(cameras_imagefile_list, cameras_annos_list, cameras_intrin_list, cameras_distort_list):
            image = cv2.imread(imagefile)
            # 临时缩放
            image, annos, intrin = \
                data_preprocess(image, annos, distort, intrin, image_size=self.image_size)
            # 数据增强
            if aug and self.augment is not None:
                image, annos, intrin  = \
                    self.augment(image, annos, intrin)
            # 多尺度训练
            if aug and image_size is not None and image_size != self.image_size:
                image, annos, intrin = \
                    data_preprocess(image, annos, None, intrin, image_size)
                
            # HWC -> CHW
            image = image.transpose(2, 0, 1).copy()
            cameras_image_list.append(image)

            # 每个摄像头对齐max_instances//2个点
            if annos.shape[0] > self.max_instances//2:
                inds = np.random.choice(annos.shape[0], self.max_instances//2, replace=False)
                annos = annos[inds]
            elif annos.shape[0] > 0:
                inds = np.random.choice(annos.shape[0], self.max_instances//2-annos.shape[0], replace=True)
                annos = np.concatenate([annos, annos[inds]], axis=0)
            else:
                annos = np.ones((self.max_instances//2, 5))*(-1) # 5: x1,y1,x2,y2,category
            new_cameras_annos_list.append(annos)

            new_cameras_intrin_list.append(intrin)

        cameras_image = np.stack(cameras_image_list)
        cameras_annos = np.stack(new_cameras_annos_list)
        cameras_extrin = np.stack(cameras_extrin_list)
        cameras_intrin = np.stack(new_cameras_intrin_list)

        # 读取点云
        points_list = []
        for pointcloudfile in lidars_pointcloudfile_list:
            # 读取ply文件
            pointcloud = o3d.io.read_point_cloud(pointcloudfile)
            # 读取xyz
            points = np.array(pointcloud.points) # Nx3
            # 对齐max_lidar_points个点
            if points.shape[0] > self.max_lidar_points:
                inds = np.random.choice(points.shape[0], self.max_lidar_points, replace=False)
                points = points[inds]
            elif points.shape[0] > 0:
                inds = np.random.choice(points.shape[0], self.max_lidar_points-points.shape[0], replace=True)
                points = np.concatenate([points, points[inds]], axis=0)
            else:
                raise Exception("no points")
            points_list.append(points)
        
        lidars_points = np.stack(points_list)

        new_lidars_annos_list = []
        for annos in lidars_annos_list:
            annos = annos.copy()
            # 世界对齐max_instances个点
            if annos.shape[0] > self.max_instances:
                inds = np.random.choice(annos.shape[0], self.max_instances, replace=False)
                annos = annos[inds]
            elif annos.shape[0] > 0:
                inds = np.random.choice(annos.shape[0], self.max_instances-annos.shape[0], replace=True)
                annos = np.concatenate([annos, annos[inds]], axis=0)
            else:
                annos = np.ones((self.max_instances, 8))*(-1) # 8: X,Y,Z,L,W,H,R,category
            new_lidars_annos_list.append(annos)
        lidars_annos = np.stack(new_lidars_annos_list)
        lidars_extrin = np.stack(lidars_extrin_list)

        return cameras_image, cameras_extrin, cameras_intrin, \
            lidars_points, lidars_extrin, \
            cameras_annos, lidars_annos


    def pull_item_by_token(self, token):
        index = list(self.annotations_dict.keys()).index(token)
        return self.pull_item(index)
