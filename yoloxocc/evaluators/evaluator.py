#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech, Inc. and its affiliates.

import time
from tqdm import tqdm

import torch

from yoloxocc.utils import is_main_process

from loguru import logger

class Evaluator:
    """
    Evaluation class
    """
    def __init__(
        self,
        dataloader,
    ):
        self.dataloader = dataloader

    def evaluate(
        self, 
        model, 
        half=False,
    ):
        if not is_main_process():
            return None
	
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        outputs_list = []
        inference_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        print("Start evaluate ...")
        for cur_iter, _targets in enumerate(
            tqdm(self.dataloader)
        ):
            with torch.no_grad():
                targets = []
                for target in _targets:
                    target = target.type(tensor_type)
                    targets.append(target)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(*targets)
                
                if is_time_record:
                    infer_end = time.time()
                    inference_time += infer_end - start

            outputs_list.append(outputs)

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_sampler.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["inference"],
                    [a_infer_time],
                )
            ]
        )
        
        dash = "--------"*10+"\n"

        eval_results = {
            "eval_info": time_info + "\n" + dash,
            "eval_metrics": {}
        }

        eval_results = self.evaluate_prediction(outputs_list, eval_results)

        return eval_results


    def evaluate_prediction(self, outputs_list, eval_results):
        """
        Args:
            eval_results: statistics data.
        """
        logger.info("Evaluate occ postprocess.")

        occ_iou_list = []
        for outputs in outputs_list:
            if "occ_iou" in outputs:
                occ_iou_list.append(outputs["occ_iou"])
        
        if len(occ_iou_list) > 0:
            occ_iou = sum(occ_iou_list) / len(occ_iou_list)
            eval_results["eval_metrics"]["occ_iou"] = occ_iou

        for k,v in eval_results["eval_metrics"].items():
            eval_results["eval_info"] += f"{k}: {v:.4f}\n"

        return eval_results
