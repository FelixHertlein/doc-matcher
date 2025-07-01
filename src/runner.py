from copy import deepcopy
import os
import shutil
import subprocess
from typing import Optional
import torch
import warnings

from util import run_command

warnings.simplefilter("ignore", UserWarning)

os.environ["DOCTR_MULTIPROCESSING_DISABLE"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2

cv2.setNumThreads(0)


from pathlib import Path
import multiprocessing
import atexit
from colorama import init, Fore

init()

from .config import parse_config

from . import preparation
from . import segmentation
from . import line_detection
from . import preunwarp_homography
from . import line_matching
from . import unwarp_correspondence
from . import unwarp_geotr
from . import collect_results


class Runner:
    def __init__(
        self,
        config_file: str,
        pipeline: str,
        split: str,
        max_workers: int,
        gpu: int,
        limit_samples: Optional[int],
    ):
        self.config = parse_config(
            config_file=config_file,
            pipeline=pipeline,
            split=split,
            export_parsed_config=True,
        )
        self.pipeline = pipeline
        self.split = split
        self.max_workers = max_workers
        self.gpu = gpu
        self.base_dir = Path(".")
        self.data_dir = Path("cache")
        self.models_dir = Path("models")
        self.limit_samples = limit_samples

        multiprocessing.set_start_method("spawn", force=True)
        atexit.register(end_processes)

        torch.cuda.init()

    def inference(self, stage_name: str):
        stage = self._init_stage(stage_name)
        stage_category = stage["stage_category"]

        MAX_GPU_MEM_GB = {
            "segmentation": 6.6,
            "line_detection": 2.8,
            "unwarp_geotr": 9.7,
            "line_matching": 6.6,
        }

        # reset output directories
        for output in stage["outputs"].values():
            shutil.rmtree(output, ignore_errors=True)
            output.mkdir(parents=True, exist_ok=True)

        def limit_workers(max_gpu_mem_gb: float):
            available_gpu_mem_gb = (
                torch.cuda.get_device_properties(self.gpu).total_memory / 1e9
            )

            max_workers_by_mem = int(available_gpu_mem_gb / max_gpu_mem_gb)
            max_workers = min(self.max_workers, max_workers_by_mem)
            if self.limit_samples:
                max_workers = min(max_workers, self.limit_samples)

            if max_workers < self.max_workers:
                print(
                    f"Limiting number of workers to {max_workers} due to memory constraints."
                )
            return max_workers

        if stage_category == "preparation":
            preparation.prepare_data(
                data_dir=self.base_dir / stage["data_dir"],
                resolution=stage["resolution"],
                split=stage["split"],
                outputs=stage["outputs"],
                is_generic_dataset=stage["is_generic"],
                limit_samples=self.limit_samples,
            )

        if stage_category == "segmentation":
            segmentation.inference(
                inputs=stage["inputs"],
                outputs=stage["outputs"],
                model_checkpoint=self.models_dir / stage["model"],
                model_original=self.models_dir / stage["model_original"],
                model_type=stage["model_type"],
                max_rotation_angle=stage["max_rotation_angle"],
                gpu=self.gpu,
            )

        if stage_category == "line_detection":
            line_detection.inference(
                inputs=stage["inputs"],
                outputs=stage["outputs"],
                model=self.models_dir / stage["model"],
                model_resolution=stage["coco_params"]["resolution"],
                config_file=self.base_dir / stage["config"],
                parameters=stage["parameters"],
                num_workers=limit_workers(MAX_GPU_MEM_GB["line_detection"]),
                data_dir=self.base_dir / stage["data_dir"],
                split=stage["split"],
                with_borders=stage["coco_params"]["with_borders"],
                gpu=self.gpu,
                visualize=self.split == "real",
            )

        if stage_category == "preunwarp_homography":
            preunwarp_homography.preunwarp_homography(
                inputs=stage["inputs"],
                outputs=stage["outputs"],
                split=self.split,
                visualize=self.split == "real",
            )

        if stage_category == "line_matching":
            line_matching.inference(
                feature_dir=self.data_dir / stage["base_dir"],
                inputs=stage["inputs"],
                outputs=stage["outputs"],
                split=self.split,
                model_checkpoint=self.models_dir / stage["model"],
                feature_params=stage["feature_params"],
                statistics=stage["statistics"],
                min_num_features=stage["min_num_features"],
                num_workers=self.max_workers,
                gpu=self.gpu,
                visualize=self.split == "real",
            )

        if stage_category == "unwarp_correspondence":
            unwarp_correspondence.unwarp_correspondence(
                base_dir=self.data_dir / stage["base_dir"],
                inputs=stage["inputs"],
                outputs=stage["outputs"],
                split=self.split,
                min_text_longest_common_substring=stage[
                    "min_text_longest_common_substring"
                ],
                min_text_length=stage["min_text_length"],
                unwarp_version=stage["unwarp_version"],
                sort_criteria=stage["sort_criteria"],
                max_slope=stage["max_slope"],
                smooth=stage["smooth_value"],
                clip=stage["clip"],
                padding_value=stage["padding_value"],
                padding_blur=stage["padding_blur"],
                num_workers=self.max_workers,
                visualize=self.split == "real",
            )

        if stage_category == "unwarp_geotr":
            unwarp_geotr.inference(
                inputs=stage["inputs"],
                outputs=stage["outputs"],
                model_ckpt=self.models_dir / stage["model"],
                num_workers=limit_workers(MAX_GPU_MEM_GB["unwarp_geotr"]),
                gpu=self.gpu,
                visualize=self.split == "real",
            )

        if stage_category == "collect_results":
            collect_results.inference(
                inputs=stage["inputs"],
                outputs=stage["outputs"],
                data_dir=self.data_dir,
                stage_name=stage_name,
                split=self.split,
            )

        self._finalize_stage(stage)

    def prepare_training(self, stage_name: str):
        stage = self._init_stage(stage_name)
        stage_category = stage["stage_category"]

        def limit_workers(max_num_workers: int = 1):
            if self.max_workers > max_num_workers:
                num_workers = max_num_workers
                print(
                    f"Limiting number of workers to {max_num_workers} due to memory constraints."
                )
            else:
                num_workers = self.max_workers
            return num_workers

        if stage_category == "preparation":
            preparation.prepare_data(
                data_dir=self.base_dir / stage["data_dir"],
                resolution=stage["resolution"],
                split=stage["split"],
                outputs=stage["outputs"],
                is_generic_dataset=stage["is_generic"],
                limit_samples=self.limit_samples,
            )

        if stage_category == "segmentation":
            segmentation.prepare_training(
                base_dir=self.data_dir / stage["base_dir"],
                inputs=stage["inputs"],
                split=stage["split"],
            )

        if stage_category == "line_detection":
            line_detection.prepare_training(
                base_dir=self.data_dir / stage["base_dir"],
                inputs=stage["inputs"],
                outputs=stage["outputs"],
                data_dir=self.base_dir / stage["data_dir"],
                split=stage["split"],
                coco_params=stage["coco_params"],
                num_workers=self.max_workers,
                visualize=self.split == "real",
            )

        if stage_category == "line_matching":
            line_matching.prepare_training(
                feature_dir=self.data_dir / stage["base_dir"],
                inputs=stage["inputs"],
                split=stage["split"],
                feature_params=stage["feature_params"],
                num_workers=self.max_workers,
            )

        self._finalize_stage(stage)

    def training(self, stage_name: str):
        stage = self._init_stage(stage_name)
        stage_category = stage["stage_category"]

        if stage_category == "preparation":
            pass  # nothing to do here

        if stage_category == "segmentation":
            print(Fore.GREEN + f"[STAGE] Training Segment Anything" + Fore.RESET)
            run_command(
                f"/workspaces/doc-matcher/src/segmentation/train_sam.sh {self.gpu}"
            )

        if stage_category == "line_detection":
            print(Fore.GREEN + f"[STAGE] Training Lineformer" + Fore.RESET)
            train_file = "python /workspaces/doc-matcher/src/line_detection/mmdetection_tools/train.py "
            config_file = (
                "src/line_detection/config_training/lineformer_new_transforms.py"
            )
            work_dir = "/workspaces/doc-matcher/models/training/lineformer/work_dir"
            run_command(
                f"{train_file} --gpu-id={self.gpu} --work-dir={work_dir} {config_file}"
            )

        if stage_category == "preunwarp_homography":
            pass  # nothing to do there

        if stage_category == "line_matching":
            print(Fore.GREEN + f"[STAGE] Training Lightglue" + Fore.RESET)
            line_matching.train(
                feature_dir=self.data_dir / stage["base_dir"],
                log_dir=self.models_dir / "training" / "lightglue" / stage_name,
                statistics=stage["statistics"],
                min_num_features=stage["min_num_features"],
                limit_val_samples=stage["limit_val_samples"],
                batch_size=(
                    min(stage["batch_size"], self.limit_samples)
                    if self.limit_samples
                    else stage["batch_size"]
                ),
                num_workers=self.max_workers,
                gpu=self.gpu,
            )

        if stage_category == "unwarp_corresondence":
            pass  # nothting to do here

    def _init_stage(self, stage_name: str) -> dict:
        stage = deepcopy(self.config[stage_name])

        inputs = {key: self.data_dir / value for key, value in stage["inputs"].items()}
        outputs = {
            key: self.data_dir / value for key, value in stage["outputs"].items()
        }

        for output_dir in outputs.values():
            # print(stage_name, output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        stage["inputs"] = inputs
        stage["outputs"] = outputs

        return stage

    def _finalize_stage(self, stage: dict):
        outputs = stage["outputs"]

        for output_dir in outputs.values():
            for parent in [output_dir] + list(output_dir.parents):
                if not any(parent.iterdir()):
                    parent.rmdir()
                    # print(f"Removed {parent} as it is empty.")
                else:
                    break


def end_processes():
    [proc.terminate() for proc in multiprocessing.active_children()]
