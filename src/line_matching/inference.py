from typing import Dict
from colorama import Fore
from tqdm import tqdm
from pathlib import Path

import multiprocessing
import atexit

from .feature_extractor.extract import extract_all_frenet_features, _extract_task
from .line_lightglue.inference import inference_lightglue
from .line_lightglue.visualize import visualize_matching_inner
from .update_lines import update_lines


def inference(
    feature_dir: Path,
    inputs: Dict,
    outputs: Dict,
    split: str,
    feature_params: Dict,
    statistics: Dict,
    model_checkpoint: str,
    min_num_features: int,
    num_workers: int,
    gpu: int,
    visualize: bool,
):
    print(Fore.GREEN + "[STAGE] Matching lines" + Fore.RESET)

    multiprocessing.set_start_method("spawn", force=True)
    atexit.register(end_processes)

    template_feature_dir = feature_dir / "features" / "template" / split
    template_feature_dir.mkdir(parents=True, exist_ok=True)

    warped_feature_dir = feature_dir / "features" / "warped" / split
    warped_feature_dir.mkdir(parents=True, exist_ok=True)

    extract_all_frenet_features(
        lines_dir=inputs["template_lines"],
        output_dir=template_feature_dir,
        image_hq_dir=inputs["template_images_HQ"],
        feature_params=feature_params,
        num_workers=num_workers,
    )

    extract_all_frenet_features(
        lines_dir=inputs["warped_lines"],
        output_dir=warped_feature_dir,
        image_hq_dir=inputs["warped_images_HQ"],
        feature_params=feature_params,
        num_workers=num_workers,
    )

    update_lines(
        input_lines_dir=inputs["warped_lines"],
        output_lines_dir=outputs["warped_lines"],
        features_dir=warped_feature_dir,
    )

    inference_lightglue(
        inputs=inputs,
        outputs=outputs,
        template_feature_dir=template_feature_dir,
        warped_feature_dir=warped_feature_dir,
        model_checkpoint=model_checkpoint,
        statistics=statistics,
        split=split,
        min_num_features=min_num_features,
        gpu=gpu,
        num_workers=num_workers,
        visualize=visualize,
    )


def end_processes():
    [proc.terminate() for proc in multiprocessing.active_children()]
