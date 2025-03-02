from typing import Dict
from colorama import Fore
from tqdm import tqdm
from pathlib import Path

import multiprocessing
import atexit

from .feature_extractor.extract import extract_all_frenet_features
from .update_lines import update_lines


def prepare_training(
    feature_dir: Path,
    inputs: Dict,
    split: str,
    feature_params: Dict,
    num_workers: int,
):
    print(Fore.GREEN + f"[STAGE] Preparing Lightglue training ({split})" + Fore.RESET)

    multiprocessing.set_start_method("spawn", force=True)
    atexit.register(end_processes)

    template_feature_dir = feature_dir / "features" / "template" / split
    template_feature_dir.mkdir(parents=True, exist_ok=True)

    warped_feature_dir = feature_dir / "features" / "warped_GT" / split
    warped_feature_dir.mkdir(parents=True, exist_ok=True)

    extract_all_frenet_features(
        lines_dir=inputs["template_lines"],
        output_dir=template_feature_dir,
        image_hq_dir=inputs["template_images_HQ"],
        feature_params=feature_params,
        num_workers=num_workers,
    )

    extract_all_frenet_features(
        lines_dir=inputs["warped_lines_GT"],
        output_dir=warped_feature_dir,
        image_hq_dir=inputs["warped_images_HQ"],
        feature_params=feature_params,
        num_workers=num_workers,
    )


def end_processes():
    [proc.terminate() for proc in multiprocessing.active_children()]
