import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from colorama import Fore
import gdown
import yaml


project_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_dir / "src"))

from inv3d_util.path import list_dirs
from src.util import download_and_extract, download_file


model_sources = yaml.safe_load((project_dir / "models.yaml").read_text())


def train(model_part: str, gpu: int, max_cpus: int, limit_samples: Optional[int]):

    # check inv3d
    num_samples = len(list(Path("input/inv3d/").rglob("warped_document.png")))
    if num_samples != 25000:
        print(
            Fore.YELLOW
            + f"Warning: The Inv3D dataset is not complete! Found only {num_samples} of 25000 samples!"
            + Fore.RESET
        )

    if model_part == "sam":
        train_sam(max_cpus, gpu, limit_samples)
    elif model_part == "lineformer":
        train_lineformer(max_cpus, gpu, limit_samples)
    elif model_part == "lightglue":
        train_lightglue(max_cpus, gpu, limit_samples)


def train_sam(max_cpus: int, gpu, limit_samples: Optional[int]):
    pipeline = "inv3d_sam1_former2_proj2_geotmlg1_geotmlg1_former2_glue1_corr2s_res"

    if limit_samples is not None and limit_samples < 8:
        print(
            "Warning: Limiting the number of samples to less than 8 is not possible. Setting limit_samples to 8."
        )
        limit_samples = 8

    kwargs = {
        "config_file": "config.yaml",
        "pipeline": pipeline,
        "max_workers": max_cpus,
        "gpu": gpu,
        "limit_samples": limit_samples,
    }

    # downlaod sam base mode
    sam_base_model = (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    )
    download_file(sam_base_model, Path("models/training/sam/"))

    # prepare dataset
    Runner(**kwargs, split="train").prepare_training("inv3d")
    Runner(**kwargs, split="val").prepare_training("inv3d")
    Runner(**kwargs, split="test").prepare_training("inv3d")

    # prepare sam training
    Runner(**kwargs, split="train").prepare_training("inv3d_sam1")
    Runner(**kwargs, split="val").prepare_training("inv3d_sam1")
    Runner(**kwargs, split="test").prepare_training("inv3d_sam1")

    # train sam
    Runner(**kwargs, split="train").training("inv3d_sam1")


def train_lineformer(max_cpus: int, gpu, limit_samples: Optional[int]):
    pipeline = "inv3d_former2"

    kwargs = {
        "config_file": "config.yaml",
        "pipeline": pipeline,
        "max_workers": max_cpus,
        "gpu": 0,  # use 0 since we use CUDA_VISIBLE_DEVICES to set the GPU
        "limit_samples": limit_samples,
    }

    # downlaod lineformer base mode
    line_former_base_model_dir = project_dir / "models/training/lineformer"
    line_former_base_model_dir.mkdir(parents=True, exist_ok=True)
    lineformer_base_model_url = (
        "https://drive.google.com/u/0/uc?id=1cIWM7lTisd1GajDR98IymDssvvLAKH1n"
    )
    gdown.cached_download(
        url=lineformer_base_model_url,
        path=line_former_base_model_dir / "lineformer_base_iter_3000.pth",
    )

    # download descripable textures dataset
    dtd_path = Path("input/assets/dtd/")
    dtd_zip_url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"

    if len(list_dirs(dtd_path)) != 3:
        shutil.rmtree(dtd_path, ignore_errors=True)
        download_and_extract([dtd_zip_url], dtd_path, unpack_top_level=True)

    # prepare dataset
    Runner(**kwargs, split="train").prepare_training("inv3d")
    Runner(**kwargs, split="val").prepare_training("inv3d")
    Runner(**kwargs, split="test").prepare_training("inv3d")

    # prepare lineformer training
    Runner(**kwargs, split="train").prepare_training("inv3d_former2")
    Runner(**kwargs, split="val").prepare_training("inv3d_former2")
    Runner(**kwargs, split="test").prepare_training("inv3d_former2")

    # train lineformer
    Runner(**kwargs, split="train").training("inv3d_former2")


def train_lightglue(max_cpus: int, gpu, limit_samples: Optional[int]):
    pipeline = "inv3d_former2_glue1"

    kwargs = {
        "config_file": "config.yaml",
        "pipeline": pipeline,
        "max_workers": max_cpus,
        "gpu": 0,  # use 0 since we use CUDA_VISIBLE_DEVICES to set the GPU
        "limit_samples": limit_samples,
    }

    # prepare dataset
    Runner(**kwargs, split="train").inference("inv3d")
    Runner(**kwargs, split="val").inference("inv3d")
    Runner(**kwargs, split="test").inference("inv3d")

    # detect lines in dataset
    Runner(**kwargs, split="train").inference("inv3d_former2")
    Runner(**kwargs, split="val").inference("inv3d_former2")
    Runner(**kwargs, split="test").inference("inv3d_former2")

    # prepare lightglue training
    Runner(**kwargs, split="train").prepare_training("inv3d_former2_glue1")
    Runner(**kwargs, split="val").prepare_training("inv3d_former2_glue1")
    Runner(**kwargs, split="test").prepare_training("inv3d_former2_glue1")

    # train lightglue
    Runner(**kwargs, split="train").training("inv3d_former2_glue1")


if __name__ == "__main__":

    default_cpu_count = max(int(os.cpu_count() * 0.75), 1)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-part",
        type=str,
        required=True,
        choices=["sam", "lineformer", "lightglue"],
        help="The model part to train.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        required=False,
        default=0,
        help="The index of the GPU to use for training.",
    )
    parser.add_argument(
        "--max_cpus",
        type=int,
        required=False,
        default=default_cpu_count,
        help="The maximum number of cpus to use for training.",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        required=False,
        default=None,
        help="Limit the number of dataset samples to process.",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # import the runner after setting the CUDA_VISIBLE_DEVICES environment variable to initialize torch with the correct GPU
    from src.runner import Runner

    train(
        model_part=args.model_part,
        gpu=args.gpu,
        max_cpus=args.max_cpus,
        limit_samples=args.limit_samples,
    )
