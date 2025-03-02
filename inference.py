import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import gdown
import yaml
import shutil


project_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_dir / "src"))

from inv3d_util.path import list_dirs
from src.util import download_and_extract

model_sources = yaml.safe_load((project_dir / "models.yaml").read_text())

MODEL_TO_PIPELINE = {
    "identity": "_ident1_res",
    "dewarpnet": "_dwnet1_res",
    "geotr": "_geo1_res",
    "geotr_template": "_geotm1_res",
    "geotr_template_large": "_geotmlg1_res",
    "docmatcher": "_sam1_former2_proj2_geotmlg1_geotmlg1_former2_glue1_corr2s_res",
}


def inference(
    model_name: str, dataset: str, max_cpus: int, limit_samples: Optional[int]
):

    model_parts = model_name.split("@") + [None]
    model_architecture = model_parts[0]
    model_dataset = model_parts[1]

    # dataset name mapping
    if dataset == "inv3d_real":
        dataset_internal = "inv3d"
        download_inv3d_real()
    else:
        dataset_internal = dataset.replace(" ", "").replace("_", "")

    # gather model(s)
    model_url_data = model_sources[model_name]
    if isinstance(model_url_data, str):
        gdown.cached_download(
            url=model_url_data, path=project_dir / f"models/{model_name}.ckpt"
        )
    else:
        for key, value in model_url_data.items():
            gdown.cached_download(
                url=value,
                path=project_dir
                / f"models/{model_architecture}-{key}@{model_dataset}.ckpt",
            )

    pipeline = dataset_internal + MODEL_TO_PIPELINE[model_architecture]

    runner = Runner(
        config_file="config.yaml",
        pipeline=pipeline,
        split="real",
        max_workers=max_cpus,
        gpu=0,
        limit_samples=limit_samples,
    )

    pipeline_parts = pipeline.split("_")

    for i in range(1, len(pipeline_parts) + 1):
        stage_name = "_".join(pipeline_parts[:i])
        # print(f"Running stage {stage_name}")
        runner.inference(stage_name)

    output_dir = project_dir / "output" / f"{dataset}-{model_name}"

    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.copytree(f"cache/collect_results/{pipeline}/results/real", output_dir)


def download_inv3d_real():
    data_dir = Path("input/inv3d/real")
    data_dir.mkdir(parents=True, exist_ok=True)

    num_samples = len(list(data_dir.rglob("warped_*.jpg")))

    if num_samples == 360:
        return

    if num_samples != 360:
        shutil.rmtree(data_dir, ignore_errors=True)
        data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading inv3d_real dataset...")

    inv3d_real_urls = [
        "https://felixhertlein.github.io/inv3d/static/downloads/inv3d_real_part_1_of_2.zip",
        "https://felixhertlein.github.io/inv3d/static/downloads/inv3d_real_part_2_of_2.zip",
    ]

    download_and_extract(inv3d_real_urls, extract_to=data_dir, unpack_top_level=True)


if __name__ == "__main__":

    default_cpu_count = max(int(os.cpu_count() * 0.75), 1)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=list(model_sources.keys()),
        required=True,
        help="Select the model and the dataset used for training.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(
            map(lambda x: x.name, list_dirs(project_dir / "input/custom_datasets"))
        )
        + ["inv3d_real"],
        required=True,
        help="Selects the inference dataset. All folders in the input directory can be selected.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        required=True,
        help="The index of the GPU to use for inference.",
    )
    parser.add_argument(
        "--max_cpus",
        type=int,
        required=False,
        default=default_cpu_count,
        help="The maximum number of cpus to use for inference.",
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

    inference(
        model_name=args.model,
        dataset=args.dataset,
        max_cpus=args.max_cpus,
        limit_samples=args.limit_samples,
    )
