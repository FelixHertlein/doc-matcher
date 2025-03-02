from typing import Dict
from pathlib import Path
from colorama import Fore
import tqdm
import multiprocessing
import atexit

from inv3d_util.parallel import process_tasks

from .prepare_lines.prepare_lines import prepare_lines
from .lineformer.infer import load_model_cached


def inference(
    inputs: Dict,
    outputs: Dict,
    model: Path,
    model_resolution: int,
    config_file: Path,
    parameters: Dict,
    num_workers: int,
    data_dir: str,
    split: str,
    with_borders: bool,
    gpu: int,
    visualize: bool,
):
    print(Fore.GREEN + "[STAGE] Detecting lines in documents" + Fore.RESET)

    multiprocessing.set_start_method("spawn", force=True)
    atexit.register(end_processes)

    # prepare template lines
    prepare_lines(
        inputs=inputs,
        outputs=outputs,
        data_dir=data_dir,
        split=split,
        resolution=512,
        with_borders=with_borders,
        num_workers=num_workers,
        visualize=visualize,
    )

    images = list(inputs["warped_images"].glob("*.jpg"))
    images = sorted(images)

    tasks = []

    for image_path in tqdm.tqdm(images, desc="Detecting lines"):
        image_HQ_path = inputs["warped_images_HQ"] / image_path.name
        output_json_file = outputs["warped_lines"] / f"{image_path.stem}.json"
        output_png_file = outputs["warped_lines"] / f"{image_path.stem}.png"

        tasks.append(
            {
                "image_path": image_path.as_posix(),
                "image_HQ_path": image_HQ_path.as_posix(),
                "ckpt": model.as_posix(),
                "model_resolution": model_resolution,
                "config": config_file.as_posix(),
                "output_json_file": output_json_file.as_posix(),
                "output_png_file": output_png_file.as_posix(),
                "device": f"cuda:{gpu}",
                "visualize": visualize,
                "score_threshold": parameters["score_threshold"],
                "max_line_length": parameters["max_line_length"],
                "smooth_sigma": parameters["smooth_sigma"],
                "duplicate_line_thickness": parameters["duplicate_line_thickness"],
                "duplicate_containment_ratio": parameters[
                    "duplicate_containment_ratio"
                ],
                "min_text_margin": parameters["min_text_margin"],
                "max_text_margin": parameters["max_text_margin"],
                "num_text_margin_steps": parameters["num_text_margin_steps"],
                "distraction_ratio": parameters["distraction_ratio"],
            }
        )

    process_tasks(
        detect_lines_task, tasks, num_workers, use_indexes=True, desc="Detecting lines"
    )

    load_model_cached.cache_clear()  # clear cache to avoid memory leak


def detect_lines_task(task):
    # prevent initialization
    from .detect_lines import detect_lines

    detect_lines(**task)


def end_processes():
    [proc.terminate() for proc in multiprocessing.active_children()]
