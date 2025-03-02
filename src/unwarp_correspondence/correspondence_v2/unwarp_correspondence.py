import json
import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import LineString
from scipy.ndimage import median_filter

from inv3d_util.mapping import apply_map
from inv3d_util.parallel import process_tasks
from inv3d_util.load import load_image, save_image


from .create_backward_map import create_backward_map_from_correspondences_v2
from ...line_detection.load_lines import load_lines
from ..preunwarp_outline import preunwarp_lines as unwarp_lines
from ..correspondence.util import apply_map_v2


def unwarp_correspondence(
    input_image_dir: Path,
    input_image_hq_dir: Path,
    input_matches_dir: Path,
    input_template_lines_dir: Path,
    input_warped_lines_dir: Path,
    output_image_dir: Path,
    output_image_hq_dir: Path,
    max_slope: float,
    smooth: Optional[int],
    clip: bool,
    padding_value: Optional[float],
    padding_blur: bool,
    num_workers: int,
    visualize: bool,
    **kwargs,
):
    input_image_files = list(input_image_dir.glob("*.jpg"))
    input_image_files.sort()

    tasks = []

    for input_image_file in input_image_files:

        json_name = input_image_file.name.replace(".jpg", ".json")
        jgp_name = input_image_file.name

        input_image_hq_file = input_image_hq_dir / jgp_name
        input_matches_file = input_matches_dir / json_name
        input_template_lines_file = input_template_lines_dir / json_name
        input_warped_lines_file = input_warped_lines_dir / json_name
        output_image_file = output_image_dir / jgp_name
        output_image_hq_file = output_image_hq_dir / jgp_name

        tasks.append(
            {
                "input_image_file": input_image_file,
                "input_image_hq_file": input_image_hq_file,
                "input_matches_file": input_matches_file,
                "input_template_lines_file": input_template_lines_file,
                "input_warped_lines_file": input_warped_lines_file,
                "output_image_file": output_image_file,
                "output_image_hq_file": output_image_hq_file,
                "max_slope": max_slope,
                "smooth": smooth,
                "clip": clip,
                "padding_value": padding_value,
                "padding_blur": padding_blur,
                "visualize": visualize,
            }
        )

    # for task in tqdm(tasks, desc="Unwarping corespondence"):
    # unwarp_correspondence_task(task)
    # exit(0)  # TODO: run all tasks in parallel

    process_tasks(
        unwarp_correspondence_task,
        tasks,
        num_workers=num_workers,
        use_indexes=True,
        desc="Unwarping correspondence",
    )


def unwarp_correspondence_task(task: dict):
    unwarp_correspondence_sample(**task)


def unwarp_correspondence_sample(
    input_image_file: Path,
    input_image_hq_file: Path,
    input_matches_file: Path,
    input_template_lines_file: Path,
    input_warped_lines_file: Path,
    output_image_file: Path,
    output_image_hq_file: Path,
    max_slope: float,
    smooth: Optional[int],
    clip: bool,
    padding_value: Optional[float],
    padding_blur: bool,
    visualize: bool,
):
    matches = load_matches(input_matches_file)
    tempalte_lines = load_lines(input_template_lines_file)
    warped_lines = load_lines(input_warped_lines_file)

    bm = create_backward_map_from_correspondences_v2(
        matches=matches,
        template_lines=tempalte_lines["lines"],
        warped_lines=warped_lines["lines"],
        warped_image_file=input_image_file,
        max_slope=max_slope,
        smooth=smooth,
        clip=clip,
        visualize=visualize,
    )

    image = load_image(input_image_file)
    image_hq = load_image(input_image_hq_file)

    image_mapped = apply_map_v2(
        image, bm, padding_value=padding_value, padding_blur=padding_blur
    )
    image_hq_mapped = apply_map_v2(
        image_hq,
        bm,
        resolution=(image_hq.shape[0], image_hq.shape[1]),
        padding_value=padding_value,
        padding_blur=padding_blur,
    )

    save_image(output_image_file, image_mapped, override=True)
    save_image(output_image_hq_file, image_hq_mapped, override=True)

    # uv not matching bm
    # unwarp_lines(
    # input_warped_lines_file,
    # uv,
    # output_warped_lines_file,
    # visualize,
    # output_image_file,
    # )


def load_matches(input_matches_file: Path) -> Dict:
    return json.loads(input_matches_file.read_text())
