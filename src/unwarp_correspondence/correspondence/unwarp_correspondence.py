import json
import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import LineString
from scipy.ndimage import median_filter

from inv3d_util.mapping import apply_map, create_identity_map
from inv3d_util.parallel import process_tasks
from inv3d_util.load import load_image, save_image


from ...line_matching.feature_extractor.extract import extrude_line
from ...line_detection.load_lines import load_lines
from ..preunwarp_outline import preunwarp_lines as unwarp_lines
from .util import is_border_line, is_horizontal_line, apply_map_v2
from .forward_map import ForwardMap


def unwarp_correspondence(
    input_image_dir: Path,
    input_image_hq_dir: Path,
    input_matches_dir: Path,
    input_template_lines_dir: Path,
    input_warped_lines_dir: Path,
    output_image_dir: Path,
    output_image_hq_dir: Path,
    sort_criteria: str,
    max_slope: float,
    smooth: Optional[int],
    clip: bool,
    padding_value: Optional[float],
    padding_blur: bool,
    num_workers: int,
    visualize: bool,
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
                "sort_criteria": sort_criteria,
                "max_slope": max_slope,
                "smooth": smooth,
                "clip": clip,
                "padding_value": padding_value,
                "padding_blur": padding_blur,
                "visualize": visualize,
            }
        )

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
    sort_criteria: str,
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

    bm = create_backward_map_from_correspondences(
        matches,
        tempalte_lines["lines"],
        warped_lines["lines"],
        input_image_file,
        sort_criteria,
        max_slope,
        smooth,
        clip,
        visualize,
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

    # uv does not match bm
    # unwarp_lines(
    # input_warped_lines_file,
    # uv,
    # output_warped_lines_file,
    # visualize,
    # output_image_file,
    # )


def load_matches(input_matches_file: Path) -> Dict:
    return json.loads(input_matches_file.read_text())


def create_backward_map_from_correspondences(
    matches: Dict,
    template_lines: Dict,
    warped_lines: Dict,
    warped_image_file: Path,
    sort_criteria: str,
    max_slope: float,
    smooth: Optional[int],
    clip: bool,
    visualize: bool,
) -> Tuple[np.ndarray, np.ndarray]:

    # prepare matches
    warped_lines_map = {line["id"].split("-", 1)[1]: line for line in warped_lines}
    template_lines_map = {line["id"].split("-", 1)[1]: line for line in template_lines}

    def filter_fn(match):
        return (
            match["warped"] in warped_lines_map
            and match["template"] in template_lines_map
        )

    def sort_by_type_fn(match):
        warped_line = warped_lines_map[match["warped"]]
        return warped_line["type"] == "text_line", len(warped_line["text"])

    def sort_by_assignment_probablity_fn(match):
        return match["log_assignment"]

    sort_fn = {
        "line_type": sort_by_type_fn,
        "assignment_probability": sort_by_assignment_probablity_fn,
    }[sort_criteria.strip()]

    matches_filterd = [m for m in matches["line_matches"] if filter_fn(m)]
    matches_sorted = sorted(matches_filterd, key=sort_fn, reverse=True)

    # prepare unmatched
    unmatched_lines = matches["unmatched_warped_lines"] + [
        m["warped"] for m in matches["removed_line_matches"]
    ]
    unmatched_lines = [
        warped_lines_map[l] for l in unmatched_lines if l in warped_lines_map
    ]
    unmatched_lines = list(
        sorted(unmatched_lines, key=lambda l: l["line"].length, reverse=True)
    )

    # create forward map

    uwp = ForwardMap(warped_image_file, min_slope=0, max_slope=max_slope)

    for idx, match in enumerate(matches_sorted):
        warped_line = warped_lines_map[match["warped"]]
        template_line = template_lines_map[match["template"]]

        if np.any(np.isnan(np.array(warped_line["line"].coords))):
            continue

        if is_border_line(warped_line["line"]):
            continue

        if warped_line["type"] == "text_line":
            uwp.add_line(warped_line["line"], template_line["line"], "matched")
        else:
            if is_horizontal_line(template_line["line"]):
                uwp.add_horizontal_line(
                    warped_line["line"], template_line["line"], "matched"
                )
            else:
                uwp.add_vertical_line(
                    warped_line["line"], template_line["line"], "matched"
                )

    uwp.project_points()

    for line in unmatched_lines:
        if np.any(np.isnan(np.array(line["line"].coords))):
            continue

        if is_border_line(line["line"]):
            continue

        if line["type"] == "text_line":
            text_line = extrude_line(line["line"], 5)
            interp_coords = [
                text_line.interpolate(alpha, normalized=True).xy
                for alpha in np.linspace(0, 1, 10)
            ]
            interp_coords = (
                np.clip(np.array(interp_coords), 0, 511).astype(int).tolist()
            )
            v_values = [uwp.v_map.interp[coord[1], coord[0]] for coord in interp_coords]
            mean_v = np.array(v_values).mean() * 512
            ref_line = LineString([(0, mean_v), (512, mean_v)])

            uwp.add_horizontal_line(line["line"], ref_line, "unmatched")
        else:
            if is_horizontal_line(line["line"]):
                text_line = extrude_line(line["line"], 5)
                interp_coords = [
                    text_line.interpolate(alpha, normalized=True).xy
                    for alpha in np.linspace(0, 1, 10)
                ]
                interp_coords = (
                    np.clip(np.array(interp_coords), 0, 511).astype(int).tolist()
                )
                v_values = [
                    uwp.v_map.interp[coord[1], coord[0]] for coord in interp_coords
                ]
                mean_v = np.array(v_values).mean() * 512
                ref_line = LineString([(0, mean_v), (512, mean_v)])

                uwp.add_horizontal_line(line["line"], ref_line, "unmatched")
            else:
                text_line = extrude_line(line["line"], 5)
                interp_coords = [
                    text_line.interpolate(alpha, normalized=True).xy
                    for alpha in np.linspace(0, 1, 10)
                ]
                interp_coords = (
                    np.clip(np.array(interp_coords), 0, 511).astype(int).tolist()
                )
                h_values = [
                    uwp.h_map.interp[coord[1], coord[0]] for coord in interp_coords
                ]
                mean_h = np.array(h_values).mean() * 512
                ref_line = LineString([(mean_h, 0), (mean_h, 512)])

                uwp.add_vertical_line(line["line"], ref_line, "unmatched")

    # uwp.visualize(Path("tmp.png"), show_convex_hull=True)
    bm = uwp.create_good_bm_map()
    uv = uwp.create_uv_map()

    # post process bm
    if smooth is None:  # for backwards compatiblity
        size = 15
        kernel = np.ones((size, size), np.float32) / (size * size)
        bm = cv2.filter2D(bm, -1, kernel)
    else:
        kernel = np.ones((smooth, smooth), np.float32) / (smooth * smooth)
        id_map = create_identity_map(512)
        diff = bm - id_map
        diff = cv2.filter2D(diff, -1, kernel)
        bm = id_map + diff

    if clip:
        eps = 1e-6
        bm = np.clip(bm, eps, 1 - eps)

    return bm
