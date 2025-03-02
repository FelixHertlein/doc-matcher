from typing import Dict
import json
from colorama import Fore
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.optimize import minimize
from shapely.geometry import LineString, Polygon
from inv3d_util.load import save_image
from skimage.transform import ProjectiveTransform, warp

from ..line_detection.visualization.visualize_lines import visualize_lines
from .optimize import (
    optim,
    xtoH,
    calculate_norm_factors,
    project_mask,
    project_shapely_line,
)
from .load import load
from .utils import (
    approximate_line,
    project_image_hq,
)


def preunwarp_homography(inputs: Dict, outputs: Dict, split: str, visualize: bool):
    print(
        Fore.GREEN
        + "[STAGE] Pre-unwarping documents using homography estimation"
        + Fore.RESET
    )

    input_lines_dir = inputs["warped_lines"]
    input_masks_dir = inputs["masks"]

    lines_files = list(input_lines_dir.glob("*.json"))

    for lines_file in tqdm(lines_files, desc=f"Preunwarping {split} samples"):

        input_image_hq_file = inputs["warped_images_HQ"] / f"{lines_file.stem}.jpg"
        input_image_file = inputs["warped_images"] / f"{lines_file.stem}.jpg"
        input_mask_file = input_masks_dir / f"{lines_file.stem}.json"

        output_image_hq_file = outputs["warped_images_HQ"] / f"{lines_file.stem}.jpg"
        output_image_file = outputs["warped_images"] / f"{lines_file.stem}.jpg"
        output_mask_file = outputs["masks"] / f"{lines_file.stem}.json"
        output_lines_file = outputs["warped_lines"] / f"{lines_file.stem}.json"

        preunwarp_homography_sample(
            input_image_hq_file=input_image_hq_file,
            input_image_file=input_image_file,
            lines_file=lines_file,
            mask_file=input_mask_file,
            output_image_hq_file=output_image_hq_file,
            output_image_file=output_image_file,
            output_lines_file=output_lines_file,
            output_mask_file=output_mask_file,
            visualize=visualize,
        )


def preunwarp_homography_sample(
    input_image_hq_file: Path,
    input_image_file: Path,
    lines_file: Path,
    mask_file: Path,
    output_image_hq_file: Path,
    output_image_file: Path,
    output_lines_file: Path,
    output_mask_file: Path,
    visualize: bool = False,
):
    data, image, image_orig, mask = load(
        lines_file, mask_file, input_image_hq_file, input_image_file
    )
    lines = data["lines"]

    mask_simple = mask.simplify(10)

    exterior_coords = list(mask_simple.exterior.coords)
    mask_lines = [
        LineString([exterior_coords[i], exterior_coords[i + 1]])
        for i in range(len(exterior_coords) - 1)
    ]

    text_lines = [
        approximate_line(line_data["line"])
        for line_data in lines
        if line_data["type"] == "text_line"
    ]
    structural_lines = [
        approximate_line(line_data["line"])
        for line_data in lines
        if line_data["type"] != "text_line"
    ]

    text_lines_np = np.array([line.coords for line in text_lines])
    structural_lines_np = np.array([line.coords for line in structural_lines])
    mask_lines_np = np.array([line.coords for line in mask_lines])

    # serach for a good start angle
    angles = np.linspace(-np.pi / 2, np.pi / 2, 100)
    scores = np.array(
        [
            optim(
                np.array([angle, 0, 0, 0]),
                text_lines_np,
                structural_lines_np,
                mask_lines_np,
                mask,
                rotation_only=True,
            )
            for angle in angles
        ]
    )

    # intialize the optimization
    angle = angles[np.argmin(scores)]
    shear = 0
    vanishing_0 = 0
    vanishing_1 = 0
    x0 = np.array([angle, shear, vanishing_0, vanishing_1])

    res = minimize(
        optim,
        x0,
        bounds=(
            (angle - np.pi / 4, angle + np.pi / 4),
            (None, None),
            (-1e-3, 1e-3),
            (-1e-3, 1e-3),
        ),
        args=(text_lines_np, structural_lines_np, mask_lines_np, mask),
    )

    # finalize the prespective transformation matrix
    norm_factors = calculate_norm_factors(res.x, mask)
    H = xtoH(res.x)
    P = np.array(
        [
            [
                norm_factors["xfact"],
                0,
                norm_factors["xoff"] * norm_factors["xfact"],
            ],
            [
                0,
                norm_factors["yfact"],
                norm_factors["yoff"] * norm_factors["yfact"],
            ],
            [
                0,
                0,
                1,
            ],
        ],
        dtype=np.float64,
    )
    # apply postprocessing (scale and offset)
    H = np.matmul(P, H)

    # transform mask, lines and image
    transform = ProjectiveTransform(H)
    proj_mask = project_mask(transform, mask)
    proj_mask = Polygon(np.clip(np.array(proj_mask.exterior.coords), 0, 512))

    def project_line(line_data):
        line_data["line"] = project_shapely_line(transform, line_data["line"])
        line_data["line"] = LineString(
            np.clip(np.array(line_data["line"].coords), 0, 512)
        )
        line_data["line"] = line_data["line"].wkt

        return line_data

    transform_data = [{"type": "preunwarp_perspective", "matrix": H.tolist()}]

    data["original_image_path"] = data["image_path"]
    data["image_path"] = str(output_image_file)
    data["lines"] = [project_line(line_data) for line_data in lines]
    data["transforms"] = transform_data

    proj_image = warp(image, transform.inverse, output_shape=(512, 512))
    proj_image = (proj_image * 255).astype(np.uint8)

    proj_image_hq = project_image_hq(image_orig, H)

    # export lines
    output_lines_file.write_text(json.dumps(data))

    # export the image
    save_image(output_image_file, proj_image, override=True)

    # export HQ image
    save_image(output_image_hq_file, proj_image_hq, override=True)

    # export the mask
    output_mask_file.write_text(
        json.dumps(
            {
                "mask": proj_mask.wkt,
                "transforms": transform_data,
            },
        )
    )

    # visualize the results
    if visualize:
        visualize_lines(output_lines_file)
