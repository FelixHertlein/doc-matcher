from pathlib import Path
from typing import Optional

from colorama import Fore

from .clean_text_lines import clean_text_lines
from .preunwarp_outline import preunwarp_outline
from .correspondence.unwarp_correspondence import (
    unwarp_correspondence as unwarp_correspondence_core,
)
from .correspondence_v2.unwarp_correspondence import (
    unwarp_correspondence as unwarp_correspondence_core_v2,
)


def unwarp_correspondence(
    base_dir: str,
    inputs: dict,
    outputs: dict,
    split: str,
    min_text_longest_common_substring: int,
    min_text_length: int,
    num_workers: int,
    unwarp_version: int,
    sort_criteria: str,
    max_slope: float,
    smooth: Optional[int],
    clip: bool,
    padding_value: Optional[float],
    padding_blur: bool,
    visualize: bool,
):
    print(Fore.GREEN + "[STAGE] Unwarping documents using correspondences" + Fore.RESET)

    base_dir = Path(base_dir)

    template_lines_clean_dir = base_dir / "template_lines_clean" / split
    template_lines_clean_dir.mkdir(parents=True, exist_ok=True)

    warped_lines_clean_dir = base_dir / "warped_lines_clean" / split
    warped_lines_clean_dir.mkdir(parents=True, exist_ok=True)

    warped_images_outline_dir = base_dir / "warped_images_outline" / split
    warped_images_outline_dir.mkdir(parents=True, exist_ok=True)

    warped_images_hq_outline_dir = base_dir / "warped_images_HQ_outline" / split
    warped_images_hq_outline_dir.mkdir(parents=True, exist_ok=True)

    warped_lines_outline_dir = base_dir / "warped_lines_outline" / split
    warped_lines_outline_dir.mkdir(parents=True, exist_ok=True)

    clean_text_lines(
        input_template_line_dir=inputs["template_lines"],
        input_warped_line_dir=inputs["warped_lines"],
        input_matches_dir=inputs["matches"],
        output_template_line_dir=template_lines_clean_dir,
        output_warped_line_dir=warped_lines_clean_dir,
        visualize=visualize,
        template_image_dir=inputs["template_images"],
        warped_image_dir=inputs["warped_images"],
        min_text_longest_common_substring=min_text_longest_common_substring,
        min_text_length=min_text_length,
    )

    if "masks" in inputs:
        preunwarp_outline(
            input_mask_dir=inputs["masks"],
            input_image_dir=inputs["warped_images"],
            input_image_hq_dir=inputs["warped_images_HQ"],
            input_warped_lines_dir=warped_lines_clean_dir,
            output_image_dir=warped_images_outline_dir,
            output_image_hq_dir=warped_images_hq_outline_dir,
            output_warped_lines_dir=warped_lines_outline_dir,
            num_workers=num_workers,
            visualize=visualize,
        )
    else:
        warped_images_outline_dir = inputs["warped_images"]
        warped_images_hq_outline_dir = inputs["warped_images_HQ"]
        warped_lines_outline_dir = warped_lines_clean_dir

    print("Unwarping correspondence...")
    print(f"Unwarp version: {unwarp_version}")

    unwarp_methods = {
        1: unwarp_correspondence_core,
        2: unwarp_correspondence_core_v2,
    }

    unwarp_method = unwarp_methods[unwarp_version]

    unwarp_method(
        input_image_dir=warped_images_outline_dir,
        input_image_hq_dir=warped_images_hq_outline_dir,
        input_matches_dir=inputs["matches"],
        input_template_lines_dir=template_lines_clean_dir,
        input_warped_lines_dir=warped_lines_outline_dir,
        output_image_dir=outputs["warped_images"],
        output_image_hq_dir=outputs["warped_images_HQ"],
        sort_criteria=sort_criteria,
        max_slope=max_slope,
        smooth=smooth,
        clip=clip,
        padding_value=padding_value,
        padding_blur=padding_blur,
        num_workers=num_workers,
        visualize=visualize,
    )
