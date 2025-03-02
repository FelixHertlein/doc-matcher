import matplotlib
import pylcs
import tqdm
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from shapely import wkt

import matplotlib.pyplot as plt

import cv2


def visualize_matching(
    output_dir: Path,
    search_path: Path,
    matches_file: Path,
    detector_config: str,
    variant: str = "lineformer",
):

    search_path = Path(search_path)
    matches_file = Path(matches_file)

    sample_id = matches_file.stem[: -len("_matches")]

    [template_file] = list(
        search_path.rglob(f"*{detector_config}_template/[!HQ]*/{sample_id}.jpg")
    )
    [warpedgt_file] = list(
        search_path.rglob(f"*{detector_config}_warpedgt/[!HQ]*/{sample_id}.jpg")
    )
    [template_lines] = list(
        search_path.rglob(
            f"detector/lines/{detector_config}_template/*/{sample_id}_lines.json"
        )
    )
    [warped_lines] = list(
        search_path.rglob(
            f"*detector/lines/{detector_config}_{variant}/*/{sample_id}_lines.json"
        )
    )

    output_file = output_dir / f"{sample_id}_matches.png"

    visualize_matching_inner(
        template_image_file=template_file,
        warped_image_file=warpedgt_file,
        template_lines_file=template_lines,
        warped_lines_file=warped_lines,
        output_file=output_file,
    )


def visualize_matching_preunwarp(
    output_dir: str, matches_file: str, config: str, variant: str
):
    output_dir = Path(output_dir)
    matches_file = Path(matches_file)

    sample_id = matches_file.stem[: -len("_matches")]

    output_file = output_dir / f"{sample_id}_matches.png"

    tempalte_image_file = Path(
        f"data/detector/coco/{config.replace('samv3_', '')}_template/real_images/{sample_id}.jpg"
    )
    warped_image_file = Path(
        f"data/preunwarp_perspective/{config}_{variant}/real/{sample_id}.jpg"
    )

    template_lines_file = Path(
        f"data/detector/lines/{config.replace('samv3_', '')}_template/real/{sample_id}_lines.json"
    )
    warped_lines_file = Path(
        f"data/preunwarp_perspective/{config}_{variant}/real/{sample_id}_lines.json"
    )

    visualize_matching_inner(
        matches_file=matches_file,
        template_image_file=tempalte_image_file,
        warped_image_file=warped_image_file,
        template_lines_file=template_lines_file,
        warped_lines_file=warped_lines_file,
        output_file=output_file,
    )


def visualize_matching_inner(
    matches_file: Path,
    template_image_file: Path,
    warped_image_file: Path,
    template_lines_file: Path,
    warped_lines_file: Path,
    output_file: Path,
):
    output_file.parent.mkdir(exist_ok=True, parents=True)

    template_lines_file = json.loads(template_lines_file.read_text())["lines"]
    warped_lines_file = json.loads(warped_lines_file.read_text())["lines"]

    left_image = cv2.imread(template_image_file.as_posix())
    right_image = cv2.imread(warped_image_file.as_posix())

    fig = plt.figure(figsize=(16, 8))

    matches = json.loads(matches_file.read_text())

    removed_match_lines = [
        line_id
        for match in matches["removed_line_matches"]
        for line_id in match.values()
    ]

    fig.add_subplot(1, 2, 1)
    plt.imshow(left_image)

    def sample_colors_from_cmap(cmap_name, n):
        cmap = plt.get_cmap(cmap_name)
        norm = plt.Normalize(0, n - 1)
        colors = cmap(norm(np.arange(n)))

        hex_colors = [matplotlib.colors.to_hex(color) for color in colors]

        return hex_colors

    num_matches = len(matches["line_matches"])
    colors = sample_colors_from_cmap("viridis", num_matches) if num_matches > 0 else []

    template_matched_lines = [m["template"] for m in matches["line_matches"]]
    warped_matched_lines = [m["warped"] for m in matches["line_matches"]]

    for line_data in template_lines_file:
        line = wkt.loads(line_data["line"])
        line_id = line_data["id"].split("-", 1)[1]

        if line_id in matches["unmatched_template_lines"]:
            plt.plot(*line.xy, color="red", linewidth=2, linestyle="--")
        elif line_id in removed_match_lines:
            plt.plot(*line.xy, color="orange", linewidth=2, linestyle="--")
        else:
            idx = template_matched_lines.index(line_id)
            color = colors[idx]
            plt.plot(*line.xy, color=color, linewidth=3)

    ax = plt.gca()
    ax.set_xlim([-1, 513])
    ax.set_ylim([513, -1])

    fig.add_subplot(1, 2, 2)
    plt.imshow(right_image)

    for line_data in warped_lines_file:
        line = wkt.loads(line_data["line"])
        line_id = line_data["id"].split("-", 1)[1]

        if line_id in matches["unmatched_warped_lines"]:
            plt.plot(*line.xy, color="red", linewidth=2, linestyle="--")
        elif line_id in removed_match_lines:
            plt.plot(*line.xy, color="orange", linewidth=2, linestyle="--")
        else:
            try:
                idx = warped_matched_lines.index(line_id)
                color = colors[idx]
                plt.plot(*line.xy, color=color, linewidth=3)
            except ValueError:
                pass  # if line was filtered out

    ax = plt.gca()
    ax.set_xlim([-1, 513])
    ax.set_ylim([513, -1])

    plt.savefig(output_file.as_posix())
    plt.close()


# python -m src.inv3d_line_matcher.line_lightglue.visualize
if __name__ == "__main__":
    matches_dir = Path(
        "/workspaces/doc-matcher/data/matcher/line_matches/frenet_linewidth_8_feat_8x256_posdepth_8/inv3d_text_v2_512px_3px_lineformer/real"
    )
    matches_files = list(matches_dir.glob("*_matches.json"))
    matches_files = sorted(matches_files, key=lambda x: x.as_posix())

    for matches_file in tqdm.tqdm(matches_files, desc="Visualizing matches"):
        visualize_matching(
            output_dir=matches_dir,
            search_path="/workspaces/doc-matcher/data",
            matches_file=matches_file,
        )
