import cv2
import json
from copy import deepcopy
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.geometry import LineString
import random
import matplotlib.pyplot as plt
from src.line_detection.load_lines import load_lines
from typing import *
import numpy as np
from pathlib import Path

from inv3d_util.mapping import create_identity_map

from .forward_map import ForwardMap
from ...line_matching.feature_extractor.extract import extrude_line
import numpy as np

from .util import is_border_line, is_horizontal_line


def create_backward_map_from_correspondences_v2(
    matches: Dict,
    template_lines: Dict,
    warped_lines: Dict,
    warped_image_file: Path,
    max_slope: float,
    smooth: Optional[int],
    clip: bool,
    visualize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:

    warped_lines_map = {line["id"].split("-", 1)[1]: line for line in warped_lines}
    template_lines_map = {line["id"].split("-", 1)[1]: line for line in template_lines}

    def filter_fn(match):
        return (
            match["warped"] in warped_lines_map
            and match["template"] in template_lines_map
        )

    matches_filtered = [m for m in matches["line_matches"] if filter_fn(m)]

    unmatched_lines = matches["unmatched_warped_lines"] + [
        m["warped"] for m in matches["removed_line_matches"]
    ]
    unmatched_lines = [
        warped_lines_map[l] for l in unmatched_lines if l in warped_lines_map
    ]
    unmatched_lines = list(
        sorted(unmatched_lines, key=lambda l: l["line"].length, reverse=True)
    )

    def build_uwp_with_matches(matches):
        uwp = ForwardMap(warped_image_file, min_slope=0, max_slope=max_slope)

        def add_line(warped_line, template_line):
            if np.any(np.isnan(np.array(warped_line["line"].coords))):
                # print("skipping due to nan")  # TODO fix data source
                return

            if is_border_line(warped_line["line"]):
                # print("skipping due to border")
                return

            if warped_line["type"] == "text_line":
                uwp.add_line(
                    warped_line["line"],
                    template_line["line"],
                    "matched",
                    allow_invalid=True,
                )
            else:
                if is_horizontal_line(template_line["line"]):
                    uwp.add_horizontal_line(
                        warped_line["line"],
                        template_line["line"],
                        "matched",
                        allow_invalid=True,
                    )
                else:
                    uwp.add_vertical_line(
                        warped_line["line"],
                        template_line["line"],
                        "matched",
                        allow_invalid=True,
                    )

        for idx, match in enumerate(matches):
            warped_line = warped_lines_map[match["warped"]]
            template_line = template_lines_map[match["template"]]

            add_line(warped_line, template_line)

        return uwp

    uwp = build_uwp_with_matches(matches_filtered)
    if visualize:
        uwp.visualize(show_convex_hull=True, output_file=None)

    def badness(uwp, visualize=False):

        h_diffs = uwp.h_map.interp[..., 1:] - uwp.h_map.interp[..., :-1]
        v_diffs = uwp.v_map.interp[1:] - uwp.v_map.interp[:-1]
        diff = np.stack([h_diffs.reshape(-1), v_diffs.reshape(-1)])

        neg_badness = np.sum(diff < 0)
        pos_badness = np.sum(diff > uwp.h_map.max_slope)

        if visualize:
            print(neg_badness, pos_badness)
            plt.clf()
            plt.subplot(2, 2, 1)
            plt.imshow(h_diffs < 0)
            plt.subplot(2, 2, 2)
            plt.imshow(h_diffs > uwp.h_map.max_slope)
            plt.subplot(2, 2, 3)
            plt.imshow(v_diffs < 0)
            plt.subplot(2, 2, 4)
            plt.imshow(v_diffs > uwp.h_map.max_slope)
            plt.show()

        return neg_badness + pos_badness

    badness(uwp)

    def badness_without_match(idx, matches):
        matches_without = matches[:idx] + matches[idx + 1 :]
        uwp = build_uwp_with_matches(matches_without)
        return badness(uwp)

    def eleminate_match(matches):
        badness_list = [
            (idx, badness_without_match(idx, matches))
            for idx, match in enumerate(matches)
        ]
        badness_list = sorted(badness_list, key=lambda x: x[1])
        remove_idx, remove_badness = badness_list[0]

        matches_without = matches[:remove_idx] + matches[remove_idx + 1 :]
        removed_match = matches[remove_idx]
        return matches_without, remove_badness, removed_match

    matches = matches_filtered
    prev_matches = matches
    prev_badness = None
    final_matches = []

    for _ in range(len(matches)):
        matches, new_badness, removed_match = eleminate_match(matches)

        if prev_badness is not None and new_badness >= prev_badness:
            final_matches = prev_matches
            break

        if new_badness == 0:
            final_matches = matches
            break

        prev_matches = matches
        prev_badness = new_badness

    uwp = build_uwp_with_matches(final_matches)
    uwp.project_points()
    if visualize:
        uwp.visualize(show_convex_hull=True, output_file=None)

    # for line in unmatched_lines:
    def map_unmatched_line(uwp, line):
        if np.any(np.isnan(np.array(line["line"].coords))):
            # print("skipping due to nan")  # TODO fix data source
            return

        if is_border_line(line["line"]):
            # print("skipping due to border")
            return

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

            return line["line"], ref_line, "horizontal"
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

                return line["line"], ref_line, "horizontal"
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

                return line["line"], ref_line, "vertical"

    mapped_lines = [map_unmatched_line(uwp, line) for line in unmatched_lines]
    mapped_lines = [line for line in mapped_lines if line is not None]

    def extend_uwp_with_lines(uwp, mapped_lines):
        uwp2 = deepcopy(uwp)
        for line, ref_line, kind in mapped_lines:
            if kind == "vertical":
                uwp2.add_vertical_line(line, ref_line, "unmatched", allow_invalid=True)
            else:
                uwp2.add_horizontal_line(
                    line, ref_line, "unmatched", allow_invalid=True
                )

        return uwp2

    def badness_without_line(idx, lines, uwp):
        matches_without = lines[:idx] + lines[idx + 1 :]
        uwp = extend_uwp_with_lines(uwp, matches_without)
        return badness(uwp)

    def eleminate_line(lines, uwp):
        badness_list = [
            (idx, badness_without_line(idx, lines, uwp))
            for idx, line in enumerate(lines)
        ]
        badness_list = sorted(badness_list, key=lambda x: x[1])
        remove_idx, remove_badness = badness_list[0]

        lines_without = lines[:remove_idx] + lines[remove_idx + 1 :]
        removed_line = lines[remove_idx]
        return lines_without, remove_badness, removed_line

    lines = mapped_lines
    prev_lines = mapped_lines
    prev_badness = badness(uwp)
    final_lines = mapped_lines
    for _ in range(len(mapped_lines)):
        lines, new_badness, removed_line = eleminate_line(lines, deepcopy(uwp))

        if prev_badness is not None and new_badness >= prev_badness:
            final_lines = prev_lines
            break

        if new_badness == 0:
            final_matches = matches
            break

        prev_lines = lines
        prev_badness = new_badness

    uwp_final = extend_uwp_with_lines(deepcopy(uwp), final_lines)
    if visualize:
        uwp_final.visualize(show_convex_hull=True, output_file=None)

    # create final mappings
    bm = uwp.create_good_bm_map()

    # post process bm
    if smooth is None:  # for backward compatiblity
        size = 25
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
