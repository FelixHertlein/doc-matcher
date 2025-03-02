import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from inv3d_util.image import scale_image
from inv3d_util.mapping import transform_coords
from shapely.geometry import LineString, Point
from skimage.morphology import skeletonize
from typing import Dict, List
from uuid import uuid4


def find_template_structural_lines(
    image: np.ndarray, with_borders: bool, verbose: bool = True
) -> List[Dict]:
    horizontal_lines = _find_horizontal_lines(image, verbose=verbose)
    vertical_lines = _find_vertical_lines(image, verbose=verbose)

    size = image.shape[0]

    template_lines = [*horizontal_lines, *vertical_lines]

    if with_borders:
        # add border lines
        template_lines.extend(
            [
                {
                    "type": "horizontal_line",
                    "id": str(uuid4()),
                    "line": LineString([(0, 0), (1, 0)]),
                },
                {
                    "type": "horizontal_line",
                    "id": str(uuid4()),
                    "line": LineString([(0, 1), (1, 1)]),
                },
                {
                    "type": "vertical_line",
                    "id": str(uuid4()),
                    "line": LineString([(0, 0), (0, 1)]),
                },
                {
                    "type": "vertical_line",
                    "id": str(uuid4()),
                    "line": LineString([(1, 0), (1, 1)]),
                },
            ]
        )

    return template_lines


def _find_horizontal_lines(image: np.ndarray, verbose: bool) -> List[LineString]:
    assert image.shape == (
        2200,
        1700,
        3,
    ), "image must be 2200x1700x3 since absolute kernel size is used"

    horizontal_gap_fill = 15
    vertical_gap_fill = (
        10  # careful: too larger gap fill will cause the lines to be connected
    )
    min_line_search_length = 20
    min_line_length = 50
    allowed_line_deviation = 5

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 0, apertureSize=3)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (min_line_search_length, 1)
    )
    horizontal_mask = cv2.morphologyEx(
        edges, cv2.MORPH_OPEN, horizontal_kernel, iterations=1
    )

    closing_kernel = np.ones((vertical_gap_fill, horizontal_gap_fill), np.uint8)
    closing = cv2.morphologyEx(horizontal_mask, cv2.MORPH_CLOSE, closing_kernel)

    skeleton = skeletonize(closing / 255)

    contours, _ = cv2.findContours(
        skeleton.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []
    for contour in contours:
        contour = contour.squeeze()
        x = contour[:, 0]
        y = contour[:, 1]

        max_deviation = np.absolute(y.mean() - y).max()
        if max_deviation > allowed_line_deviation:
            if verbose:
                print(
                    f"WARNING: found large block of pixels. most likely are multiple lines connected. gap: {max_deviation} pixels. skipping. "
                )
            continue

        x0 = x.min() / image.shape[1]
        x1 = x.max() / image.shape[1]
        y0 = y.mean() / image.shape[0]
        y1 = y.mean() / image.shape[0]

        if x1 - x0 < min_line_length / image.shape[1]:
            continue

        lines.append(LineString([Point(x0, y0), Point(x1, y1)]))

    lines = [
        {
            "type": "horizontal_line",
            "id": str(uuid4()),
            "line": line,
        }
        for line in lines
    ]
    return lines


def _find_vertical_lines(image: np.ndarray, verbose: bool) -> List[LineString]:
    assert image.shape == (
        2200,
        1700,
        3,
    ), "image must be 2200x1700x3 since absolute kernel size is used"

    horizontal_gap_fill = (
        10  # careful: too larger gap fill will cause the lines to be connected
    )
    vertical_gap_fill = 15
    min_line_search_length = 20
    min_line_length = 50
    allowed_line_deviation = 5

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 0, apertureSize=3)

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, min_line_search_length)
    )
    vertical_mask = cv2.morphologyEx(
        edges, cv2.MORPH_OPEN, vertical_kernel, iterations=1
    )

    closing_kernel = np.ones((vertical_gap_fill, horizontal_gap_fill), np.uint8)
    closing = cv2.morphologyEx(vertical_mask, cv2.MORPH_CLOSE, closing_kernel)

    skeleton = skeletonize(closing / 255)

    contours, _ = cv2.findContours(
        skeleton.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []
    for contour in contours:
        contour = contour.squeeze()
        x = contour[:, 0]
        y = contour[:, 1]

        max_deviation = np.absolute(x.mean() - x).max()
        if max_deviation > allowed_line_deviation:
            if verbose:
                print(
                    f"WARNING: found large block of pixels. most likely are multiple lines connected. gap: {max_deviation} pixles. skipping. "
                )
            continue

        x0 = x.mean() / image.shape[1]
        x1 = x.mean() / image.shape[1]
        y0 = y.min() / image.shape[0]
        y1 = y.max() / image.shape[0]

        if y1 - y0 < min_line_length / image.shape[0]:
            continue

        lines.append(LineString([Point(x0, y0), Point(x1, y1)]))

    lines = [
        {
            "type": "vertical_line",
            "id": str(uuid4()),
            "line": line,
        }
        for line in lines
    ]
    return lines


def project_line(line: LineString, bm: np.ndarray) -> LineString:
    bm_rolled = np.roll(bm, axis=2, shift=1)

    points = np.array(
        [line.interpolate(i, normalized=True).xy for i in np.linspace(0, 1, 100)]
    )
    points = points.squeeze()
    points = np.roll(points, axis=1, shift=1)
    trans = transform_coords(bm_rolled, points)
    trans = LineString(trans)

    return trans


def create_mask(lines: List[LineString], height: int, width: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)

    assert len(lines) < 255

    for idx, line in enumerate(lines, 1):
        verts = np.array(line.coords).squeeze()
        verts[:, 0] *= width
        verts[:, 1] *= height
        verts = verts.reshape((-1, 1, 2)).astype("int32")
        color = (idx, idx, idx)

        cv2.polylines(image, [verts], False, color, thickness=5)

    mask = image[:, :, 0]

    return mask


def visualize_lines(image: np.ndarray, lines: Dict[str, List[LineString]]):
    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    H, W, C = image.shape

    for kind, line_list in lines.items():
        for line in line_list:
            verts = np.array(line.coords).squeeze()
            verts[:, 0] *= W
            verts[:, 1] *= H
            verts = verts.reshape((-1, 1, 2)).astype("int32")
            color = (
                1 + random.randint(0, 254),
                random.randint(0, 255),
                random.randint(0, 255),
            )

            cv2.polylines(image, [verts], False, color, thickness=5)

    plt.imshow(image)
    plt.show()
