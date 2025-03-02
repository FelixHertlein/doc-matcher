import json
import rasterio
from tqdm import tqdm
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from typing import Tuple
from inv3d_util.mapping import invert_map
from inv3d_util.mapping import apply_map, transform_coords
from shapely.affinity import scale, translate
from shapely.geometry import (
    Polygon,
    Point,
    LineString,
    MultiLineString,
    MultiPoint,
    box,
)
from shapely.ops import nearest_points, split, linemerge, snap
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely import wkt
from pathlib import Path
from inv3d_util.load import load_image, save_image
from inv3d_util.mapping import scale_map
from inv3d_util.parallel import process_tasks

from .utils import NoStdStreams
from ..line_detection.load_lines import load_lines, save_lines
from ..line_detection.visualization.visualize_lines import visualize_lines


def preunwarp_outline(
    input_mask_dir: Path,
    input_image_dir: Path,
    input_image_hq_dir: Path,
    input_warped_lines_dir: Path,
    output_image_dir: Path,
    output_image_hq_dir: Path,
    output_warped_lines_dir: Path,
    num_workers: int,
    visualize: bool,
):
    input_mask_files = list(input_mask_dir.glob("*.json"))
    input_mask_files.sort()

    tasks = []

    for input_mask_file in tqdm(input_mask_files, desc="Preunwarping outline"):

        json_name = input_mask_file.name
        jgp_name = input_mask_file.name.replace(".json", ".jpg")

        input_image_file = input_image_dir / jgp_name
        input_image_hq_file = input_image_hq_dir / jgp_name
        output_image_file = output_image_dir / jgp_name
        output_image_hq_file = output_image_hq_dir / jgp_name
        input_warped_file = input_warped_lines_dir / json_name
        output_warped_file = output_warped_lines_dir / json_name

        tasks.append(
            {
                "input_mask_file": input_mask_file,
                "input_image_file": input_image_file,
                "input_image_hq_file": input_image_hq_file,
                "input_lines_file": input_warped_file,
                "output_image_file": output_image_file,
                "output_image_hq_file": output_image_hq_file,
                "output_lines_file": output_warped_file,
                "visualize": visualize,
            }
        )

    process_tasks(
        preunwarp_outline_sample_task,
        tasks,
        num_workers=num_workers,
        use_indexes=True,
        desc="Preunwarping outline",
    )


def preunwarp_outline_sample_task(task: dict):
    preunwarp_outline_sample(**task)


def preunwarp_outline_sample(
    input_mask_file: Path,
    input_image_file: Path,
    input_image_hq_file: Path,
    input_lines_file: Path,
    output_image_file: Path,
    output_image_hq_file: Path,
    output_lines_file: Path,
    visualize: bool,
):
    bm, uv = create_backward_map_from_mask(input_mask_file)

    image = load_image(input_image_file)
    image_hq = load_image(input_image_hq_file)

    image_mapped = apply_map(image, bm)
    image_hq_mapped = apply_map(
        image_hq, bm, resolution=(image_hq.shape[0], image_hq.shape[1])
    )

    save_image(output_image_file, image_mapped, override=True)
    save_image(output_image_hq_file, image_hq_mapped, override=True)

    preunwarp_lines(
        input_lines_file, uv, output_lines_file, visualize, output_image_file
    )


def preunwarp_lines(
    input_warped_file: Path,
    uv: np.ndarray,
    output_warped_file: Path,
    visualize: bool,
    visualize_warped_image_file: Path,
):
    lines_data = load_lines(input_warped_file)

    def preunwarp_line(line: LineString) -> LineString:
        coords = np.array(line.coords) / 512
        if np.any(np.isnan(coords)):
            return None
        coords = np.roll(coords, shift=1, axis=-1)
        coords = np.clip(coords, 0, 1)
        coords_mapped = transform_coords(uv, coords)
        if np.any(np.isnan(coords_mapped)):
            return None
        coords_mapped = np.roll(coords_mapped, shift=1, axis=-1)
        coords_mapped = coords_mapped * 512
        coords_mapped = np.clip(coords_mapped, 0, 512)
        line_mapped = LineString(coords_mapped)
        return line_mapped

    def preunwarp_line_data(line_data: Dict) -> Dict:
        line_data["line"] = preunwarp_line(line_data["line"])
        line_data["words"] = None
        return line_data

    lines_data["lines"] = [
        preunwarp_line_data(line_data) for line_data in lines_data["lines"]
    ]
    lines_data["lines"] = [
        line_data for line_data in lines_data["lines"] if line_data["line"] is not None
    ]
    save_lines(output_warped_file, lines_data)

    if visualize:
        visualize_lines(output_warped_file, visualize_warped_image_file)


def create_backward_map_from_mask(input_mask_file) -> np.ndarray:
    mask = wkt.loads(json.loads(input_mask_file.read_text())["mask"])
    # mask = scale(mask, xfact=0.98, yfact=0.98, origin=mask.centroid) TODO maybe read?

    mask_binary = rasterio.features.rasterize([mask], out_shape=(512, 512)).astype(
        np.bool_
    )

    corners = [Point(0, 0), Point(0, 512), Point(512, 512), Point(512, 0)]
    corners_snap = [nearest_points(c, mask)[1] for c in corners]

    segments = split_polygon(mask, corners_snap)
    segments = classify_segments(segments, corners_snap)

    segments = {
        k: LineString(
            [v.interpolate(alpha, normalized=True) for alpha in np.linspace(0, 1, 1000)]
        )
        for k, v in segments.items()
    }

    grid_y, grid_x = np.mgrid[0:512, 0:512]

    v_border1 = segments["top"]
    v_border2 = segments["bottom"]
    v_values = np.array([0] * len(v_border1.coords) + [1] * len(v_border2.coords))
    v_points = np.array(list(v_border1.coords) + list(v_border2.coords))
    v_interp = LinearNDInterpolator(v_points, v_values)(grid_x, grid_y)
    v_mask = np.isnan(v_interp)
    v_queries_dense = np.array(np.where(v_mask)).T
    v_points_dense = np.array(np.where(~v_mask)).T
    v_values_dense = v_interp[~v_mask]
    v_interp[v_mask] = NearestNDInterpolator(v_points_dense, v_values_dense)(
        v_queries_dense
    )

    h_border1 = segments["left"]
    h_border2 = segments["right"]
    h_values = np.array([0] * len(h_border1.coords) + [1] * len(h_border2.coords))
    h_points = np.array(list(h_border1.coords) + list(h_border2.coords))
    h_interp = LinearNDInterpolator(h_points, h_values)(grid_x, grid_y)
    h_mask = np.isnan(h_interp)
    h_queries_dense = np.array(np.where(h_mask)).T
    h_points_dense = np.array(np.where(~h_mask)).T
    h_values_dense = h_interp[~h_mask]
    h_interp[h_mask] = NearestNDInterpolator(h_points_dense, h_values_dense)(
        h_queries_dense
    )

    uv = np.stack([v_interp, h_interp], axis=-1)
    uv[~mask_binary] = np.nan

    # less resolution leads to errors in the mapping: bm(uv(point)) != point
    with NoStdStreams():
        uv_intermediate = scale_map(uv, resolution=1024)
    bm_intermediate = invert_map(uv_intermediate)

    bm = scale_map(bm_intermediate, 512)

    return bm, uv


def split_polygon(polygon: Polygon, corners: List[Point]) -> List[LineString]:
    boundary = snap(polygon.boundary, MultiPoint(corners), 0.0000001)
    segments = split(boundary, MultiPoint(corners)).geoms
    assert len(segments) in [4, 5]

    if len(segments) == 4:
        return segments

    return [
        linemerge([segments[0], segments[4]]),
        segments[1],
        segments[2],
        segments[3],
    ]


def classify_segments(
    segments: Dict[str, LineString], corners: List[Point]
) -> Dict[str, LineString]:
    bbox = box(*MultiLineString(segments).bounds)
    bbox_points = np.array(bbox.exterior.coords[:-1])

    # classify bbox nodes
    df = pd.DataFrame(data=bbox_points, columns=["bbox_x", "bbox_y"])
    name_x = (df.bbox_x == df.bbox_x.min()).replace({True: "left", False: "right"})
    name_y = (df.bbox_y == df.bbox_y.min()).replace({True: "top", False: "bottom"})
    df["name"] = name_y + "-" + name_x
    assert len(df.name.unique()) == 4

    # find bbox node to corner node association
    approx_points = np.array([c.xy for c in corners]).squeeze()
    assert approx_points.shape == (4, 2)

    assignments = [
        np.roll(np.array(range(4))[::step], shift=i)
        for i in range(4)
        for step in [1, -1]
    ]

    costs = [
        np.linalg.norm(bbox_points - approx_points[assignment], axis=-1).sum()
        for assignment in assignments
    ]

    min_assignment = min(zip(costs, assignments))[1]
    df["corner_x"] = approx_points[min_assignment][:, 0]
    df["corner_y"] = approx_points[min_assignment][:, 1]

    # retrieve correct segment and fix direction if necessary
    segment_endpoints = {frozenset([s.coords[0], s.coords[-1]]): s for s in segments}

    def get_directed_segment(start_name, end_name):
        start = df[df.name == start_name]
        start = (float(start.corner_x.iloc[0]), float(start.corner_y.iloc[0]))

        end = df[df.name == end_name]
        end = (float(end.corner_x.iloc[0]), float(end.corner_y.iloc[0]))

        segment = segment_endpoints[frozenset([start, end])]
        if start != segment.coords[0]:
            segment = LineString(reversed(segment.coords))

        return segment

    return {
        "top": get_directed_segment("top-left", "top-right"),
        "bottom": get_directed_segment("bottom-left", "bottom-right"),
        "left": get_directed_segment("top-left", "bottom-left"),
        "right": get_directed_segment("top-right", "bottom-right"),
    }
