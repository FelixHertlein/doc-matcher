import json
from uuid import uuid4
import cv2
from pathlib import Path
import numpy as np
import networkx as nx
import random

from einops import repeat
from networkx import from_edgelist
from typing import Dict, List, Optional
from shapely.geometry import LineString, Polygon
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter1d
from shapely.affinity import scale

from inv3d_util.image import scale_image


from .lineformer.infer import load_model_cached, do_instance
from .visualization.visualize_lines import visualize_lines

random.seed(42)


def detect_lines(
    image_path: str,
    image_HQ_path: str,
    ckpt: str,
    model_resolution: int,
    config: str,
    output_json_file: str,
    output_png_file: str,
    device: str = "cpu",  # "cuda:0",
    visualize: bool = False,
    score_threshold: float = 0.3,
    max_line_length: float = 50,
    smooth_sigma: float = 5,
    duplicate_line_thickness: float = 3,
    duplicate_containment_ratio: float = 0.9,
    min_text_margin: float = 1,
    max_text_margin: float = 50,
    num_text_margin_steps: int = 100,
    verbose: bool = False,
    distraction_ratio: Optional[float] = None,
) -> List[LineString]:
    image_path = Path(image_path)
    image_HQ_path = Path(image_HQ_path)

    img = cv2.imread(image_HQ_path.as_posix())
    img = scale_image(img, resolution=model_resolution)

    config = Path(config).absolute().as_posix()
    ckpt = Path(ckpt).absolute().as_posix()
    model = load_model_cached(config=config, ckpt=ckpt, device=device)

    inst_masks = do_instance(model, img, score_thr=score_threshold)

    lines = extract_all_lines(
        inst_masks["line"],
        line_type="visual_line",
        calc_margin=False,
        resolution=model_resolution,
        min_text_margin=min_text_margin,
        max_text_margin=max_text_margin,
        num_text_margin_steps=num_text_margin_steps,
        verbose=verbose,
    )
    text_lines = extract_all_lines(
        inst_masks["text"],
        line_type="text_line",
        calc_margin=True,
        resolution=model_resolution,
        min_text_margin=min_text_margin,
        max_text_margin=max_text_margin,
        num_text_margin_steps=num_text_margin_steps,
        verbose=verbose,
    )

    cleaned_lines = remove_duplicate_lines(
        lines,
        line_thickness=duplicate_line_thickness,
        containment_ratio=duplicate_containment_ratio,
    )
    cleaned_text_lines = remove_duplicate_lines(
        text_lines,
        line_thickness=duplicate_line_thickness,
        containment_ratio=duplicate_containment_ratio,
    )

    long_lines = filter_lines_by_length(cleaned_lines, max_line_length)

    if distraction_ratio is not None:  # for ablation
        long_lines = add_distraction_lines(long_lines, distraction_ratio)
        cleaned_text_lines = add_distraction_lines(
            cleaned_text_lines, distraction_ratio
        )

    smoothed_lines = smooth_lines(long_lines, sigma=smooth_sigma)
    smoothed_text_lines = smooth_lines(cleaned_text_lines, sigma=smooth_sigma)

    all_lines = smoothed_lines + smoothed_text_lines

    # scale lines
    scale_factor = 512 / model_resolution

    def scale_line_data(line_data):
        line_data["line"] = scale_line(line_data["line"], scale_factor)
        line_data["width"] = (
            line_data["width"] * scale_factor
            if line_data["width"] is not None
            else None
        )
        return line_data

    all_lines = [scale_line_data(line_data) for line_data in all_lines]

    # Comporess lines for export
    compact_lines = compress_lines(all_lines)

    # filter for lines with length > 0
    compact_lines = [line for line in compact_lines if line["line"].length > 0]

    export_lines(
        lines=compact_lines,
        output_file=Path(output_json_file),
        image_path=image_path,
        ckpt=ckpt,
        config=config,
        score_threshold=score_threshold,
        max_line_length=max_line_length,
        smooth_sigma=smooth_sigma,
        duplicate_line_thickness=duplicate_line_thickness,
        duplicate_containment_ratio=duplicate_containment_ratio,
        min_text_margin=min_text_margin,
        max_text_margin=max_text_margin,
        num_text_margin_steps=num_text_margin_steps,
    )

    if visualize:
        visualize_lines(Path(output_json_file))

    return compact_lines


def longest_path(G: nx.Graph, verbose: bool) -> Optional[LineString]:
    """
    Find the longest path in an undirected connected graph and converts it to a LineString.

    Parameters:
    G (nx.Graph): The input graph.
    verbose (bool): Whether to print warnings.

    Returns:
    Optional[LineString]: The longest path as a LineString object, or None if the graph has cycles.

    Raises:
    None

    """
    assert not G.is_directed()
    assert nx.is_connected(G)

    cycle = next(nx.simple_cycles(G), None)
    if cycle is not None:
        if verbose:
            print("Warning: Graph has cycles. Skipping!")
        return None

    if len(G.edges()) == 0:
        return []

    def farthest_node(n):
        return list(nx.bfs_edges(G, n))[-1][-1]

    random_node = list(G.nodes())[0]
    node_0 = farthest_node(random_node)
    node_1 = farthest_node(node_0)

    path = nx.shortest_path(G, node_0, node_1)  # is also the longest path

    return LineString(path)


def mask2line(
    mask: np.ndarray,
    verbose: bool,
    line_type: str,
    calc_margin: bool,
    resolution: int,
    min_text_margin: float,
    max_text_margin: float,
    num_text_margin_steps: int,
) -> Optional[Dict]:
    """
    Converts a binary mask image into a LineString object representing the longest line segment in the mask.

    Args:
        mask (np.ndarray): Binary mask image where white pixels represent the line and black pixels represent the background.
        verbose (bool): Whether to print warnings.
        line_type (str): Type of line (either line or text)
        calc_margin (bool): Whether or not to calcualte the margin of given line
        min_text_margin (float): The minimum margin for text lines in pixels.
        max_text_margin (float): The maximum margin for text lines in pixels.
        num_text_margin_steps (int): The number of steps to calculate the margin for text lines.

    Returns:
        Optional[LineString]: The longest line segment represented as a LineString object. Returns None if no line is found.

    """
    skeleton = skeletonize(mask)
    points = np.array(np.where(skeleton)).transpose(1, 0)

    # compare all points with their respective neighbors
    edges = []
    neighbor_offsets = [
        (-1, 1),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    for point in points:
        for offset in neighbor_offsets:
            neighbor = point + offset
            if np.any(neighbor < 0) or np.any(neighbor >= skeleton.shape):
                continue

            if skeleton[tuple(neighbor)] != 0:
                edges.append((tuple(point), tuple(neighbor)))

    G = from_edgelist(edges)

    lines = [
        longest_path(G.subgraph(subgraph), verbose=verbose)
        for subgraph in nx.connected_components(G)
    ]

    lines = [line for line in lines if line is not None]

    if len(lines) == 0:
        return None

    line = max(lines, key=lambda l: l.length)

    # correct the x and y coordinates
    line = LineString(np.roll(np.array(line.coords), shift=1, axis=-1))

    return {
        "id": f"lineformer-{uuid4()}",
        "type": line_type,
        "line": line,
        "width": (
            search_text_line_margin(
                mask=mask,
                line=line,
                resolution=resolution,
                min_margin=min_text_margin,
                max_margin=max_text_margin,
                num_steps=num_text_margin_steps,
            )
            * 2
            if calc_margin
            else None
        ),
    }


def extract_all_lines(
    inst_masks: List[np.ndarray],
    line_type: str,
    calc_margin: bool,
    resolution: int,
    min_text_margin: float,
    max_text_margin: float,
    num_text_margin_steps: int,
    verbose: bool,
) -> List[Dict]:
    """
    Extracts all lines from a list of instance masks.

    Args:
        inst_masks (List[np.ndarray]): A list of instance masks.
        line_type (str): Type of line (either line or text)
        calc_margin (bool): Whether or not to calcualte the margin of given line
        min_text_margin (float): The minimum margin for text lines in pixels.
        max_text_margin (float): The maximum margin for text lines in pixels.
        num_text_margin_steps (int): The number of steps to calculate the margin for text lines.
        verbose (bool): Whether to print warnings.

    Returns:
        List[LineString]: A list of LineString objects representing the extracted lines.
    """

    lines = [
        mask2line(
            mask,
            line_type=line_type,
            calc_margin=calc_margin,
            resolution=resolution,
            min_text_margin=min_text_margin,
            max_text_margin=max_text_margin,
            num_text_margin_steps=num_text_margin_steps,
            verbose=verbose,
        )
        for mask in inst_masks
    ]
    lines = [line for line in lines if line is not None]
    return lines


def remove_duplicate_lines(
    lines: List[Dict], line_thickness: float, containment_ratio: float
) -> List[Dict]:
    """
    Removes duplicate lines from a list of LineString based on their containment in other linestrings.

    Args:
        lines (List[Dict]): The list of LineString objects to remove duplicates from.
        line_thickness (float): The thickness of the lines used for containment calculation.
        containment_ratio (float): The minimum containment ratio required for a line to be considered a duplicate.

    Returns:
        List[Dict]: The list of LineString objects with duplicate lines removed.
    """
    if len(lines) == 0:
        return []

    def containment(line_0: LineString, line_1: LineString) -> float:
        line_0 = line_0.buffer(line_thickness)
        line_1 = line_1.buffer(line_thickness)

        if not line_0.intersects(line_1):
            return 0

        return line_0.intersection(line_1).area / line_0.area

    current = max(lines, key=lambda l: l["line"].length)
    keep = [current]
    remaining = [line for line in lines if line != keep]

    while len(remaining) > 0:
        remaining = [
            line
            for line in remaining
            if containment(line["line"], current["line"]) < containment_ratio
        ]

        if len(remaining) == 0:
            break

        current = max(remaining, key=lambda l: l["line"].length)
        keep.append(current)
        remaining = [line for line in remaining if line != current]

    return keep


def filter_lines_by_length(lines: List[Dict], min_length: float) -> List[Dict]:
    """
    Filters a list of LineString objects based on their length.

    Args:
        lines (List[Dict]): The list of LineString objects to filter.
        min_length (float): The minimum length required for a LineString to be included in pixles.

    Returns:
        List[Dict]: The filtered list of LineString objects.
    """
    return [line for line in lines if line["line"].length > min_length]


def smooth_lines(lines: List[Dict], sigma: float) -> List[LineString]:
    """
    Smooths the given list of LineString objects using Gaussian smoothing.

    Args:
        lines (List[Dict]): The list of LineString objects to be smoothed.
        sigma (float): The standard deviation of the Gaussian filter.

    Returns:
        List[Dict]: The list of smoothed LineString objects.

    """
    for line_data in lines:
        line = line_data["line"]
        x_smoothed = gaussian_filter1d(line.xy[0], sigma)
        y_smoothed = gaussian_filter1d(line.xy[1], sigma)
        smoothed_line = LineString(zip(x_smoothed, y_smoothed))
        line_data["line"] = smoothed_line

    return lines


def scale_line(line: LineString, factor: float) -> LineString:
    return scale(line, xfact=factor, yfact=factor, origin=(0, 0))


def search_text_line_margin(
    mask: np.ndarray,
    line: LineString,
    resolution: int,
    min_margin: float = 1,
    max_margin: float = 50,
    num_steps: int = 100,
) -> float:
    margins = np.linspace(min_margin, max_margin, num_steps)
    poly_masks = [poly2mask(line.buffer(i), resolution) for i in margins.tolist()]

    mask_expanded = repeat(mask, "h w -> n h w", n=margins.shape[0])
    intersections = np.count_nonzero(
        np.stack([mask_expanded, poly_masks]).all(axis=0), axis=(1, 2)
    )
    unions = np.count_nonzero(
        np.stack([mask_expanded, poly_masks]).any(axis=0), axis=(1, 2)
    )

    ious = intersections / unions

    max_idx = np.argmax(ious)

    return margins[max_idx]


def poly2mask(poly: Polygon, resolution: int) -> np.ndarray:
    poly_mask = np.zeros([resolution, resolution])

    # check the polygon validity
    # sometimes the polygon boundary is MultiLineString (if the polygon contains holes)
    if (
        not poly.geom_type == "Polygon"
        or not poly.is_valid
        or not poly.boundary.geom_type == "LineString"
    ):
        return poly_mask

    points = [[x, y] for x, y in zip(*poly.boundary.coords.xy)]

    return cv2.fillPoly(poly_mask, np.array([points]).astype(np.int32), color=1).astype(
        "bool"
    )


def compress_line(line_data: Dict) -> Dict:
    line = line_data["line"]

    # reorder line coordinate from left to right
    if line.coords.xy[0][0] > line.coords.xy[0][-1]:
        line = LineString(list(reversed(list(line.coords))))

    simple_line = line.simplify(0.01)
    compact_line = LineString(
        [round(p, 4) for p in coord] for coord in simple_line.coords
    )
    line_data["line"] = compact_line
    return line_data


def compress_lines(lines: List[Dict]) -> List[Dict]:
    return [compress_line(line) for line in lines]


def export_lines(
    lines: List[Dict],
    output_file: Path,
    image_path: Path,
    ckpt: Optional[str] = None,
    config: Optional[str] = None,
    score_threshold: Optional[float] = None,
    max_line_length: Optional[float] = None,
    smooth_sigma: Optional[float] = None,
    duplicate_line_thickness: Optional[float] = None,
    duplicate_containment_ratio: Optional[float] = None,
    min_text_margin: Optional[float] = None,
    max_text_margin: Optional[float] = None,
    num_text_margin_steps: Optional[int] = None,
):
    output_data = {
        "image_path": image_path.as_posix(),
        "ckpt": ckpt,
        "config": config,
        "lines": [
            {k: v.wkt if k == "line" else v for k, v in line_data.items()}
            for line_data in lines
        ],
        "score_threshold": score_threshold,
        "max_line_length": max_line_length,
        "smooth_sigma": smooth_sigma,
        "duplicate_line_thickness": duplicate_line_thickness,
        "duplicate_containment_ratio": duplicate_containment_ratio,
        "min_text_margin": min_text_margin,
        "max_text_margin": max_text_margin,
        "num_text_margin_steps": num_text_margin_steps,
    }

    output_file.write_text(json.dumps(output_data))


def add_distraction_lines(lines: List[Dict], ratio: float) -> List[Dict]:
    """
    Add distractor lines to the given list of lines.

    Args:
        lines (List[Dict]): The list of lines to add distractor lines to.
        ratio (float): The ratio of distractor lines to add to the list.

    Returns:
        List[Dict]: The list of lines with distractor lines added.
    """
    if len(lines) < 2:
        return lines 

    assert 0 <= ratio <= 1

    num_distractions = int(len(lines) * ratio)

    distractor_lines = []
    while len(distractor_lines) < num_distractions:
        [base1, base2] = random.sample(lines, 2)

        assert base1["type"] == base2["type"]

        alpha = random.uniform(0.25, 0.75)
        merged_line = interpolate_lines(base1["line"], base2["line"], alpha)

        widths = [base1["width"], base2["width"]]
        widths = [w for w in widths if w is not None]
        merged_width = float(np.mean(widths)) if len(widths) > 0 else None

        distractor_lines.append(
            {
                "id": f"lineformer-{uuid4()}",
                "type": base1["type"],
                "line": merged_line,
                "width": merged_width,
            }
        )

    return lines + distractor_lines


def interpolate_lines(line1: LineString, line2: LineString, alpha: float) -> LineString:
    # interpolate 101 points along line1
    points1 = np.array(
        [line1.interpolate(i / 100, normalized=True).xy for i in range(101)]
    )
    points2 = np.array(
        [line2.interpolate(i / 100, normalized=True).xy for i in range(101)]
    )

    merged_points = points1 * alpha + points2 * (1 - alpha)

    return LineString(merged_points.squeeze())


# python3 -m src.inv3d_line_detector.find_lines
if __name__ == "__main__":
    raise ValueError("revise paths")
    split = "real"
    output_dir = Path(
        f"data/detector/lines/inv3d_samv3_text_v2_512px_3px_lineformer/{split}"
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    ckpt = "models/lineformer_inv3d_text_512px_3px_swin_t/best_segm_mAP_iter_85250.pth"
    config = "models/lineformer_inv3d_text_512px_3px_swin_t/lineformer_inv3d_text_512px_3px_swin_t_config.py"
    data_path = Path(f"data/segmenter/inv3d_masks_v3/real")

    image_paths = list(data_path.glob("*[!_mask].jpg"))
    image_paths = list(sorted(image_paths))

    find_and_export_lines_parallel(
        num_workers=1,
        image_paths=image_paths,
        ckpt=ckpt,
        config=config,
        output_dir=output_dir,
        visualize=True,
        device="cuda:7",
    )
