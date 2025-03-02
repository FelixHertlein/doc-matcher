from typing import Tuple
from functools import lru_cache
import cv2
from shapely.affinity import scale
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.ops import nearest_points
from shapely.affinity import scale
import numpy as np
from scipy.interpolate import griddata
from shapely.geometry import Polygon
from einops import rearrange
from pathlib import Path
import os
from doctr.models import ocr_predictor

from pathlib import Path
from shapely import wkt
import json
from inv3d_util.mapping import apply_map, create_identity_map
from inv3d_util.load import load_image
from inv3d_util.parallel import process_tasks
from flatten_dict import flatten

from .io import load_frenet_features


@lru_cache(maxsize=1)
def load_doctr_model_cached():
    return ocr_predictor(pretrained=True)


def extract_all_frenet_features(
    lines_dir: Union[str, Path],
    output_dir: Union[str, Path],
    image_hq_dir: Path,
    feature_params: Dict,
    num_workers: int,
):
    lines_files = list(lines_dir.glob("*.json"))
    lines_files = sorted(lines_files)

    tasks = [
        {
            "lines_file": lines_file.as_posix(),
            "output_file": output_dir / f"{lines_file.stem}.npz",
            "feature_params": feature_params,
            "image_hq_file": image_hq_dir / f"{lines_file.stem}.jpg",
        }
        for lines_file in lines_files
    ]

    # filter tasks to avoid duplicate feature extraction
    tasks = [
        task
        for task in tasks
        if not features_complete(task["output_file"], task["lines_file"])
    ]

    load_doctr_model_cached()  # load model in main process

    process_tasks(
        _extract_task,
        tasks,
        num_workers=num_workers,
        use_indexes=True,
        desc="Extracting frenet features",
    )

    load_doctr_model_cached.cache_clear()  # clear cache to free memory


def _extract_task(task: Dict):
    extract_frenet_features(**task)


def features_complete(features_file: str, lines_file: str) -> bool:
    features_file = Path(features_file)
    lines_file = Path(lines_file)

    if not features_file.exists():
        return False

    lines_data = json.loads(lines_file.read_text())["lines"]

    features = set(load_frenet_features(features_file).keys())
    line_ids = set([line["id"] for line in lines_data])

    return features == line_ids


def extract_frenet_features(
    lines_file: Path,
    output_file: Path,
    feature_params: Path,
    image_hq_file: Dict,
) -> Dict[str, np.ndarray]:

    lines_file = Path(lines_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    lines_data = json.loads(lines_file.read_text())
    image_high_quality = load_image(image_hq_file)

    features = {
        line_data["id"]: extract_line_frenet_features(
            image=image_high_quality,
            line_data=line_data,
            feature_params=feature_params,
        )
        for line_data in lines_data["lines"]
    }

    features = {k: v for k, v in features.items() if v is not None}

    features = flatten(features, reducer="dot")

    np.savez_compressed(output_file, **features)


def extract_line_frenet_features(
    image: np.ndarray, line_data: Dict, feature_params: Dict
):
    line = line_data["line"]

    if isinstance(line, str):
        line = wkt.loads(line)

    if line.length == 0:
        return None

    line_margin = feature_params["line_context"] / 2
    if line_data["width"] is None:
        margin = line_margin
    else:
        margin = int(line_data["width"] / 2) + line_margin

    # ensure that the line orientation is correct (relevant for text lines)
    if line.coords.xy[0][0] > line.coords.xy[0][-1]:
        line = LineString(list(reversed(list(line.coords))))

    # calculate mapping from cartesian to frenet space
    frenet_mapping = create_frenet_mapping(line, margin, 128)

    # scale high res image to be square
    height, width, channels = image.shape
    new_size = max(height, width)

    image = cv2.resize(image, (new_size, new_size))
    scale_factor = new_size / 512

    # scale line and margin to high-res format
    line = scale(line, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
    margin = int((margin / 512) * new_size)

    # actually unwarp the cutouts
    feature_shape = (feature_params["feature_width"], feature_params["feature_length"])
    visual_features = apply_map(
        image.astype("float64"), frenet_mapping, resolution=feature_shape
    ).astype("uint8")

    # create positional features
    cartesian_coordinates = create_identity_map(feature_shape)
    cartesian_coordinates = (
        apply_map(cartesian_coordinates, frenet_mapping, resolution=feature_shape) * 512
    )
    positional_features = np.apply_along_axis(
        func1d=lambda x: embedd2d(
            x=x[0], y=x[1], depth=feature_params["positional_encoding_dim"]
        ),
        axis=2,
        arr=cartesian_coordinates,
    )

    if "text" not in line_data:
        line_data["text"] = None

    if "words" not in line_data:
        line_data["words"] = None

    if line_data["type"] == "text_line" and line_data["text"] is None:
        text, words = detect_text(image, line, line_data["width"] * scale_factor)
    else:
        text = line_data["text"]
        words = line_data["words"]

    return {
        "visual": visual_features,
        "text": text,
        "words": words,
        "positional": positional_features,
    }


def detect_text(
    image: np.ndarray, line: LineString, line_width: float
) -> Tuple[str, List[Dict]]:
    h, w, c = image.shape

    coordinates = create_identity_map((h, w)) * 512
    image_extended = np.concatenate([image.astype(np.float64), coordinates], axis=-1)

    extension = (50 / 512) * h

    margin_small = line_width / 2
    margin_large = line_width * 2

    # extrude line to gather more context around real line
    line_large = extrude_line(line, extension)

    # create mapping between frenet and cartesian space
    frenet_mapping_large = create_frenet_mapping(
        line_large, margin_large, 128, image_size=h
    )

    # create line cropping in frenet space
    visual_featurs_extended = apply_map(
        image_extended,
        frenet_mapping_large,
        resolution=(int(margin_large * 2), int(line_large.length)),
    )

    # split mapped image and coordinates
    visual_features = visual_featurs_extended[:, :, :c].astype(np.uint8)
    coordinates = visual_featurs_extended[:, :, c:]

    # create a binary mask in frenet space representing the text content
    mask = poly2mask(line.buffer(margin_small), shape=image.shape[:2])
    mask = np.expand_dims(mask, axis=-1).astype(np.float32)
    mask = apply_map(
        mask,
        frenet_mapping_large,
        resolution=(margin_large * 2, int(line_large.length)),
    )
    mask = mask > 0.5
    mask = (mask).astype(np.uint8)
    mask = mask.squeeze()

    # detect text in image and collect text within the mask
    text, words = detect_text_in_frenet(visual_features, mask)
    ch, cw, cc = coordinates.shape

    def project_word(word: Polygon) -> Polygon:
        word["poly"] = Polygon(
            [
                Point(reversed(coordinates[min(int(y), ch - 1), min(int(x), cw - 1)]))
                for x, y in zip(*word["poly"].exterior.coords.xy)
            ]
        )

        return word

    words = [project_word(word) for word in words]

    return text, words


def poly2mask(poly: Polygon, shape=(512, 512)) -> np.ndarray:
    poly_mask = np.zeros(shape)

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


def detect_text_in_frenet(
    image: np.ndarray, mask: np.ndarray
) -> Tuple[str, List[dict]]:
    model = load_doctr_model_cached()
    result = model([image])

    words = []

    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                (x0, y0), (x1, y1) = word.geometry
                x0 = int(x0 * image.shape[1])
                y0 = int(y0 * image.shape[0])
                x1 = int(x1 * image.shape[1])
                y1 = int(y1 * image.shape[0])

                word_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                word_mask[y0:y1, x0:x1] = 1

                contain_mask = np.all(np.stack([mask, word_mask], axis=-1), axis=-1)

                containment = np.count_nonzero(contain_mask) / np.count_nonzero(
                    word_mask
                )

                if containment > 0.5:
                    x_mean = (x0 + x1) / 2
                    poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                    words.append((x_mean, word.value, poly))

    words = sorted(words, key=lambda x: x[0])
    text = " ".join([word[1] for word in words])

    words = [
        {
            "text": word_text,
            "poly": word_poly,
        }
        for _, word_text, word_poly in words
    ]
    return text, words


def extrude_line(line: LineString, extrude_length: float) -> LineString:
    def extrude_endpoint(
        inner: Tuple[float, float], outer: Tuple[float, float]
    ) -> Tuple[float, float]:
        segment = LineString([inner, outer])
        fact = (segment.length + extrude_length) / segment.length
        segment = scale(segment, xfact=fact, yfact=fact, origin=Point(inner))

        return segment.coords[-1]

    new_start = extrude_endpoint(line.coords[1], line.coords[0])
    new_end = extrude_endpoint(line.coords[-2], line.coords[-1])

    return LineString([new_start] + line.coords[1:-1] + [new_end])


def create_frenet_mapping(
    line: LineString, margin: float, resolution: int, image_size: int = 512
):

    norm_factor = 1 / image_size
    line = scale(line, xfact=norm_factor, yfact=norm_factor, origin=(0, 0))
    line = line.simplify(0.0001)  # to avoid segments of length 0
    margin = margin * norm_factor

    polygon = line.buffer(margin)

    minx, miny, maxx, maxy = polygon.bounds
    minx = minx - 2 * norm_factor
    miny = miny - 2 * norm_factor
    maxx = maxx + 2 * norm_factor
    maxy = maxy + 2 * norm_factor

    grid_step = max((maxx - minx) / resolution, (maxy - miny) / resolution)
    extrude_length = grid_step * 1.1

    # extrude line to avoid NaNs in the edge cases
    line_extruded = extrude_line(line, extrude_length)

    def get_side_polygon(line, distance, side):
        off = line.parallel_offset(distance, side)
        p = Polygon(list(line.coords) + list(reversed(list(off.coords))))
        return p

    left_polygon = get_side_polygon(line_extruded, margin, "left")
    right_polygon = get_side_polygon(line_extruded, margin, "right")

    frenet_points = []
    catesian_points = []

    # collect corresponding points for unstructured interpolation
    for i in np.linspace(minx, maxx, resolution):
        for j in np.linspace(miny, maxy, resolution):
            p = Point(i, j)

            sign = 1 if p.distance(left_polygon) < p.distance(right_polygon) else -1

            _, p_line = nearest_points(p, line_extruded)

            projection = line_extruded.project(p_line) - extrude_length
            dist = p.distance(p_line) * sign

            if abs(dist) > margin + 2 * grid_step:
                continue

            frenet_points.append((projection, dist))
            catesian_points.append((i, j))

    frenet_points = np.array(frenet_points)
    catesian_points = np.array(catesian_points)

    xi = np.mgrid[0 : line.length : resolution * 1j, -margin : margin : resolution * 1j]
    xi = rearrange(xi, "ch x y -> (x y) ch")

    mapping = griddata(points=frenet_points, values=catesian_points, xi=xi)
    mapping = rearrange(mapping, "(x y) ch -> y x ch", x=resolution, y=resolution)
    mapping = np.roll(mapping, shift=1, axis=-1)

    return mapping


def embedd2d(x: float, y: float, depth: int = 8) -> np.ndarray:
    assert depth % 4 == 0, "depth must be divisible by 4"

    embedding = []

    for pos in [x, y]:
        for i in range(depth // 4):
            embedding.append(np.sin(pos / 10000 ** (4 * i / depth)))
            embedding.append(np.cos(pos / 10000 ** (4 * i / depth)))

    return np.array(embedding).astype(np.float32)


def calculate_mean_and_std(
    input_dirs: List[Union[str, Path]], output_file: Union[str, Path]
) -> dict:
    print("calculating mean and std...")

    input_files = [
        file
        for input_dir in input_dirs
        for file in Path(input_dir).glob("*_features.npz")
    ]

    data = [
        features["visual"]
        for input_file in input_files
        for features in load_frenet_features(input_file).values()
    ]

    data = np.stack(data)

    mean = np.mean(data, axis=(0, 1, 2))
    std = np.std(data, axis=(0, 1, 2))
    count = data.shape[0]

    print("mean: ", mean)
    print("std: ", std)
    print("count", count)

    res = {"mean": mean.tolist(), "std": std.tolist(), "count": count}

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(res, indent=4))

    return res
