from typing import Dict, List
from shapely.geometry import LineString
from shapely.affinity import scale
import numpy as np

from inv3d_util.mapping import tight_crop_map, transform_coords


def scale_line(line: LineString, factor: float) -> LineString:
    return scale(line, xfact=factor, yfact=factor, origin=(0, 0))


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
