import torch
import torch.nn.functional as F
from typing import Tuple, Union, Optional
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from scipy.spatial import ConvexHull

from inv3d_util.mapping import scale_map
from einops import rearrange, repeat

from inv3d_util.misc import median_blur


def resample_line(line: LineString, ref_line: LineString) -> LineString:
    progreses = [
        ref_line.project(Point(coord), normalized=True) for coord in ref_line.coords
    ]

    return LineString(
        [line.interpolate(progress, normalized=True) for progress in progreses]
    )


def create_convex_hull(points):
    # Check if there are enough points to create a polygon
    if len(points) == 0:
        return None

    if len(points) == 1:
        [point] = points
        return Polygon([point, point, point, point])

    if len(points) == 2:
        [first, second] = points
        return Polygon([first, second, second, first])

    # Calculate the convex hull of the points
    hull = ConvexHull(points)

    # Extract the vertices of the convex hull
    hull_vertices = [points[i] for i in hull.vertices]

    # Create a Shapely polygon from the convex hull vertices
    shapley_polygon = Polygon(hull_vertices)

    return shapley_polygon


def is_horizontal_line(line: LineString):
    minx, miny, maxx, maxy = line.bounds
    width = maxx - minx
    height = maxy - miny
    return width > height


def is_border_line(line: LineString):
    square_size = 512
    proximity = 0.05 * square_size

    def is_border_point(x, y):
        return (
            x < proximity
            or y < proximity
            or x > square_size - proximity
            or y > square_size - proximity
        )

    return all(is_border_point(x, y) for x, y in line.coords)


def apply_map_v2(
    image: np.ndarray,
    bm: np.ndarray,
    resolution: Union[None, int, Tuple[int, int]] = None,
    padding_value: Optional[float] = None,
    padding_blur: bool = False,
):
    if resolution is not None:
        bm = scale_map(bm, resolution)

    input_dtype = image.dtype
    img = rearrange(image, "h w c -> 1 c h w")
    img = torch.from_numpy(img).double()

    bm = torch.from_numpy(bm).unsqueeze(0).double()
    bm = (bm * 2) - 1
    bm = torch.roll(bm, shifts=1, dims=-1)

    if padding_value is not None or padding_blur:
        padding_mode = "border"
    else:
        padding_mode = "zeros"

    res = F.grid_sample(
        input=img, grid=bm, align_corners=True, padding_mode=padding_mode
    )

    if padding_value is not None:
        mask = torch.any(bm.abs() > 1, dim=-1, keepdim=False)
        mask = repeat(mask, "n h w -> n c h w", c=3)
        res[mask] = padding_value

    if padding_blur:
        blurred = median_blur(res.float()).double()
        mask = torch.any(bm.abs() > 1, dim=-1, keepdim=False)
        mask = repeat(mask, "n h w -> n c h w", c=3)
        res = torch.where(mask, blurred, res)

    res = rearrange(res[0], "c h w -> h w c")
    res = res.numpy().astype(input_dtype)
    return res
