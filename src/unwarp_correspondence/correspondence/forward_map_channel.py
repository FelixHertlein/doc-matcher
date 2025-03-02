import numpy as np
from copy import deepcopy
from typing import List, Tuple, Optional
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from shapely.geometry import LineString, Point, Polygon

from .util import resample_line, create_convex_hull


class ForwardMapChannel:
    def __init__(
        self,
        directional_borders: Tuple[LineString, LineString],
        orthogonal_borders: Tuple[LineString, LineString],
        min_slope: float,
        max_slope: float,
        is_vertical_gradient: bool,
    ):
        self.grid = list(reversed(np.mgrid[0:512, 0:512]))

        self.directional_borders = directional_borders
        self.orthogonal_borders = orthogonal_borders

        start_line, end_line = directional_borders

        self.values = [0] * len(start_line.coords) + [1] * len(end_line.coords)
        self.points = list(start_line.coords) + list(end_line.coords)
        self.types = ["start"] * len(self.points)

        self.tri = Delaunay(self.points)
        self.interp = LinearNDInterpolator(self.tri, self.values)(*self.grid)

        self.min_slope = min_slope
        self.max_slope = max_slope
        self.is_vertical_gradient = is_vertical_gradient

    def convex_hull(self, points_type: Optional[str]) -> Optional[Polygon]:
        points = self.points_by_type(points_type)
        hull = create_convex_hull(points)
        return hull

    def points_by_type(self, point_type: Optional[str]) -> np.ndarray:
        if point_type is None:
            return np.array(self.points)

        return np.array(
            [
                point
                for point, type in zip(self.points, self.types)
                if type == point_type
            ]
        )

    def add_line(
        self, warped_line: LineString, template_line: LineString, type: str
    ) -> bool:
        old_self = deepcopy(self)

        points_extra = list(warped_line.coords)

        idx = 1 if self.is_vertical_gradient else 0

        values_extra = (
            np.array(resample_line(template_line, warped_line).xy[idx]) / 512
        ).tolist()

        self.values.extend(values_extra)
        self.points.extend(points_extra)
        self.types.extend([type] * len(points_extra))

        self.tri = Delaunay(self.points)
        self.interp = LinearNDInterpolator(self.tri, self.values)(*self.grid)

        if self.is_vertical_gradient:
            diffs = np.abs(self.interp[..., 1:] - self.interp[..., :-1])
        else:
            diffs = np.abs(self.interp[1:] - self.interp[:-1])

        min_slope = diffs.min()
        max_slope = diffs.max()

        if min_slope < self.min_slope or max_slope > self.max_slope:
            # print("Invalid line found. Reverting changes.")
            self.restore(old_self)
            return False

        return True

    def add_points(self, points: List[Point], values: List[float], type: str):
        for point, value in zip(points, values):
            self.add_point(point, value, type)

    def add_point(self, point: Point, value: float, type: str):
        old_self = deepcopy(self)

        point = np.array(point.xy).squeeze().tolist()

        self.values.append(value)
        self.points.append(point)
        self.types.append(type)
        self.tri = Delaunay(self.points)
        self.interp = LinearNDInterpolator(self.tri, self.values)(*self.grid)

        if self.is_vertical_gradient:
            diffs = np.abs(self.interp[..., 1:] - self.interp[..., :-1])
        else:
            diffs = np.abs(self.interp[1:] - self.interp[:-1])

        min_slope = diffs.min()
        max_slope = diffs.max()

        if min_slope < self.min_slope or max_slope > self.max_slope:
            # print("Invalid point found. Reverting changes.")
            self.restore(old_self)
            return False

        return True

    def restore(self, old_self: "ForwardMapChannel"):
        self.values = old_self.values
        self.points = old_self.points
        self.types = old_self.types
        self.tri = old_self.tri
        self.interp = old_self.interp

    def project_points(self):
        all_values = self.values
        all_points = [Point(coord) for coord in self.points]

        hull = self.convex_hull("matched")
        if hull is None:
            return

        hull_points = [Point(coord) for coord in hull.exterior.coords]

        left_line, right_line = self.orthogonal_borders

        new_points = []
        new_values = []

        def process_hull_point(point):
            # find the closest point in all_points by index
            closest_point_idx, closed_point = min(
                enumerate(all_points), key=lambda p: p[1].distance(point)
            )
            value = all_values[closest_point_idx]

            if self.is_vertical_gradient:
                ref_line = LineString([(-1, point.y), (513, point.y)])
            else:
                ref_line = LineString([(point.x, -1), (point.x, 513)])

            parts = list(ref_line.difference(hull).geoms)
            assert len(parts) == 2

            selected_parts = [p for p in parts if p.distance(point) < 1e-6]
            assert len(parts) in [1, 2]

            for p in selected_parts:
                if p.intersects(left_line):
                    new_points.append(p.intersection(left_line))
                    new_values.append(value)

                if p.intersects(right_line):
                    new_points.append(p.intersection(right_line))
                    new_values.append(value)

        for point in hull_points:
            process_hull_point(point)

        self.add_points(new_points, new_values, "projected")
