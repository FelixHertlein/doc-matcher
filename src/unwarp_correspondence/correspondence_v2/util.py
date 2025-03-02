from shapely.geometry import LineString, Point, Polygon
from scipy.spatial import ConvexHull


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
    proximity = 0.025 * square_size

    def is_border_point(x, y):
        return (
            x < proximity
            or y < proximity
            or x > square_size - proximity
            or y > square_size - proximity
        )

    return all(is_border_point(x, y) for x, y in line.coords)
