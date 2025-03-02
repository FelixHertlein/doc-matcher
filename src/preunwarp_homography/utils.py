import numpy as np
from shapely.geometry import LineString
from sklearn.linear_model import LinearRegression
from inv3d_util.load import save_image
from inv3d_util.image import scale_image
from skimage.transform import ProjectiveTransform, warp
import matplotlib.pyplot as plt


def approximate_line(line: LineString) -> LineString:
    # Extract coordinates from the LineString
    coords = np.array(line.xy).T

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the coordinates
    model.fit(coords[:, 0].reshape(-1, 1), coords[:, 1])

    # Get the slope and intercept of the regression line
    slope = model.coef_[0]
    intercept = model.intercept_

    # Create a new LineString representing the best-fit line
    min_x, max_x = np.min(coords[:, 0]), np.max(coords[:, 0])
    line_coords = np.array(
        [[min_x, min_x * slope + intercept], [max_x, max_x * slope + intercept]]
    )
    best_fit_line = LineString(line_coords)

    return best_fit_line


def project_image_hq(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    heigth, width, _ = image.shape
    max_side = max(heigth, width)
    image_orig_square = scale_image(image, resolution=max_side)
    H_hq = H.copy()

    # see https://stackoverflow.com/questions/21019338/how-to-change-the-homography-with-the-scale-of-the-image/48915151#48915151
    s = 512 / max_side
    H_hq[2, 0] = H_hq[2, 0] * s
    H_hq[2, 1] = H_hq[2, 1] * s
    H_hq[0, 2] = H_hq[0, 2] / s
    H_hq[1, 2] = H_hq[1, 2] / s

    transform_hq = ProjectiveTransform(H_hq)
    proj_image_hq = warp(
        image_orig_square, transform_hq.inverse, output_shape=(max_side, max_side)
    )
    proj_image_hq = (proj_image_hq * 255).astype(np.uint8)
    return proj_image_hq
