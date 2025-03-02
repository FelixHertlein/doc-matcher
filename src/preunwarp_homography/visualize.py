import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import ProjectiveTransform
from shapely.geometry import Polygon


from .optimize import project_mask, project_lines


def visualize_optimization(
    H: np.ndarray,
    text_lines: np.ndarray,
    structural_lines: np.ndarray,
    mask_lines: np.ndarray,
    mask: Polygon,
):
    transform = ProjectiveTransform(H)

    text_lines = project_lines(transform, text_lines)
    structural_lines = project_lines(transform, structural_lines)
    mask_lines = project_lines(transform, mask_lines)
    proj_mask = project_mask(transform, mask)

    plt.clf()
    for line in text_lines:
        plt.plot(*line.T, color="blue")

    for line in structural_lines:
        plt.plot(*line.T, color="red")

    for line in mask_lines:
        plt.plot(*line.T, color="purple")

    plt.plot(*proj_mask.exterior.xy, color="green")

    plt.gca().invert_yaxis()
