from typing import Dict
import numpy as np
from shapely.geometry import Polygon, LineString
from skimage.transform import ProjectiveTransform


def project_mask(transform, polygon):
    return Polygon(transform(polygon.exterior.coords))


def project_lines(transform, lines: np.ndarray):
    return transform(lines.reshape(-1, 2)).reshape(-1, 2, 2)


def project_shapely_line(transform, line):
    return LineString(transform(line.coords))


def calc_directions(lines: np.ndarray):
    lengths = calc_weights(lines)
    directions = lines[..., 1, :] - lines[..., 0, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(directions, lengths[..., None])


def calc_weights(lines: np.ndarray):
    return np.linalg.norm(lines[..., 0, :] - lines[..., 1, :], axis=-1)


def horizontal_angles(directions: np.ndarray):
    # dirctions: (N, 2)

    # ref: (2, 2)
    ref = np.array([[-1, 0], [1, 0]])

    # cos_angles: (N, 2, 1)
    cos_angles = np.dot(directions, ref[..., None])

    # angles: (N, 2, 1)
    angles = np.arccos(np.clip(cos_angles, -1, 1))

    # return (N,)
    return np.min(angles, axis=1).squeeze()


def aligned_angles(directions: np.ndarray):
    # dirctions: (N, 2)

    # ref: (4, 2)
    ref = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

    # cos_angles: (N, 4, 1)
    cos_angles = np.dot(directions, ref[..., None])

    # angles: (N, 4, 1)
    angles = np.arccos(np.clip(cos_angles, -1, 1))

    # return (N,)
    return np.min(angles, axis=1).squeeze()


def xtoH(
    x: np.ndarray,
    rotation_only: bool = False,
) -> np.ndarray:
    # start with the identity matrix
    H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

    # build the rotation matrix
    rot = np.array(
        [[np.cos(x[0]), -np.sin(x[0])], [np.sin(x[0]), np.cos(x[0])]], dtype=np.float64
    )

    # build the shear matrix
    shear = np.array([[1, x[1]], [0, 1]], dtype=np.float64)

    # apply the transformations
    H[:2, :2] = np.matmul(H[:2, :2], rot)

    if not rotation_only:
        H[:2, :2] = np.matmul(H[:2, :2], shear)

        # set the vanishing vector
        H[2, 0] = x[2]
        H[2, 1] = x[3]

    return H


def calculate_norm_factors(x: np.ndarray, mask: Polygon) -> Dict[str, float]:

    transform = ProjectiveTransform(xtoH(x))

    mask_proj = project_mask(transform, mask)

    minx, miny, maxx, maxy = mask_proj.bounds

    width = maxx - minx
    height = maxy - miny

    scalex = 512 / width
    scaley = 512 / height

    return {
        "xoff": -minx,
        "yoff": -miny,
        "xfact": scalex,
        "yfact": scaley,
    }


def optim(
    x: np.ndarray,
    text_lines: np.ndarray,
    structural_lines: np.ndarray,
    mask_lines: np.ndarray,
    mask: Polygon,
    rotation_only: bool = False,
):

    transform = ProjectiveTransform(xtoH(x, rotation_only))

    proj_text_lines = project_lines(transform, text_lines)
    proj_stru_lines = project_lines(transform, structural_lines)
    proj_mask_lines = project_lines(transform, mask_lines)

    # noramaliyze coordinate
    norm_factors = calculate_norm_factors(x, mask)

    proj_text_lines[..., 0] += norm_factors["xoff"]
    proj_text_lines[..., 1] += norm_factors["yoff"]

    proj_stru_lines[..., 0] += norm_factors["xoff"]
    proj_stru_lines[..., 1] += norm_factors["yoff"]

    proj_mask_lines[..., 0] += norm_factors["xoff"]
    proj_mask_lines[..., 1] += norm_factors["yoff"]

    proj_text_lines[..., 0] *= norm_factors["xfact"]
    proj_text_lines[..., 1] *= norm_factors["yfact"]

    proj_stru_lines[..., 0] *= norm_factors["xfact"]
    proj_stru_lines[..., 1] *= norm_factors["yfact"]

    proj_mask_lines[..., 0] *= norm_factors["xfact"]
    proj_mask_lines[..., 1] *= norm_factors["yfact"]

    text_lines_directions = calc_directions(proj_text_lines)
    text_angles = horizontal_angles(text_lines_directions)
    text_weights = calc_weights(proj_text_lines)
    text_weighted_angles = text_angles * text_weights

    stru_lines_directions = calc_directions(proj_stru_lines)
    stru_angles = aligned_angles(stru_lines_directions)
    stru_weights_np = calc_weights(proj_stru_lines)
    stru_weighted_angles = stru_angles * stru_weights_np

    mask_lines_directions = calc_directions(proj_mask_lines)
    mask_angles = aligned_angles(mask_lines_directions)
    mask_weights_np = calc_weights(proj_mask_lines)
    mask_weighted_angles = mask_angles * mask_weights_np

    return np.concatenate(
        [
            text_weighted_angles[~np.isnan(text_weighted_angles)] ** 2,
            stru_weighted_angles[~np.isnan(stru_weighted_angles)] ** 2,
            mask_weighted_angles[~np.isnan(mask_weighted_angles)] ** 2,
        ]
    ).sum()
