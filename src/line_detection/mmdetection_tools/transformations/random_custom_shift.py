import os

os.sys.path.append("/workspaces/doc-matcher/src")

import numpy as np
import scipy
import random
from einops import rearrange
from inv3d_util.mapping import apply_map
from shapely.geometry import Point
from mmdet.datasets.builder import PIPELINES

from .forward_map import ForwardMap


@PIPELINES.register_module()
class RandomCustomShift:
    def __init__(
        self,
        prob: float = 0.5,
        max_shift_px: int = 50,
    ):
        self.prob = prob
        self.max_shift_px = max_shift_px

    def __call__(self, results):
        if random.random() < self.prob:
            shift_x = random.randint(0, self.max_shift_px)
            shift_y = random.randint(0, self.max_shift_px)

            # map image
            results["img"] = scipy.ndimage.shift(
                results["img"],
                shift=(shift_x, shift_y, 0),
                mode="constant",
                cval=0,
            )

            # map masks
            masks = results["gt_masks"].masks
            masks = np.array(
                [
                    scipy.ndimage.shift(
                        m,
                        shift=(shift_x, shift_y),
                        mode="constant",
                        cval=0,
                    )
                    for m in masks
                ]
            )
            results["gt_masks"].masks = masks

            # recalculate bboxes
            results["gt_bboxes"] = get_bounding_boxes(results["gt_masks"].masks)

            if "ann_info" in results:
                results.pop(
                    "ann_info"
                )  # just to make sure that it is not needed anymore

        return results


def get_bounding_boxes(mask_array):
    n, h, w = mask_array.shape
    bounding_boxes = []

    for i in range(n):
        mask = mask_array[i, :, :]
        non_zero_rows = np.any(mask, axis=1)
        non_zero_cols = np.any(mask, axis=0)

        if np.any(non_zero_rows) and np.any(non_zero_cols):
            y_min, y_max = np.where(non_zero_rows)[0][[0, -1]]
            x_min, x_max = np.where(non_zero_cols)[0][[0, -1]]

            bounding_boxes.append([x_min, y_min, x_max, y_max])
        else:
            # No shape in the mask
            bounding_boxes.append([0, 0, 0, 0])

    return np.array(bounding_boxes)
