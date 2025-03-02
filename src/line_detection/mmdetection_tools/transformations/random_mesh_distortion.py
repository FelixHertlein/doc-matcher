import os

os.sys.path.append("/workspaces/doc-matcher/src")

import numpy as np
from pathlib import Path
import random
from inv3d_util.load import load_npz, save_npz
from einops import rearrange
from inv3d_util.mapping import apply_map
from shapely.geometry import Point
from mmdet.datasets.builder import PIPELINES

from .forward_map import ForwardMap


@PIPELINES.register_module()
class RandomMeshDistortion:
    def __init__(
        self,
        cache_dir: str,
        prob: float = 0.5,
        min_points: int = 1,
        max_points: int = 50,
        gauss_scale: float = 0.025,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.prob = prob
        self.min_points = min_points
        self.max_points = max_points
        self.gauss_scale = gauss_scale

    def __call__(self, results):
        if random.random() < self.prob:
            sample_name = results["img_info"]["filename"].split(".")[0]
            bm = self.calculate_bm(sample_name)

            h, w, c = results["img"].shape
            resolution = (h, w)

            # map image
            results["img"] = apply_map(results["img"], bm, resolution=resolution)

            # map masks
            masks = results["gt_masks"].masks
            masks = rearrange(masks, "c h w -> h w c").astype(np.float32)
            masks = apply_map(masks, bm, resolution=resolution)
            masks = rearrange(masks, "h w c -> c h w")
            masks = np.round(masks).astype(np.uint8)
            results["gt_masks"].masks = masks

            # recalculate bboxes
            results["gt_bboxes"] = get_bounding_boxes(masks)

            if "ann_info" in results:
                results.pop(
                    "ann_info"
                )  # just to make sure that it is not needed anymore

        return results

    def calculate_bm(self, sample_name: str):
        cache_file = self.cache_dir / f"{sample_name}.npz"

        if cache_file.is_file():
            return load_npz(cache_file, allow_pickle=True)
        else:
            m = ForwardMap(warped_image_file=None)

            for idx in range(random.randint(self.min_points, self.max_points)):
                for channel in [m.v_map, m.h_map]:
                    point = Point(random.random() * 512, random.random() * 512)
                    shift = float(
                        np.random.normal(loc=0, scale=self.gauss_scale, size=(1))
                    )
                    old_value = channel.interp[int(point.y), int(point.x)]
                    new_value = old_value + shift
                    channel.add_point(point, new_value, "distortion")

            bm = m.create_good_bm_map()
            save_npz(cache_file, bm, override=True)
            return bm


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
