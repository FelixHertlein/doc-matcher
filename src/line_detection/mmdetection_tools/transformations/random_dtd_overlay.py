import os

os.sys.path.append("/workspaces/doc-matcher/src")

import random
import numpy as np
from pathlib import Path
from inv3d_util.load import load_image
from inv3d_util.image import scale_image
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class RandomDTDOverlay:
    def __init__(
        self,
        dtd_dir: str,
        prob: float = 0.5,
        max_alpha: float = 0.2,
    ):
        self.prob = prob
        self.max_alpha = max_alpha
        self.dtd_dir = Path(dtd_dir)
        self.categories = ["stained", "wrinkled"]

        self.image_files = []

        for category in self.categories:
            self.image_files += list((self.dtd_dir / "images" / category).glob("*.jpg"))

        assert len(self.image_files) > 0, "No images found in DTD directory"

    def __call__(self, results):
        if random.random() < self.prob:
            overlay_image = load_image(random.choice(self.image_files))

            h, w, c = overlay_image.shape
            crop_height = random.randint(h // 2, h - 1)
            crop_width = random.randint(w // 2, w - 1)

            overlay_image = get_random_crop(overlay_image, crop_height, crop_width)

            h, w, c = results["img"].shape
            overlay_image = scale_image(overlay_image, resolution=(h, w))
            overlay_image = overlay_image.astype(np.float32)

            alpha = random.uniform(0.0, self.max_alpha)

            new_image = (1 - alpha) * results["img"] + alpha * overlay_image

            results["img"] = new_image

        return results


def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y : y + crop_height, x : x + crop_width]

    return crop
