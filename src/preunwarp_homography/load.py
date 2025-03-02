import numpy as np
import json
from pathlib import Path
from shapely import wkt
from shapely.geometry import Polygon
from shapely.affinity import scale
from typing import List, Dict, Any, Tuple
from inv3d_util.image import scale_image
from inv3d_util.load import load_image


def load(
    lines_file: Path, mask_file: Path, input_image_hq_file: Path, input_image_file: Path
) -> Tuple[np.ndarray, List[Dict[str, Any]], Polygon]:
    data = json.loads(lines_file.read_text())

    image_orig = load_image(input_image_hq_file)
    height, width, channels = image_orig.shape

    image = load_image(input_image_file)

    lines = data["lines"]
    for line_data in lines:
        line_data["line"] = wkt.loads(line_data["line"])

    mask = wkt.loads(json.loads(mask_file.read_text())["mask"])
    mask = scale(mask, xfact=512 / width, yfact=512 / height, origin=(0, 0))

    return data, image, image_orig, mask
