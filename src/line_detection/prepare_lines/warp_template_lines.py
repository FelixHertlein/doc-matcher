from pathlib import Path
from typing import Dict, List, Optional


from inv3d_util.load import load_npz
from inv3d_util.mapping import tight_crop_map, transform_coords
import numpy as np
from shapely.affinity import scale
from shapely.geometry import Polygon, Point

from .line_utils import compress_line, scale_line, project_line


def warp_template_lines(BM_file: Path, template_lines: List[Dict], rescale_size: int):
    bm = load_npz(BM_file)
    bm = tight_crop_map(bm)

    def warp_line(line_data, key):
        if key not in line_data or line_data[key] is None:
            return None

        line = line_data[key]
        line = scale_line(line, 1 / rescale_size)
        line = project_line(line, bm)
        line = scale_line(line, rescale_size)
        new_line_data = compress_line({"line": line})
        return new_line_data["line"]

    def project_word(word: Dict) -> Dict:
        poly = word["poly"]

        poly = scale(
            poly, xfact=1 / rescale_size, yfact=1 / rescale_size, origin=(0, 0)
        )

        bm_rolled = np.roll(bm, axis=2, shift=1)
        poly = Polygon(
            [
                Point(
                    transform_coords(
                        bm_rolled, np.expand_dims(np.roll(np.array(p), shift=1), 0)
                    )
                )
                for p in poly.exterior.coords
            ]
        )
        poly = scale(poly, xfact=rescale_size, yfact=rescale_size, origin=(0, 0))
        return {
            "text": word["text"],
            "poly": poly,
        }

    def project_words(words: Optional[List[Dict]]) -> Optional[Dict]:
        if words is None:
            return None
        return [project_word(word) for word in words]

    return [
        {
            "type": line["type"],
            "id": line["id"].replace("template-", "warpedgt-"),
            "line": warp_line(line, "line"),
            "line_template": warp_line(line, "line_template"),
            "line_full": warp_line(line, "line_full"),
            "width": line["width"],
            "width_template": line["width_template"],
            "width_full": line["width_full"],
            "text_template": line["text_template"],
            "text_full": line["text_full"],
            "words_template": project_words(line["words_template"]),
            "words_full": project_words(line["words_full"]),
        }
        for line in template_lines
    ]
