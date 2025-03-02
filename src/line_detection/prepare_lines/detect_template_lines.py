import json
from pathlib import Path
from typing import Dict, Optional
from shapely.affinity import scale
from shapely.geometry import Polygon

from inv3d_util.load import load_image

from .line_utils import compress_line, scale_line
from .find_template_structural_lines import find_template_structural_lines
from .find_template_text_lines import find_template_text_lines
from .detect_template_text_lines import detect_template_text_lines


def detect_template_lines(
    template_file: Path,
    rescale_size: int,
    with_borders: bool,
):
    template = load_image(template_file)

    template_lines = []
    template_lines.extend(
        find_template_structural_lines(template, with_borders, verbose=False)
    )
    template_lines.extend(
        detect_template_text_lines(
            template=template,
            height=template.shape[0],
            width=template.shape[1],
            verbose=False,
        )
    )

    def postprocess(line_data, key):
        if key not in line_data or line_data[key] is None:
            return None

        line = line_data[key]
        line = scale_line(line, rescale_size)
        new_line_data = compress_line({"line": line})
        return new_line_data["line"]

    def postprocess_width(line_data, key):
        if key not in line_data or line_data[key] is None:
            return None

        return int(line_data[key] * rescale_size)

    def postprocess_word(word: Dict) -> Dict:
        poly = Polygon(
            [
                (word["left"], word["top"]),
                (word["left"] + word["width"], word["top"]),
                (word["left"] + word["width"], word["top"] + word["height"]),
                (word["left"], word["top"] + word["height"]),
            ]
        )
        poly = scale(poly, xfact=rescale_size, yfact=rescale_size, origin=(0, 0))
        return {
            "text": word["text"],
            "poly": poly,
        }

    def postprocess_words(line_data, key) -> Optional[Dict]:
        if key not in line_data or line_data[key] is None:
            return None

        words = line_data[key]
        words = [postprocess_word(word) for word in words]
        return words

    # remap ids to make unique with wapredgt lines
    template_lines = [
        {
            "type": line["type"],
            "id": "template-" + line["id"],
            "line": postprocess(line, "line"),
            "line_template": postprocess(line, "line_template"),
            "line_full": postprocess(line, "line_full"),
            "width": postprocess_width(line, "width"),
            "width_template": postprocess_width(line, "width_template"),
            "width_full": postprocess_width(line, "width_full"),
            "text_template": line.get("text_template", None),
            "text_full": line.get("text_full", None),
            "words_template": postprocess_words(line, "words_template"),
            "words_full": postprocess_words(line, "words_full"),
        }
        for line in template_lines
    ]

    return template_lines
