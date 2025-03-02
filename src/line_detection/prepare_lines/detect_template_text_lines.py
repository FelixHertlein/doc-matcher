from functools import lru_cache
import random
from itertools import combinations

import cv2
import matplotlib.pyplot as plt
import numpy as np
from inv3d_util.image import scale_image
from inv3d_util.mapping import transform_coords
from shapely.geometry import LineString, Point
from skimage.morphology import skeletonize
from typing import Dict, List, Tuple
from uuid import uuid4
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
import networkx as nx

from doctr.models import ocr_predictor


@lru_cache(maxsize=1)
def load_doctr_model_cached():
    return ocr_predictor(pretrained=True)


def detect_template_text_lines(
    template: np.ndarray,
    height: int,
    width: int,
    verbose: bool = True,
) -> List[Dict]:

    words = detect_words(template)

    edges = []
    for (word1_idx, word1), (word2_idx, word2) in combinations(enumerate(words), 2):
        combine = should_combine(word1, word2)
        if combine:
            edges.append((word1_idx, word2_idx))

    G = nx.Graph()
    G.add_nodes_from(range(len(words)))
    G.add_edges_from(edges)

    text_lines = []
    for component in nx.connected_components(G):
        words_full = [words[i] for i in component]
        words_template = [word for word in words_full if word["in_template"]]

        template_line, template_width = words2linestring(words_template)
        line_full, full_width = words2linestring(words_full)

        text_lines.append(
            {
                "type": "text_line",
                "id": str(uuid4()),
                "line_template": template_line,
                "width_template": template_width,
                "text_template": words2text(words_template),
                "line_full": line_full,
                "width_full": full_width,
                "text_full": words2text(words_full),
                "words_template": words_template,
                "words_full": words_full,
            }
        )

    return text_lines


def detect_words(image: np.ndarray) -> List[dict]:
    model = load_doctr_model_cached()
    result = model([image])

    words = []

    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                (x0, y0), (x1, y1) = word.geometry

                words.append(
                    {
                        "text": word.value,
                        "top": y0,
                        "left": x0,
                        "width": x1 - x0,
                        "height": y1 - y0,
                        "in_template": True,
                    }
                )

    return words


def words2text(words: List[Dict]) -> str:
    words = list(sorted(words, key=lambda x: x["left"]))

    return " ".join(word["text"] for word in words)


def words2linestring(words: List[Dict]) -> Tuple[LineString, float]:
    boxes = [word2polygon(word) for word in words]

    if len(boxes) == 0:
        return None, None

    union = unary_union(boxes)

    # get min and max coordinates
    minx, miny, maxx, maxy = union.bounds

    midy = (miny + maxy) / 2

    line = LineString([(minx, midy), (maxx, midy)])
    width = maxy - miny

    return line, width


def should_combine(word1: Dict, word2: Dict, min_y_iou: float = 0.9) -> bool:
    y_line1 = LineString([(0, word1["top"]), (0, word1["top"] + word1["height"])])
    y_line2 = LineString([(0, word2["top"]), (0, word2["top"] + word2["height"])])

    y_line1 = y_line1.buffer(1e-5)
    y_line2 = y_line2.buffer(1e-5)

    intersection = y_line1.intersection(y_line2).area
    union = y_line1.union(y_line2).area

    iou = intersection / union

    if iou < min_y_iou:
        return False

    average_word_height = (word1["height"] + word2["height"]) / 2
    allow_factor = 1.2
    allowed_gap = average_word_height * allow_factor

    if (
        word1["left"] + word1["width"]
        < word2["left"]
        < word1["left"] + word1["width"] + allowed_gap
    ):
        return True

    if (
        word2["left"] + word2["width"]
        < word1["left"]
        < word2["left"] + word2["width"] + allowed_gap
    ):
        return True

    return False


def word2polygon(word: Dict) -> Polygon:
    return Polygon(
        [
            (word["left"], word["top"]),
            (word["left"] + word["width"], word["top"]),
            (word["left"] + word["width"], word["top"] + word["height"]),
            (word["left"], word["top"] + word["height"]),
        ]
    )
