from doctr.io import DocumentFile
from shapely.geometry import Polygon
from doctr.models import ocr_predictor
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from itertools import groupby
import numpy as np


from pathlib import Path
from typing import *

import json
import pandas as pd
import lpips
import torch
from Levenshtein import distance
from PIL import Image
from einops import rearrange
from cachetools import cached
from inv3d_util.image import scale_image
from inv3d_util.mapping import scale_map_torch
from inv3d_util.misc import to_numpy_image
from inv3d_util.parallel import process_tasks
from pytesseract import pytesseract
from pytorch_msssim import ms_ssim
import tqdm
from functools import lru_cache
from .mncer import document2words, calculate_mncer


def score_dcotr(states: List[Dict]):
    print("Calculating textual scores")

    tasks = [
        {
            "input_true_image_file": state["true_image_file"],
            "input_norm_image_file": state["norm_image_file"],
            "output_true_words_file": state["output_dir"] / "true_words_doctr.json",
            "output_norm_words_file": state["output_dir"] / "norm_words_doctr.json",
            "output_true_text_file": state["output_dir"] / "true_text_doctr.txt",
            "output_norm_text_file": state["output_dir"] / "norm_text_doctr.txt",
        }
        for state in states
        if state["text_evaluation"]
    ]

    all_results = {}
    for idx, task in tqdm.tqdm(
        enumerate(tasks), total=len(tasks), desc="Calculating doctr scores"
    ):
        all_results[idx] = _score_text_doctr_task(task)

    # all_results = process_tasks(
    # _score_text_doctr_task, tasks, num_workers=num_workers, use_indexes=True
    # )

    for idx, results in all_results.items():
        states[idx]["results"].update(**results)


def _score_text_doctr_task(task: Dict):
    true_doc = DocumentFile.from_images(task["input_true_image_file"].as_posix())
    norm_doc = DocumentFile.from_images(task["input_norm_image_file"].as_posix())
    model = get_model()

    true_result = model(true_doc)
    norm_result = model(norm_doc)

    true_text = collect_text_doctr(true_result)
    norm_text = collect_text_doctr(norm_result)

    true_words = document2words(true_result)
    norm_words = document2words(norm_result)

    mncer = calculate_mncer(true_words=true_words, norm_words=norm_words)

    task["output_true_text_file"].write_text(true_text)
    task["output_norm_text_file"].write_text(norm_text)

    task["output_true_words_file"].write_text(
        json.dumps([word.to_dict() for word in true_words], indent=4)
    )
    task["output_norm_words_file"].write_text(
        json.dumps([word.to_dict() for word in norm_words], indent=4)
    )

    text_distance = distance(true_text, norm_text)

    return {
        "doctr_mncer": mncer,
        "doctr_ed": text_distance,
        "doctr_cer": text_distance / len(true_text),
    }


def collect_text_doctr(result) -> str:
    page = result.pages[0]

    result_text = ""

    for block in page.blocks:
        for line in block.lines:
            result_text += " ".join([word.value for word in line.words]) + "\n"

    return result_text


@lru_cache(maxsize=1)
def get_model():
    return ocr_predictor(
        pretrained=True, det_arch="db_resnet50", reco_arch="crnn_vgg16_bn"
    )


"""
does not work perfectly
sorts blocks by bbox centroid y value and clusters the close blocks with near y value. proceeds to sort blocks within a cluster

def page2text_sorted(page):
    blocks = []
    for block in page.blocks: 
        blocks.append(block)

    def block2poly(line) -> Polygon:
        return bboxdata2polygon(line.geometry)

    block_indicies = sort_stable(map(block2poly, blocks), return_indexes=True)

    blocks = sorted(zip(blocks, block_indicies), key=lambda x: x[1])
    blocks = [block for block, _ in blocks]

    def line2text(line) -> str:
        return " ".join([word.value for word in line.words])

    def block2text(block):
        lines = list(block.lines)
        lines = list(
            sorted(lines, key=lambda line: line.geometry[0][1])
        )  # sort by y_min
        return "\n".join([line2text(line) for line in lines])

    return "\n".join([block2text(block) for block in blocks])


def bboxdata2polygon(data):
    (x1, y1), (x2, y2) = data
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def sort_stable(bboxes, return_indexes=False):
    bboxes = list(sorted(enumerate(bboxes), key=lambda pair: pair[1].centroid.y))
    cluster_data = np.array([pair[1].centroid.y for pair in bboxes]).reshape(-1, 1)
    linkage_matrix = linkage(cluster_data, "single")
    cluster_indexes = fcluster(linkage_matrix, 40 / 2200, criterion="distance")

    final_bboxes = []
    for cluster_idx, group_elements in groupby(
        zip(cluster_indexes, bboxes), key=lambda x: x[0]
    ):
        group_bboxes = [bbox for _, bbox in group_elements]
        group_bboxes = list(sorted(group_bboxes, key=lambda pair: pair[1].centroid.x))
        final_bboxes.extend(group_bboxes)

    if return_indexes:
        return [i for i, _ in final_bboxes]
    else:
        return [bbox for _, bbox in final_bboxes]
"""
