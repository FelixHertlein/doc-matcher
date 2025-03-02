import Levenshtein
import numpy as np

from typing import Optional, List, Dict
from dataclasses import dataclass
from shapely.geometry import Polygon
from doctr.io.elements import Document
from scipy.optimize import linear_sum_assignment


@dataclass
class Word:
    text: str
    bbox: Polygon

    def to_dict(self) -> Dict:
        return {"text": self.text, "bbox": self.bbox.wkt}


@dataclass
class Matching:
    true_word: Word
    norm_word: Optional[Word]
    similarity: float


def document2words(document: Document) -> List[Word]:
    assert len(document.pages) == 1

    words = []
    page = document.pages[0]
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                (x_min, y_min), (x_max, y_max) = word.geometry
                words.append(
                    Word(
                        text=word.value,
                        bbox=Polygon(
                            [
                                (x_min, y_min),
                                (x_max, y_min),
                                (x_max, y_max),
                                (x_min, y_max),
                            ]
                        ),
                    )
                )

    return words


def calculate_mncer(
    true_words: List[Word], norm_words: List[Word], return_matchings: bool = False
) -> float:
    matchings = match_words(true_words, norm_words)

    ncer_values = []
    for matching in matchings:
        true_word = matching["true_word"]
        norm_word = matching["norm_word"]

        if norm_word is None:
            ncer_values.append(1.0)
        else:
            ncer_values.append(calculate_ncer(true_word.text, norm_word.text))

    mncer = np.array(ncer_values).mean()

    if return_matchings:
        return mncer, matchings

    return mncer


def calculate_ncer(text1: str, text2: str) -> float:
    codes = Levenshtein.opcodes(text1, text2)

    num_correct = 0
    num_substitutions = 0
    num_insertions = 0
    num_deletions = 0

    for code_type, s0, s1, t0, t1 in codes:
        source_substring = text1[s0:s1]
        target_substring = text2[t0:t1]

        if code_type == "equal":
            assert source_substring == target_substring
            num_correct += len(source_substring)
        elif code_type == "insert":
            assert len(source_substring) == 0
            num_insertions += len(target_substring)
        elif code_type == "delete":
            assert len(target_substring) == 0
            num_deletions += len(source_substring)
        elif code_type == "replace":
            assert len(source_substring) == len(target_substring)
            num_substitutions += len(source_substring)
        else:
            raise ValueError(f"Unknown code type: {code_type}")

    ncer = (num_substitutions + num_insertions + num_deletions) / (
        num_substitutions + num_insertions + num_deletions + num_correct
    )

    return ncer


def match_words(true_words: List[Word], norm_words: List[Word]) -> List[Matching]:
    similarity_matrix = np.zeros((len(true_words), len(norm_words)))

    for i, true_word in enumerate(true_words):
        for j, norm_word in enumerate(norm_words):
            similarity_matrix[i, j] = calculate_similarity(
                true_word.bbox, norm_word.bbox
            )

    true_ind, norm_ind = linear_sum_assignment(similarity_matrix, maximize=True)

    matchings = []
    for i, j in zip(true_ind, norm_ind):
        true_word = true_words[i]
        norm_word = norm_words[j]
        similarity = similarity_matrix[i, j]

        matchings.append(
            {"true_word": true_word, "norm_word": norm_word, "similarity": similarity}
        )

    unmatched_true_indexes = list(set(range(len(true_words))) - set(true_ind))
    for unmatched_true_index in unmatched_true_indexes:
        true_word = true_words[unmatched_true_index]
        matchings.append(
            {"true_word": true_word, "norm_word": None, "similarity": float("nan")}
        )

    return matchings


def calculate_iou(a: Polygon, b: Polygon) -> float:
    if not a.intersects(b):
        return 0
    return a.intersection(b).area / a.union(b).area


def calculate_distance(a: Polygon, b: Polygon) -> float:
    return a.distance(b)


def calculate_similarity(a: Polygon, b: Polygon) -> float:
    iou = calculate_iou(a, b)

    if iou > 1e-10:
        return iou

    return -calculate_distance(a, b)
