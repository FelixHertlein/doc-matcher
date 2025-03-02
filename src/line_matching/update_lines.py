from typing import Dict
import json
from pathlib import Path
import numpy as np

from ..line_detection.visualization import visualize_lines


def update_lines(input_lines_dir: str, output_lines_dir: str, features_dir: str):
    input_lines_dir = Path(input_lines_dir)
    output_lines_dir = Path(output_lines_dir)
    features_dir = Path(features_dir)

    input_lines_files = list(input_lines_dir.glob("*.json"))

    for input_lines_file in input_lines_files:
        update_lines_file(
            input_lines_file=input_lines_file,
            output_lines_file=output_lines_dir / input_lines_file.name,
            features_file=features_dir / f"{input_lines_file.stem}.npz",
        )


def update_lines_file(
    input_lines_file: Path,
    output_lines_file: Path,
    features_file: Path,
):
    # gather texts and polygons from npz files
    features = np.load(features_file, allow_pickle=True)

    texts = {}
    words = {}

    for feature_name in features.files:
        line_id = feature_name.split(".", 1)[0]

        if feature_name.endswith("text"):
            texts[line_id] = str(features[feature_name])

        if feature_name.endswith("words"):
            words[line_id] = features[feature_name]
            words_data = features[feature_name].tolist()
            words[line_id] = words_data

    # update lines
    lines_data = json.loads(input_lines_file.read_text())

    for line in lines_data["lines"]:
        line["text"] = texts.get(line["id"], None)
        line["words"] = words2wkt(words.get(line["id"], None))

        del texts[line["id"]]
        del words[line["id"]]

    assert len(texts) == 0
    assert len(words) == 0

    # export updated lines
    output_lines_file.write_text(json.dumps(lines_data))


def words2wkt(words: Dict):
    if words is None:
        return None

    def towkt(word: Dict):
        return {k: v.wkt if k == "poly" else v for k, v in word.items()}

    return [towkt(word) for word in words]
