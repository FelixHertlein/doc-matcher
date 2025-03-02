import pylcs
import json
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
from shapely.geometry import LineString, Point
from shapely.ops import substring


from ..line_matching.line_lightglue.visualize import visualize_matching_inner
from ..line_detection.load_lines import load_lines, save_lines


def clean_text_lines(
    input_template_line_dir: Path,
    input_warped_line_dir: Path,
    input_matches_dir: Path,
    output_template_line_dir: Path,
    output_warped_line_dir: Path,
    visualize: bool,
    template_image_dir: Path,
    warped_image_dir: Path,
    min_text_longest_common_substring: int,
    min_text_length: int,
):
    input_template_lines_files = list(input_template_line_dir.glob("*.json"))
    input_template_lines_files.sort()

    for template_lines_file in tqdm(input_template_lines_files, "Cleaning text lines"):
        clean_text_lines_sample(
            input_template_lines_file=template_lines_file,
            input_warped_lines_file=input_warped_line_dir / template_lines_file.name,
            input_matches_file=input_matches_dir / template_lines_file.name,
            output_template_lines_file=output_template_line_dir
            / template_lines_file.name,
            output_warped_lines_file=output_warped_line_dir / template_lines_file.name,
            visualize=visualize,
            template_image_file=template_image_dir / f"{template_lines_file.stem}.jpg",
            warped_image_file=warped_image_dir / f"{template_lines_file.stem}.jpg",
            min_text_longest_common_substring=min_text_longest_common_substring,
            min_text_length=min_text_length,
        )


def clean_text_lines_sample(
    input_template_lines_file: Path,
    input_warped_lines_file: Path,
    input_matches_file: Path,
    output_template_lines_file: Path,
    output_warped_lines_file: Path,
    visualize: bool,
    template_image_file: Path,
    warped_image_file: Path,
    min_text_longest_common_substring: int,
    min_text_length: int,
):
    template_lines = load_lines(input_template_lines_file)
    warped_lines = load_lines(input_warped_lines_file)

    matches = json.loads(input_matches_file.read_text())

    template_line_map = {l["id"].split("-", 1)[1]: l for l in template_lines["lines"]}
    warped_line_map = {l["id"].split("-", 1)[1]: l for l in warped_lines["lines"]}

    new_lines = {}
    new_texts = {}

    # generate new lines for all mateches based on the longest common substring
    for match in matches["line_matches"]:
        template_line_data = template_line_map[match["template"]]
        warped_line_data = warped_line_map[match["warped"]]

        if template_line_data["type"] != "text_line":
            continue

        words_template = template_line_data["words"]
        words_warped = warped_line_data["words"]

        template_line = template_line_data["line"]
        warped_line = warped_line_data["line"]

        chars_template = [
            char for word in words_template for char in word2chars(word, template_line)
        ]
        chars_warped = [
            char for word in words_warped for char in word2chars(word, warped_line)
        ]

        chars_template = list(sorted(chars_template, key=lambda char: char["pos"]))
        chars_warped = list(sorted(chars_warped, key=lambda char: char["pos"]))

        text_template = "".join(char["value"] for char in chars_template)
        text_warped = "".join(char["value"] for char in chars_warped)

        mapping_idxs = pylcs.lcs_string_idx(text_template, text_warped)

        common_text = "".join(
            warped_line_data["text"][i] for i in mapping_idxs if i >= 0
        )

        mapping = [
            {
                "template_char": chars_template[idx],
                "warped_char": chars_warped[warpedgt_idx],
            }
            for idx, warpedgt_idx in enumerate(mapping_idxs)
            if warpedgt_idx >= 0
        ]

        if len(mapping) < min_text_longest_common_substring:
            new_lines[template_line_data["id"]] = None
            new_lines[warped_line_data["id"]] = None
            new_texts[template_line_data["id"]] = None
            new_texts[warped_line_data["id"]] = None
            continue

        template_line_substr = substring(
            template_line_data["line"],
            start_dist=mapping[0]["template_char"]["pos"],
            end_dist=mapping[-1]["template_char"]["pos"],
        )
        warped_line_substr = substring(
            warped_line_data["line"],
            start_dist=mapping[0]["warped_char"]["pos"],
            end_dist=mapping[-1]["warped_char"]["pos"],
        )

        if (
            template_line_substr.geom_type != "LineString"
            or warped_line_substr.geom_type != "LineString"
        ):
            new_lines[template_line_data["id"]] = None
            new_lines[warped_line_data["id"]] = None
            new_texts[template_line_data["id"]] = None
            new_texts[warped_line_data["id"]] = None
            continue

        new_lines[template_line_data["id"]] = template_line_substr
        new_lines[warped_line_data["id"]] = warped_line_substr

        new_texts[template_line_data["id"]] = common_text
        new_texts[warped_line_data["id"]] = common_text

    unmatched_line_ids = matches["unmatched_warped_lines"] + [
        m["warped"] for m in matches["removed_line_matches"]
    ]

    # generate new lines for all unamteched lines based on the detected characters
    for unmatched_line_id in unmatched_line_ids:
        line_data = warped_line_map[unmatched_line_id]
        line = line_data["line"]

        if line_data["type"] != "text_line":
            continue

        chars = [char for word in line_data["words"] for char in word2chars(word, line)]
        chars = list(sorted(chars, key=lambda char: char["pos"]))

        if len(chars) < min_text_length:
            new_texts[line_data["id"]] = None
            new_lines[line_data["id"]] = None
            continue

        new_text = "".join(char["value"] for char in chars)

        line_substr = substring(
            line_data["line"],
            start_dist=chars[0]["pos"],
            end_dist=chars[-1]["pos"],
        )

        if line_substr.geom_type != "LineString":
            new_texts[line_data["id"]] = None
            new_lines[line_data["id"]] = None
            continue

        new_texts[line_data["id"]] = new_text
        new_lines[line_data["id"]] = line_substr

    # copy results to final line
    for line_data in template_lines["lines"] + warped_lines["lines"]:
        line_data["words"] = None

        if line_data["id"] not in new_lines:
            continue

        line_data["line"] = new_lines[line_data["id"]]
        line_data["text"] = new_texts[line_data["id"]]

        new_texts.pop(line_data["id"])
        new_lines.pop(line_data["id"])

    assert len(new_texts) == 0
    assert len(new_lines) == 0

    # remove filtered lines
    template_lines["lines"] = [
        l for l in template_lines["lines"] if l["line"] is not None
    ]
    warped_lines["lines"] = [l for l in warped_lines["lines"] if l["line"] is not None]

    # export lines
    save_lines(output_template_lines_file, template_lines)
    save_lines(output_warped_lines_file, warped_lines)

    if visualize:
        visualize_matching_inner(
            matches_file=input_matches_file,
            template_lines_file=output_template_lines_file,
            warped_lines_file=output_warped_lines_file,
            template_image_file=template_image_file,
            warped_image_file=warped_image_file,
            output_file=output_warped_lines_file.with_suffix(".jpg"),
        )


def word2chars(word: Dict, line: LineString) -> List[Dict]:
    num_chars = len(word["text"])

    assert len(word["poly"].exterior.coords) == 5

    start = word["poly"].exterior.coords[0]
    end = word["poly"].exterior.coords[1]

    start = line.project(Point(start))
    end = line.project(Point(end))

    char_positions = np.linspace(start, end, num_chars * 2 + 1)[1::2]

    return [
        {
            "value": char,
            "pos": float(pos),
        }
        for char, pos in zip(word["text"], char_positions)
    ]
