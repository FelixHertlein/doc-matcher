import json
from pathlib import Path
from typing import Dict, List


def export_template_lines(
    output_file: Path, lines: List[Dict], image_path: Path, is_template: bool
):

    def select_line(line_data):
        if is_template:
            return line_data["line_template"] or line_data["line"]
        else:
            return line_data["line_full"] or line_data["line"]

    def select_width(line_data):
        if is_template:
            return line_data["width_template"] or line_data["width"]
        else:
            return line_data["width_full"] or line_data["width"]

    def select_text(line_data):
        if is_template:
            return line_data.get("text_template", None)
        else:
            return line_data.get("text_full", None)

    def select_word(line_data):
        if is_template:
            return line_data.get("words_template", None)
        else:
            return line_data.get("words_full", None)

    def map_words(words: List[Dict]) -> List[Dict]:
        if words is None:
            return None

        return [
            {
                "text": word["text"],
                "poly": word["poly"].wkt,
            }
            for word in words
        ]

    data = {
        "image_path": str(image_path),
        "lines": [
            {
                "type": line_data["type"],
                "id": line_data["id"],
                "line": select_line(line_data).wkt,
                "width": select_width(line_data),
                "text": select_text(line_data),
                "words": map_words(select_word(line_data)),
            }
            for line_data in lines
            if select_line(line_data) is not None
        ],
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, output_file.open("w"))
