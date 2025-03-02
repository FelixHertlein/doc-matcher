import json
from pathlib import Path
from typing import Dict
from shapely import wkt


def load_lines(lines_file: str) -> Dict:
    lines_file = Path(lines_file)
    lines_data = json.loads(lines_file.read_text())

    lines_data["lines"] = [
        _load_lines_data(line_data) for line_data in lines_data["lines"]
    ]

    return lines_data


def _load_lines_data(line_data: Dict) -> Dict:
    line_data["line"] = wkt.loads(line_data["line"])

    if "words" in line_data and line_data["words"] is not None:
        line_data["words"] = [
            _load_word_data(word_data) for word_data in line_data["words"]
        ]

    return line_data


def _load_word_data(word_data: Dict) -> Dict:
    word_data["poly"] = wkt.loads(word_data["poly"])
    return word_data


def save_lines(lines_file: str, lines_data: Dict):
    def map_data(data):
        if isinstance(data, dict):
            return {k: map_data(v) for k, v in data.items()}

        if isinstance(data, list):
            return [map_data(v) for v in data]

        if hasattr(data, "wkt"):
            return data.wkt

        return data

    mapped_data = map_data(lines_data)

    Path(lines_file).write_text(json.dumps(mapped_data))
