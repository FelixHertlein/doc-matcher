import json
import random
from pathlib import Path
import matplotlib.pyplot as plt

from typing import Optional
from shapely import wkt

import PIL

from ..load_lines import load_lines


def visualize_lines(lines_file: Path, image_file: Optional[Path] = None):
    lines_file = Path(lines_file)

    plt.clf()
    lines_data = load_lines(lines_file)
    lines = lines_data["lines"]

    if image_file is None:
        image_file = Path(lines_data["image_path"])

    if image_file is not None and image_file.is_file():
        plt.imshow(PIL.Image.open(image_file.as_posix()))
    else:
        print(f"Warning: Image '{image_file}' not found")

    for line_data in lines:

        line = line_data["line"]
        color = "#" + "%06x" % random.randint(0, 0xFFFFFF)  # generate random color

        if line_data["width"] is None:
            plt.plot(line.xy[0], line.xy[1], color=color, linewidth=3)
        else:
            poly = line.buffer(line_data["width"] / 2)
            if poly.geom_type == "MultiPolygon":  # rare edge case
                for p in poly.geoms:
                    plt.plot(*p.exterior.xy, color=color)
            else:
                plt.plot(*poly.exterior.xy, color=color)

        words = line_data.get("words", None)

        if words is None:
            continue

        for word in words:
            poly = word["poly"]
            plt.fill(*poly.exterior.xy, alpha=0.25, color="red")

    # set x min to -1 and max to 513
    plt.xlim(-1, 513)
    plt.ylim(-1, 513)
    plt.gca().invert_yaxis()

    plt.savefig(lines_file.parent / f"{lines_file.stem}.jpg")
