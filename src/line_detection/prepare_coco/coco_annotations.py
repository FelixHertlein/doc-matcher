import cv2
import json
from pathlib import Path
from typing import Dict, Tuple, List
import itertools

import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.validation import make_valid
from shapely.affinity import scale

from inv3d_util.load import load_image


class Inv3DCocoAnnotations:
    def __init__(
        self,
        output_dir: Path,
        split: str,
        rescale_size: int,
        line_width: int,
        annotation_counter: itertools.count,
        with_text: bool,
        with_mask: bool,
    ) -> None:
        self.annotations_output_dir = output_dir / f"annotations"
        self.annotations_output_dir.mkdir(parents=True, exist_ok=True)

        self.annotations_output_file = (
            self.annotations_output_dir / f"{split}_coco_annot.json"
        )
        self.split = split
        self.rescale_size = rescale_size
        self.line_width = line_width
        self.poly_simplify_tol = 0.1

        self.categories = ["line"]
        self.with_text = with_text
        self.with_mask = with_mask
        if with_text:
            self.categories.append("text")

        if with_mask:
            self.categories.append("document")

        # shared state
        self.annotation_counter = annotation_counter

        self.image_data = []
        self.annotations = []

    def add_image(
        self,
        image_name: str,
        image_id: int,
        lines: List[Dict],
        masks_file: Path,
    ):
        self._prepare_image_annotation(
            image_name=image_name,
            image_id=image_id,
        )

        self._prepare_annotations(
            image_id=image_id,
            lines=lines,
        )

        if self.with_mask:
            self._prepare_mask(
                image_id=image_id,
                masks_file=masks_file,
            )

    def export(self):
        coco_data = {
            "info": {
                "description": "Inv3D: A high-resolution 3D invoice dataset for template-guided single-image document unwarping",
                "url": "https://felixhertlein.github.io/inv3d/",
                "version": "1.0",
                "year": 2023,
                "date_created": "2023-08-14T00:00:00",
            },
            "licenses": [],
            "images": self.image_data,
            "annotations": self.annotations,
            "categories": [
                {"id": category_id, "name": category}
                for category_id, category in enumerate(self.categories)
            ],
        }

        self.annotations_output_file.write_text(json.dumps(coco_data))

    def _prepare_image_annotation(
        self,
        image_name: str,
        image_id: str,
    ) -> Tuple[np.ndarray, Dict]:

        self.image_data.append(
            {
                "id": int(image_id),
                "width": self.rescale_size,
                "height": self.rescale_size,
                "file_name": f"{image_name}.jpg",
            }
        )

    def _prepare_annotations(
        self,
        image_id: str,
        lines: List[Dict],
    ) -> Dict:

        # add polygons to each line
        for line_data in lines:
            line = line_data["line"]
            line_width = line_data["width"]

            if line is None:
                continue

            line_width = line_width or self.line_width
            offset_value = line_width / 2

            right_line = line.parallel_offset(offset_value, "right")
            left_line = line.parallel_offset(offset_value, "left")

            polygon = Polygon(list(right_line.coords) + list(left_line.coords[::-1]))
            polygon = polygon.simplify(self.poly_simplify_tol)

            polygon = Polygon(
                [
                    [round(coord, 4) for coord in point]
                    for point in polygon.exterior.coords
                ]
            )

            polygon = make_valid(polygon)

            if polygon.type == "MultiPolygon":
                polygon = sorted(
                    [p for p in polygon.geoms], key=lambda p: p.area, reverse=True
                )[0]

            # validate polygons
            assert polygon.is_valid, "Mask must be valid"
            assert polygon.is_simple, "Mask must not intersect itself"
            assert len(list(polygon.exterior.coords)) >= 4, "At least 3 points required"

            line_data["polygon"] = polygon

        for line_data in lines:

            if "polygon" not in line_data:
                continue

            minx, miny, maxx, maxy = line_data["polygon"].bounds

            category = "text" if "text" in line_data["type"] else "line"

            self.annotations.append(
                {
                    "segmentation": [
                        [
                            value
                            for coord in line_data["polygon"].exterior.coords
                            for value in coord
                        ]
                    ],
                    "area": line_data["polygon"].area,
                    "iscrowd": 0,
                    "image_id": int(image_id),
                    "bbox": [minx, miny, maxx - minx, maxy - miny],
                    "category_id": self.categories.index(category),
                    "id": next(self.annotation_counter),
                }
            )

    def _prepare_mask(
        self,
        image_id: int,
        masks_file: Path,
    ) -> None:
        mask = np.array(Image.open(masks_file.as_posix()))
        polygon = find_polygon(mask)

        h, w = mask.shape

        scale_x = self.rescale_size / w
        scale_y = self.rescale_size / h

        polygon = polygon.simplify(1)
        polygon = scale(polygon, xfact=scale_x, yfact=scale_y, origin=(0, 0))

        minx, miny, maxx, maxy = polygon.bounds

        self.annotations.append(
            {
                "segmentation": [
                    [value for coord in polygon.exterior.coords for value in coord]
                ],
                "area": polygon.area,
                "iscrowd": 0,
                "image_id": int(image_id),
                "bbox": [minx, miny, maxx - minx, maxy - miny],
                "category_id": self.categories.index("document"),
                "id": next(self.annotation_counter),
            }
        )


def find_polygon(mask: np.ndarray) -> Polygon:
    contours, _ = cv2.findContours(
        mask.astype("uint8") * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=lambda x: x.shape[0])
    contour = np.squeeze(contour).astype("float")

    polygon = Polygon(contour)
    polygon = make_valid(polygon)

    while hasattr(polygon, "geoms"):
        # print(f"Found shape {type(polygon)} instead of polygon! Unpacking shape!")
        polygon = max(polygon.geoms, key=lambda x: x.area)

    assert isinstance(
        polygon, Polygon
    ), f"Found shape {type(polygon)} instead of polygon! Abort!"

    return polygon
