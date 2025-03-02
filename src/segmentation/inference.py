import sys
from colorama import Fore
import numpy as np
import cv2
from scipy import ndimage
import tqdm
import argparse
import json
import torch
from typing import Optional
import rasterio.features
from PIL import Image
from shapely import wkt
from shapely.affinity import scale, rotate
from shapely.geometry import Polygon
from shapely.validation import make_valid
from pathlib import Path
from inv3d_util.load import load_image, save_image
from inv3d_util.image import scale_image
from typing import Dict
import random

random.seed(42)

fintune_anything_dir = Path(__file__).parent / "finetune_anything"
sys.path.insert(0, str(fintune_anything_dir.resolve()))

from .finetune_anything.extend_sam import SemanticSam


def inference(
    inputs: Dict,
    outputs: Dict,
    model_checkpoint: Path,
    model_original: Path,
    model_type: str,
    max_rotation_angle: Optional[int],
    gpu: int,
):
    print(Fore.GREEN + "[STAGE] Detecting documents" + Fore.RESET)

    input_images_HQ_dir = inputs["warped_images_HQ"]
    output_mask_dir = outputs["masks"]
    output_images_HQ_dir = outputs["warped_images_HQ"]
    output_images_dir = outputs["warped_images"]

    with torch.no_grad():
        device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"

        model = SemanticSam(
            ckpt_path=None,
            fix_img_en=True,
            fix_prompt_en=True,
            fix_mask_de=False,
            class_num=2,
            model_type=model_type,
        )
        model.load_state_dict(
            torch.load(model_checkpoint.as_posix(), map_location="cpu")
        )
        model.to(device)
        model.eval()

        image_files = list(input_images_HQ_dir.glob("*.jpg"))

        for image_file in tqdm.tqdm(image_files, desc="Inference"):
            image_orig = load_image(image_file)
            image = scale_image(image_orig, resolution=1024)
            image = (
                torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            )
            image = image.to(device)

            masks, _ = model(image)

            masks = masks.detach().cpu().numpy()
            masks = masks.squeeze(0)

            mask = (masks[1] - masks[0]) > 0
            mask = largest_connected_component(mask)
            mask = apply_closing_operation(mask)
            mask = fill_holes(mask)

            polygon = find_polygon(mask)
            polygon = polygon.simplify(2)

            h, w, c = image_orig.shape
            polygon = scale(polygon, xfact=w / 256, yfact=h / 256, origin=(0, 0))

            if max_rotation_angle is not None:  # ablation study
                polygon = rotate(
                    polygon,
                    angle=random.randint(-max_rotation_angle, max_rotation_angle),
                    origin=polygon.centroid,
                )

            # export mask as wkt
            output_data = {
                "image": image_file.as_posix(),
                "mask": wkt.dumps(polygon),
            }
            output_json_file = output_mask_dir / f"{image_file.stem}.json"
            output_json_file.write_text(json.dumps(output_data))

            # export mask as png
            output_png_file = output_mask_dir / f"{image_file.stem}.png"

            output_mask_np = rasterio.features.rasterize(
                [polygon], out_shape=(h, w)
            ).astype(np.bool_)

            palette = [255, 255, 255, 255, 105, 97] + [0] * 254 * 3

            output_mask = Image.fromarray(output_mask_np.astype(np.uint8))
            output_mask = output_mask.convert("P")
            output_mask.putpalette(palette)
            output_mask.save(output_png_file)

            # export image with removed background
            image_orig[~output_mask_np] = 0

            save_image(
                output_images_HQ_dir / f"{image_file.stem}.jpg",
                image_orig,
                override=True,
            )
            save_image(
                output_images_dir / f"{image_file.stem}.jpg",
                scale_image(image_orig, resolution=512),
                override=True,
            )

    torch.cuda.empty_cache()


def apply_closing_operation(mask):
    # Define a kernel for the closing operation
    kernel = np.ones((5, 5), np.uint8)  # You can adjust the kernel size as needed

    # Apply closing operation using OpenCV
    closed_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return closed_mask.astype(bool)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    # Label connected components in the input array
    labeled_array, num_features = ndimage.label(mask)

    if num_features == 0:
        # No connected components found
        return np.zeros_like(mask)

    # Calculate the sizes of the connected components
    component_sizes = np.bincount(labeled_array.ravel())

    # Find the label of the largest connected component
    largest_component_label = np.argmax(component_sizes[1:]) + 1

    # Create an array containing only the largest connected component
    largest_component = labeled_array == largest_component_label

    return largest_component.astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    # Convert boolean array to integer (0s and 1s)
    binary_array = mask.astype(np.int_)

    # Use binary_fill_holes function to fill holes
    filled_array = ndimage.binary_fill_holes(binary_array)

    # Convert back to boolean array
    filled_boolean_array = filled_array.astype(np.uint8)

    return filled_boolean_array


def find_polygon(mask: np.ndarray) -> Polygon:
    height, width = mask.shape
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
