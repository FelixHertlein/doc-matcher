import tqdm
from typing import Optional, List
from pathlib import Path
import itertools
from inv3d_util.load import load_image, save_image
from inv3d_util.image import scale_image
from inv3d_util.parallel import process_tasks

from .coco_annotations import Inv3DCocoAnnotations
from ..load_lines import load_lines


def prepare_coco(
    base_dir: Path,
    input_lines_dir: Path,
    input_images_dir: Path,
    input_masks_dir: Path,
    split: str,
    resolution: int,
    line_width: float,
    limit_testval: Optional[int],
    with_mask: bool,
    num_workers: int,
):
    if split == "real":
        return

    output_dir = base_dir / "coco"
    output_dir.mkdir(exist_ok=True)

    annotation_counter = itertools.count()

    lines_files = list(input_lines_dir.glob("*.json"))
    lines_files = sorted(lines_files)

    if limit_testval is not None and split in ["test", "val"]:
        lines_files = lines_files[:limit_testval]

    coco_warpedgt = Inv3DCocoAnnotations(
        output_dir=output_dir,
        split=split,
        rescale_size=resolution,
        line_width=line_width,
        annotation_counter=annotation_counter,
        with_text=True,
        with_mask=with_mask,
    )

    for line_file in tqdm.tqdm(lines_files, desc="Prepare COCO annotations"):
        coco_warpedgt.add_image(
            image_name=line_file.stem,
            image_id=int(line_file.stem),
            lines=load_lines(line_file)["lines"],
            masks_file=input_masks_dir / f"{line_file.stem}.png",
        )

    coco_warpedgt.export()

    prepare_coco_images(
        lines_files=lines_files,
        input_images_dir=input_images_dir,
        output_dir=output_dir,
        split=split,
        resolution=resolution,
        num_workers=num_workers,
    )


def prepare_coco_images(
    lines_files: List[Path],
    input_images_dir: Path,
    output_dir: Path,
    split: str,
    resolution: int,
    num_workers: int,
):
    output_images_dir = output_dir / f"{split}_images"
    output_images_dir.mkdir(exist_ok=True)

    tasks = []

    for line_file in tqdm.tqdm(lines_files, desc="Export COCO images"):
        image_name = line_file.stem
        input_image_file = input_images_dir / f"{image_name}.jpg"
        output_image_file = output_images_dir / f"{image_name}.jpg"

        tasks.append(
            {
                "input_image_file": input_image_file,
                "output_image_file": output_image_file,
                "resolution": resolution,
            }
        )

    process_tasks(
        prepare_coco_image_task,
        tasks,
        num_workers=num_workers,
        use_indexes=True,
        desc="Prepare COCO images",
    )


def prepare_coco_image_task(task: dict):
    prepare_coco_image(**task)


def prepare_coco_image(
    input_image_file: Path, output_image_file: Path, resolution: int
):
    image = load_image(input_image_file)
    image = scale_image(image, resolution=resolution)
    save_image(output_image_file, image, override=True)
