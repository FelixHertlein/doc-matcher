from typing import Dict
from pathlib import Path
import os

from colorama import Fore

from util import download_file


def prepare_training(base_dir: Path, inputs: Dict, split: str):
    if split == "real":
        return

    print(
        Fore.GREEN
        + f"[STAGE] Preparing Segment Anything training ({split})"
        + Fore.RESET
    )

    base_dir = Path(base_dir) / "training_data"

    annotations_dir = base_dir / "ann"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    images_dir = base_dir / "img"
    images_dir.mkdir(parents=True, exist_ok=True)

    annotations_link = annotations_dir / split
    images_link = images_dir / split

    if annotations_link.exists():
        annotations_link.unlink()

    if images_link.exists():
        images_link.unlink()

    os.symlink(inputs["masks_HQ_GT"].absolute().as_posix(), annotations_link.as_posix())

    os.symlink(
        inputs["warped_images_HQ"].absolute().as_posix(),
        images_link.absolute().as_posix(),
    )
