from tqdm import tqdm
from pathlib import Path
from typing import Dict

from .prepare_lines_generic import prepare_lines_generic
from .prepare_lines_inv3d import prepare_lines_inv3d


def prepare_lines(
    inputs: dict,
    outputs: dict,
    data_dir: str,
    split: str,
    resolution: int,
    with_borders: bool,
    num_workers: int,
    visualize: bool = False,
):
    data_dir = Path(data_dir)

    is_generic_dataset = data_dir.name != "inv3d"

    if is_generic_dataset:
        prepare_lines_generic(
            inputs=inputs,
            outputs=outputs,
            data_dir=data_dir,
            split=split,
            resolution=resolution,
            with_borders=with_borders,
            num_workers=num_workers,
            visualize=visualize,
        )
    else:
        prepare_lines_inv3d(
            inputs=inputs,
            outputs=outputs,
            data_dir=data_dir,
            split=split,
            resolution=resolution,
            with_borders=with_borders,
            num_workers=num_workers,
            visualize=visualize,
        )
