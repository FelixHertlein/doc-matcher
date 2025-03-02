from pathlib import Path
from typing import Optional

from colorama import Fore

from preparation.prepare_generic_data import prepare_generic_data
from preparation.prepare_inv3d_data import prepare_inv3d_data


def prepare_data(
    data_dir: Path,
    resolution: int,
    split: str,
    outputs: dict,
    is_generic_dataset: bool,
    limit_samples: Optional[int],
):
    print(Fore.GREEN + f"[STAGE] Preparing dataset ({split})" + Fore.RESET)
    if is_generic_dataset:
        prepare_generic_data(data_dir, resolution, split, outputs, limit_samples)
    else:
        prepare_inv3d_data(data_dir, resolution, split, outputs, limit_samples)
