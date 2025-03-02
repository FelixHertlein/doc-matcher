from pathlib import Path
from typing import Dict
from .line_lightglue.train import train_line_lightglue


def train(
    feature_dir: Path,
    log_dir: str,
    statistics: Dict,
    min_num_features: int,
    batch_size: int,
    limit_val_samples: int,
    num_workers: int,
    gpu: int,
):
    log_dir.mkdir(parents=True, exist_ok=True)

    train_template_feature_dir = feature_dir / "features" / "template" / "train"
    train_warped_feature_dir = feature_dir / "features" / "warped_GT" / "train"

    val_template_feature_dir = feature_dir / "features" / "template" / "val"
    val_warped_feature_dir = feature_dir / "features" / "warped_GT" / "val"

    train_line_lightglue(
        train_template_feature_dir=train_template_feature_dir,
        train_warped_feature_dir=train_warped_feature_dir,
        val_template_feature_dir=val_template_feature_dir,
        val_warped_feature_dir=val_warped_feature_dir,
        log_dir=log_dir,
        statistics=statistics,
        min_num_features=min_num_features,
        batch_size=batch_size,
        limit_val_samples=limit_val_samples,
        num_workers=num_workers,
        gpu=gpu,
    )
