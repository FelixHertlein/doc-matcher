from typing import Dict
import argparse
import json
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path

from ..feature_extractor.dataset import FrenetDataset, collate_fn

from .lit_line_lightglue import LitLineLightglue


def train_line_lightglue(
    train_template_feature_dir: str,
    train_warped_feature_dir: str,
    val_template_feature_dir: str,
    val_warped_feature_dir: str,
    log_dir: str,
    statistics: Dict,
    min_num_features: int,
    batch_size: int,
    limit_val_samples: int,
    num_workers: int,
    gpu: int,
):
    torch.set_float32_matmul_precision("high")

    model = LitLineLightglue()

    trainer = pl.Trainer(
        max_epochs=-1,
        fast_dev_run=False,
        accelerator="gpu",
        devices=[gpu],
        default_root_dir=log_dir,
    )

    output_dir = Path(log_dir) / "lightning_logs" / f"version_{trainer.logger.version}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # prepare data
    train_dataset = FrenetDataset(
        template_features_dir=Path(train_template_feature_dir),
        warped_features_dir=Path(train_warped_feature_dir),
        statistics=statistics,
        split="train",
        min_num_features=min_num_features,
        max_samples=None,
    )
    val_dataset = FrenetDataset(
        template_features_dir=Path(val_template_feature_dir),
        warped_features_dir=Path(val_warped_feature_dir),
        statistics=statistics,
        split="val",
        min_num_features=min_num_features,
        max_samples=limit_val_samples,
    )

    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=min(num_workers, train_dataset_size),
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=min(num_workers, val_dataset_size),
        drop_last=True,
        collate_fn=collate_fn,
    )

    # prepare callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/metric/accuracy",
        filename="line-lightglue-epoch={epoch:03d}-val_accuracy={val/metric/accuracy:.3f}",
        save_top_k=1,
        mode="max",
        save_last=True,
        auto_insert_metric_name=False,
    )
    trainer.callbacks.append(checkpoint_callback)

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val/metric/accuracy", patience=10, mode="max"
    )
    trainer.callbacks.append(early_stop_callback)

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
