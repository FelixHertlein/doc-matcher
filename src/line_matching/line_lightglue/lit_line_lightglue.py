import pytorch_lightning as pl
from .line_lightglue_model import LineLightGlue
import torch

from torch import nn, optim


class LitLineLightglue(pl.LightningModule):
    def __init__(self):
        super().__init__()

        conf = {
            "name": "matchers.linelightglue",
            "filter_threshold": 0.1,
            "flash": False,
            "checkpointed": True,
        }

        self.model = LineLightGlue(conf)
        self.learning_rate = 0.0001

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, mode="train"):
        out = self.model(batch)

        losses, metrics = self.model.loss(out, batch)

        if torch.isnan(losses["total"]).any():
            return None  # NaN loss found. Skipping batch!

        batch_size = batch["descriptors0"].shape[0]
        for loss_name, loss in losses.items():
            self.log(
                f"{mode}/loss/{loss_name}", torch.mean(loss), batch_size=batch_size
            )

        for metric_name, metric in metrics.items():
            self.log(
                f"{mode}/metric/{metric_name}",
                torch.mean(metric),
                batch_size=batch_size,
            )

        return torch.mean(losses["total"])

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.training_step(batch, batch_idx, mode="val")

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr=self.learning_rate)

        scheduler = get_original_lr_scheduler(optimizer)

        return [optimizer], [scheduler]


def get_original_lr_scheduler(optimizer):
    exp_div_10 = 10
    start = 20

    # backward compatibility
    def lr_fn(it):  # noqa: E306
        gam = 10 ** (-1 / exp_div_10)
        return 1.0 if it < start else gam

    return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)
