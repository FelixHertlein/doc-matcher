from typing import Optional, Union, Dict
import random
from einops import rearrange
import numpy as np
import json
import torch
from torch.utils.data import Dataset

from pathlib import Path
from torchvision.transforms import Compose

from .io import load_frenet_features
from .visualize import (
    visualize_multiple_full_embeddings,
)
from .transforms import (
    RandomShuffle,
    RandomLineCrop,
    RandomLineReverse,
    RandomSubsampleCorrespondences,
    ToLightGlueFormat,
    RandomTextMutation,
    EmbeddTexts,
)

torch.seed()


class FrenetDataset(Dataset):
    def __init__(
        self,
        template_features_dir: Path,
        warped_features_dir: Path,
        statistics: Dict,
        split: str,
        min_num_features: int,
        max_samples: Optional[int],
    ):
        self.mean = np.array(statistics["mean"])
        self.std = np.array(statistics["std"])
        self.min_num_features = min_num_features

        template_features_files = list(template_features_dir.glob("*.npz"))
        template_features_files = sorted(template_features_files)

        self.samples = [
            {
                "template_file": template_file,
                "warpedgt_file": warped_features_dir / template_file.name,
            }
            for template_file in template_features_files
        ]

        assert len(self.samples) > 0, f"No samples found for split {split}"

        def num_features(file: Path):
            files = list(np.load(file.as_posix(), mmap_mode="r").files)
            files = set(file.split(".")[0] for file in files)
            return len(files)

        if split == "train":
            self.samples = [
                sample
                for sample in self.samples
                if num_features(sample["template_file"]) >= self.min_num_features
                and num_features(sample["warpedgt_file"]) >= self.min_num_features
            ]

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        self.transforms = []

        if split == "train":
            self.transforms.extend(
                [
                    RandomSubsampleCorrespondences(),
                    ToLightGlueFormat(),
                    RandomLineCrop(),
                    RandomLineReverse(),
                    RandomTextMutation(),
                    EmbeddTexts(),
                ]
            )
        else:
            self.transforms.extend(
                [RandomShuffle(), ToLightGlueFormat(), EmbeddTexts()]
            )

        self.transforms = Compose(self.transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        data = {
            "name": sample["template_file"].stem,
            "raw_data": {
                "template_features": self._load_features(sample["template_file"]),
                "warpedgt_features": self._load_features(sample["warpedgt_file"]),
            },
            "view0": {
                "image_size": torch.Tensor([512, 512]).float(),
            },
            "view1": {
                "image_size": torch.Tensor([512, 512]).float(),
            },
        }

        return self.transforms(data)

    def _load_features(self, feature_file: Path):
        def preprocess_features(features):
            text = str(features["text"]) if features["text"] is not None else None
            visual = (features["visual"].astype("float32") - self.mean) / self.std
            positional = features["positional"].astype("float32")

            line_features = np.concatenate([visual, positional], axis=-1)
            line_features = rearrange(line_features, "h w c -> c h w")
            line_features = torch.from_numpy(line_features).float()

            return {"features": line_features, "text": text}

        def remove_prefix(line_id: str):
            return line_id.split("-", 1)[1]

        return {
            remove_prefix(line_id): preprocess_features(line_features)
            for line_id, line_features in load_frenet_features(feature_file).items()
        }

    def visualize_sample(self, index: int, line_index: int):
        sample = self[index]

        template = sample["descriptors0"][line_index]
        warpedgt = sample["descriptors1"][line_index]

        self.visualize_descriptors(
            template, warpedgt, output_file=f"sample_{index}_line_{line_index}.png"
        )

    def visualize_descriptors(
        self, template: torch.Tensor, warpedgt: torch.Tensor, output_file: str
    ):
        template = rearrange(template, "c h w -> h w c")
        warpedgt = rearrange(warpedgt, "c h w -> h w c")

        template_visual = (template[..., :3] * self.std + self.mean).to(torch.uint8)
        warpedgt_visual = (warpedgt[..., :3] * self.std + self.mean).to(torch.uint8)

        template_positional = template[..., 3:]
        warpedgt_positional = warpedgt[..., 3:]

        visualize_multiple_full_embeddings(
            [
                {
                    "visual": template_visual,
                    "positional": template_positional,
                    "title": "Template embeddings",
                },
                {
                    "visual": warpedgt_visual,
                    "positional": warpedgt_positional,
                    "title": "Warpedgt embeddings",
                },
            ],
            output_file=output_file,
        )


def collate_fn(batch):
    return {
        "name": [sample["name"] for sample in batch],
        "line_ids0": [sample["line_ids0"] for sample in batch],
        "line_ids1": [sample["line_ids1"] for sample in batch],
        "descriptors0": torch.stack([sample["descriptors0"] for sample in batch]),
        "descriptors1": torch.stack([sample["descriptors1"] for sample in batch]),
        "texts_str0": [sample["texts_str0"] for sample in batch],
        "texts_str1": [sample["texts_str1"] for sample in batch],
        "texts0": torch.stack([sample["texts0"] for sample in batch]),
        "texts1": torch.stack([sample["texts1"] for sample in batch]),
        "gt_matches0": torch.stack([sample["gt_matches0"] for sample in batch]),
        "gt_matches1": torch.stack([sample["gt_matches1"] for sample in batch]),
        "gt_assignment": torch.stack([sample["gt_assignment"] for sample in batch]),
        "view0": {
            "image_size": batch[0]["view0"]["image_size"],
        },
        "view1": {
            "image_size": batch[0]["view1"]["image_size"],
        },
    }
