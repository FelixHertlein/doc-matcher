import tqdm
import pylcs
from typing import Dict
import argparse
import json
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path

from ..feature_extractor.dataset import FrenetDataset, collate_fn
from .visualize import visualize_matching_inner

from .lit_line_lightglue import LitLineLightglue


@torch.no_grad()
def inference_lightglue(
    inputs: Dict,
    outputs: Dict,
    template_feature_dir: Path,
    warped_feature_dir: Path,
    model_checkpoint: str,
    statistics: Dict,
    split: str,
    min_num_features: int,
    gpu: int,
    num_workers: int,
    visualize: bool,
):

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"

    model = LitLineLightglue.load_from_checkpoint(
        model_checkpoint,
        map_location=torch.device(device),
    )
    model.eval()

    # prepare data
    dataset = FrenetDataset(
        template_features_dir=template_feature_dir,
        warped_features_dir=warped_feature_dir,
        statistics=statistics,
        split=split,
        min_num_features=min_num_features,
        max_samples=None,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
    )

    for sample in tqdm.tqdm(dataloader, desc="Inference"):
        sample = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()
        }
        out = model(sample)

        name = sample["name"][0]
        matches0 = out["matches0"][0]
        log_assignment = out["log_assignment"][0]
        line_ids0 = sample["line_ids0"][0]
        line_ids1 = sample["line_ids1"][0]
        texts0 = sample["texts_str0"][0]
        texts1 = sample["texts_str1"][0]

        line_matches = []
        removed_line_matches = []

        for pos, match_idx in enumerate(matches0.tolist()):
            if match_idx == -1 or log_assignment[pos, match_idx] < -1:
                continue

            match = {
                "template": line_ids0[pos],
                "warped": line_ids1[match_idx],
                "log_assignment": log_assignment[pos, match_idx].item(),
            }

            text0 = texts0[pos]
            text1 = texts1[match_idx]

            # check if both line types are different
            if bool(text0 == "None") != bool(text1 == "None"):
                removed_line_matches.append(match)
                continue

            # check if both textlines share a common substring
            if text0 != "None" and pylcs.lcs_string_length(text0, text1) < 3:
                removed_line_matches.append(match)
                continue

            line_matches.append(match)

        matched_template_lines = [
            m["template"] for m in line_matches + removed_line_matches
        ]
        matched_warped_lines = [
            m["warped"] for m in line_matches + removed_line_matches
        ]

        unmatched_template_lines = list(set(line_ids0) - set(matched_template_lines))
        unmatched_warped_lines = list(set(line_ids1) - set(matched_warped_lines))

        output_data = {
            "line_matches": line_matches,
            "removed_line_matches": removed_line_matches,
            "unmatched_template_lines": unmatched_template_lines,
            "unmatched_warped_lines": unmatched_warped_lines,
        }

        output_file = outputs["matches"] / f"{name}.json"
        output_file.write_text(json.dumps(output_data, indent=4))

        if visualize:
            visualize_matching_inner(
                matches_file=output_file,
                template_image_file=inputs["template_images"] / f"{name}.jpg",
                warped_image_file=inputs["warped_images"] / f"{name}.jpg",
                template_lines_file=inputs["template_lines"] / f"{name}.json",
                warped_lines_file=inputs["warped_lines"] / f"{name}.json",
                output_file=output_file.parent / f"{name}.jpg",
            )
