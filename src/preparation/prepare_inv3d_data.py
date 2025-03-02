import numpy as np
from pathlib import Path
from typing import Optional
import tqdm
from PIL import Image
import skimage.transform as st
import shutil

from inv3d_util.load import load_image, save_image, load_array, save_npz
from inv3d_util.image import scale_image, tight_crop_image
from inv3d_util.mapping import load_uv_map, scale_map, tight_crop_map


def prepare_inv3d_data(
    data_dir: Path,
    resolution: int,
    split: str,
    outputs: dict,
    limit_samples: Optional[int],
):
    if split == "real":
        prepare_inv3d_split_real(data_dir, resolution, outputs, limit_samples)
    else:
        prepare_inv3d_split(data_dir, resolution, split, outputs, limit_samples)


def prepare_inv3d_split(
    data_dir: Path,
    resolution: int,
    split: str,
    outputs: dict,
    limit_samples: Optional[int],
):
    split_dir = data_dir / split

    samples = [x for x in split_dir.iterdir() if x.is_dir()]
    samples.sort()

    if limit_samples:
        samples = samples[:limit_samples]

    for sample in tqdm.tqdm(samples, desc=f"Processing {split} split"):
        prepare_image(
            input_image_file=sample / "flat_template.png",
            uv_file=None,
            output_image_HQ_file=outputs["template_images_HQ"] / f"{sample.name}.jpg",
            output_image_file=outputs["template_images"] / f"{sample.name}.jpg",
            resolution=resolution,
        )

        prepare_image(
            input_image_file=sample / "warped_document.png",
            uv_file=sample / "warped_UV.npz",
            output_image_HQ_file=outputs["warped_images_HQ"] / f"{sample.name}.jpg",
            output_image_file=outputs["warped_images"] / f"{sample.name}.jpg",
            resolution=resolution,
        )

        prepare_maps(
            bm_file=sample / "warped_BM.npz",
            uv_file=sample / "warped_UV.npz",
            output_uv_file=outputs["forward_maps_GT"] / f"{sample.name}.npz",
            output_bm_file=outputs["backward_maps_GT"] / f"{sample.name}.npz",
            output_mask_file=outputs["masks_GT"] / f"{sample.name}.png",
            output_mask_hq_file=outputs["masks_HQ_GT"] / f"{sample.name}.png",
            resolution=resolution,
        )

        # preapre true image
        true_image = load_image(sample / "flat_document.png")
        save_image(
            outputs["true_images_HQ"] / f"{sample.name}.jpg", true_image, override=True
        )


def prepare_inv3d_split_real(
    data_dir: Path,
    resolution: int,
    outputs: dict,
    limit_samples: Optional[int],
):
    split_dir = data_dir / "real"

    samples = list(split_dir.rglob("warped_document_*.jpg"))
    samples.sort()
    assert len(samples) == 360, "Expected 360 samples"

    if limit_samples:
        samples = samples[:limit_samples]

    for sample in tqdm.tqdm(samples, desc=f"Processing real split", total=len(samples)):
        sample_name = f"{sample.parent.name}_{sample.stem[len('warped_document_'):]}"

        prepare_image(
            input_image_file=sample.parent / "flat_template.png",
            uv_file=None,
            output_image_HQ_file=outputs["template_images_HQ"] / f"{sample_name}.jpg",
            output_image_file=outputs["template_images"] / f"{sample_name}.jpg",
            resolution=resolution,
        )

        prepare_image(
            input_image_file=sample,
            uv_file=None,
            output_image_HQ_file=outputs["warped_images_HQ"] / f"{sample_name}.jpg",
            output_image_file=outputs["warped_images"] / f"{sample_name}.jpg",
            resolution=resolution,
        )

        # preapre true image
        true_image = load_image(sample.parent / "flat_document.png")
        save_image(
            outputs["true_images_HQ"] / f"{sample_name}.jpg", true_image, override=True
        )


def prepare_image(
    input_image_file: Path,
    uv_file: Optional[Path],
    output_image_HQ_file: Path,
    output_image_file: Path,
    resolution: int,
):
    image = load_image(input_image_file)

    if uv_file:
        mask = load_array(uv_file)[..., :1]
        image = tight_crop_image(image, mask.squeeze())

    image_scaled = scale_image(image, resolution=resolution)

    save_image(
        output_image_HQ_file,
        image,
        override=True,
    )

    save_image(output_image_file, image_scaled, override=True)


def prepare_maps(
    bm_file: Path,
    uv_file: Path,
    output_uv_file: Path,
    output_bm_file: Path,
    output_mask_file: Path,
    output_mask_hq_file: Path,
    resolution: int,
):
    uv, mask = load_uv_map(uv_file, return_mask=True)
    bm = load_array(bm_file)

    bm_crop = tight_crop_map(bm)
    uv_crop, mask_crop = tight_crop_image(uv, mask.squeeze(), return_mask=True)

    bm_scaled = scale_map(bm_crop, resolution)
    uv_scaled = scale_map(uv_crop, resolution)

    mask_scaled = st.resize(
        mask_crop,
        (resolution, resolution),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )

    save_npz(output_uv_file, uv_scaled, override=True)
    save_npz(output_bm_file, bm_scaled, override=True)

    # export mask as png
    palette = [255, 255, 255, 255, 105, 97] + [0] * 254 * 3
    output_mask = Image.fromarray(mask_scaled.astype(np.uint8))
    output_mask = output_mask.convert("P")
    output_mask.putpalette(palette)
    output_mask.save(output_mask_file)

    output_mask = Image.fromarray(mask_crop.astype(np.uint8))
    output_mask = output_mask.convert("P")
    output_mask.putpalette(palette)
    output_mask.save(output_mask_hq_file)
