from colorama import Fore
import tqdm
import gdown
import torch
import yaml
import numpy as np

from typing import Optional
from PIL import Image
from typing import Tuple, Dict
from einops import rearrange
from pathlib import Path
from functools import lru_cache

from inv3d_util.image import scale_image
from inv3d_util.load import save_image, load_image, save_npz
from inv3d_util.mapping import apply_map_torch
from inv3d_util.parallel import process_tasks
from inv3d_util.mapping import invert_map, apply_map, scale_map

from .models import model_factory
from ..unwarp_correspondence.preunwarp_outline import preunwarp_lines as unwarp_lines
from ..unwarp_correspondence.utils import NoStdStreams


def inference(
    inputs: Dict,
    outputs: Dict,
    model_ckpt: Path,
    num_workers: int,
    gpu: int,
    visualize: bool,
    save_bm: bool = False,
):
    print(Fore.GREEN + "[STAGE] Unwarping documents with legacy model" + Fore.RESET)

    tasks = []

    for input_image_file in inputs["warped_images"].glob("*.jpg"):

        input_template_file = inputs["template_images"] / input_image_file.name
        image_image_hq_file = inputs["warped_images_HQ"] / input_image_file.name
        input_template_hq_file = inputs["template_images_HQ"] / input_image_file.name
        output_image_file = outputs["warped_images"] / input_image_file.name
        output_image_hq_file = outputs["warped_images_HQ"] / input_image_file.name

        if (
            "warped_lines" in inputs
            and "warped_lines" in outputs
            and inputs["warped_lines"]
            and outputs["warped_lines"]
        ):
            input_warped_lines_file = inputs[
                "warped_lines"
            ] / input_image_file.name.replace(".jpg", ".json")
            output_warped_lines_file = outputs[
                "warped_lines"
            ] / input_image_file.name.replace(".jpg", ".json")
        else:
            input_warped_lines_file = None
            output_warped_lines_file = None

        tasks.append(
            {
                "input_image_file": input_image_file,
                "input_template_file": input_template_file,
                "input_image_hq_file": image_image_hq_file,
                "input_template_hq_file": input_template_hq_file,
                "input_warped_lines_file": input_warped_lines_file,
                "output_image_file": output_image_file,
                "output_image_hq_file": output_image_hq_file,
                "output_warped_lines_file": output_warped_lines_file,
                "model_ckpt": model_ckpt,
                "gpu": gpu,
                "visualize": visualize,
            }
        )

        if save_bm:
            output_bm_file = outputs["backward_maps"] / input_image_file.name.replace(
                ".jpg", ".npz"
            )
        else:
            output_bm_file = None
        tasks[-1]["output_bm_file"] = output_bm_file

    # for task in tqdm.tqdm(tasks, desc=f"Inference '{model_ckpt.stem}'"):
    # inference_sample_task(task)

    process_tasks(
        inference_sample_task,
        tasks,
        num_workers,
        use_indexes=True,
        desc="Unwarping documents",
    )

    prepare_model_cached.cache_clear()  # clear cache to avoid memory leak


def inference_sample_task(task: Dict):
    inference_sample(**task)


def inference_sample(
    input_image_file: Path,
    input_template_file: Path,
    input_image_hq_file: Path,
    input_warped_lines_file: Optional[Path],
    input_template_hq_file: Path,
    output_image_file: Path,
    output_image_hq_file: Path,
    output_warped_lines_file: Optional[Path],
    output_bm_file: Optional[Path],
    model_ckpt: Path,
    gpu: int,
    visualize: bool,
):
    image = load_image(input_image_file)
    image_hq = load_image(input_image_hq_file)
    template = load_image(input_template_file)
    template_hq = load_image(input_template_hq_file)

    # prepare model
    model = prepare_model_cached(model_ckpt, gpu)
    model_name = model_ckpt.stem

    # prepare image
    image_original = np.array(image)[..., :3]
    image_hq_original = np.array(image_hq)[..., :3]

    image = scale_image(image_original, resolution=model.dataset_options["resolution"])
    image = rearrange(image, "h w c -> () c h w")
    image = image.astype("float32") / 255
    image = torch.from_numpy(image)
    image = image.to(f"cuda:{gpu}")

    model_kwargs = {"image": image}

    template_height, template_width, _ = template_hq.shape
    output_shape = (template_height, template_width)

    # prepare template
    if "template" in model_name:
        template_original = np.array(template)[..., :3]
        template = scale_image(
            template_original, resolution=model.dataset_options["resolution"]
        )
        template = rearrange(template, "h w c -> () c h w")
        template = template.astype("float32") / 255
        template = torch.from_numpy(template)
        template = template.to(f"cuda:{gpu}")

        model_kwargs["template"] = template

    # inference model
    out_bm = model(**model_kwargs).detach().cpu()

    # unwarp input
    norm_image = unwarp_image(image_original, out_bm, (512, 512))
    norm_image_hq = unwarp_image(image_hq_original, out_bm, output_shape)

    # save images
    save_image(output_image_file, norm_image, override=True)
    save_image(output_image_hq_file, norm_image_hq, override=True)

    # bm to numpy
    out_bm = rearrange(out_bm, "() c h w -> h w c").numpy()

    # map and save lines
    if input_warped_lines_file and output_warped_lines_file:
        out_uv = invert_map_HQ(out_bm)
        unwarp_lines(
            input_warped_lines_file,
            out_uv,
            output_warped_lines_file,
            visualize,
            output_image_file,
        )

    if output_bm_file:
        save_npz(output_bm_file, out_bm, override=True)


def invert_map_HQ(bm: np.ndarray) -> np.ndarray:
    with NoStdStreams():
        bm_intermediate = scale_map(bm, resolution=1024)
    uv_intermediate = invert_map(bm_intermediate)

    return scale_map(uv_intermediate, 512)


def unwarp_image(
    image: np.ndarray, out_bm: torch.Tensor, output_shape: Tuple[int, int]
):
    image = rearrange(image, "h w c -> () c h w")
    image = image.astype("float32") / 255
    image = torch.from_numpy(image)

    norm_image = apply_map_torch(image=image, bm=out_bm, resolution=output_shape)

    norm_image = rearrange(norm_image, "b c h w -> b h w c")
    norm_image = norm_image * 255
    norm_image = norm_image.squeeze(0).detach().cpu().numpy().astype("uint8")

    return norm_image


@lru_cache(maxsize=None)
def prepare_model_cached(model_ckpt: Path, gpu: int):
    model_name = model_ckpt.stem

    model = model_factory.load_from_checkpoint(
        model_name.split("@")[0], model_ckpt, gpu
    )
    model.eval()

    return model
