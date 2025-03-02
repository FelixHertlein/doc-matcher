from pathlib import Path
from colorama import Fore
from tqdm import tqdm
import shutil

from inv3d_util.load import load_image, save_image
from inv3d_util.image import scale_image


def inference(inputs: dict, outputs: dict, data_dir: Path, stage_name: str, split: str):
    print(Fore.GREEN + "[STAGE] Collecting results" + Fore.RESET)

    output_dir = outputs["results"]
    unwarped_image_files = list(inputs["warped_images_HQ"].glob("*.jpg"))

    preparation_dir_name = stage_name.split("_")[0]
    warped_orig_dir = (
        data_dir / "preparation" / preparation_dir_name / "warped_images_HQ" / split
    )

    for unwarped_image_file in tqdm(unwarped_image_files, desc="Collecting results"):
        output_sample_dir = output_dir / unwarped_image_file.stem
        output_sample_dir.mkdir(parents=True, exist_ok=True)

        warped_orig_file = warped_orig_dir / unwarped_image_file.name
        template_file = inputs["template_images_HQ"] / f"{unwarped_image_file.stem}.jpg"

        unwarped_image = load_image(unwarped_image_file)
        template = load_image(template_file)

        h, w, c = template.shape

        unwarped_image = scale_image(unwarped_image, resolution=(h, w))
        save_image(output_sample_dir / "norm_image.jpg", unwarped_image, override=True)

        shutil.copyfile(warped_orig_file, output_sample_dir / "orig_image.jpg")

        if "true_images_HQ" in inputs:
            true_image_file = (
                inputs["true_images_HQ"] / f"{unwarped_image_file.stem}.jpg"
            )
            shutil.copyfile(
                true_image_file,
                output_sample_dir / "true_image.jpg",
            )
