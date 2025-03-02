from colorama import Fore
from .prepare_lines.prepare_lines import prepare_lines
from .prepare_coco.prepare_coco import prepare_coco


def prepare_training(
    base_dir: str,
    inputs: dict,
    outputs: dict,
    data_dir: str,
    split: str,
    coco_params: dict,
    num_workers: int,
    visualize: bool = False,
):
    print(Fore.GREEN + f"[STAGE] Preparing Lineformer training ({split})" + Fore.RESET)

    prepare_lines(
        inputs=inputs,
        outputs=outputs,
        data_dir=data_dir,
        split=split,
        resolution=coco_params["resolution"],
        with_borders=coco_params["with_borders"],
        num_workers=num_workers,
        visualize=visualize,
    )

    prepare_coco(
        base_dir=base_dir,
        input_lines_dir=outputs["warped_lines_GT"],
        input_images_dir=inputs["warped_images_HQ"],
        input_masks_dir=inputs["masks_HQ_GT"],
        split=split,
        resolution=coco_params["resolution"],
        line_width=coco_params["line_width"],
        limit_testval=coco_params["limit_testval"],
        with_mask=coco_params["with_mask"],
        num_workers=num_workers,
    )
