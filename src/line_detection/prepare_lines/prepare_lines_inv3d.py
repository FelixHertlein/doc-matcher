from tqdm import tqdm
from pathlib import Path
from typing import Dict
from inv3d_util.parallel import process_tasks

from .find_template_lines import find_template_lines
from .export_template_lines import export_template_lines
from .warp_template_lines import warp_template_lines

from ..visualization.visualize_lines import visualize_lines


def prepare_lines_inv3d(
    inputs: dict,
    outputs: dict,
    data_dir: str,
    split: str,
    resolution: int,
    with_borders: bool,
    num_workers: int,
    visualize: bool = False,
):
    if split == "real":
        samples = list_samples_inv3d_real(data_dir)
    else:
        samples = list_samples_inv3d(data_dir, split)

    # filtered used samples (in case of a reduced inv3d_real dataset)
    samples = {
        k: v
        for k, v in samples.items()
        if (inputs["warped_images"] / f"{k}.jpg").is_file()
    }

    assert len(samples) > 0, "No samples found"

    tasks = []
    for sample_name, sample in samples.items():
        tasks.append(
            {
                "inputs": inputs,
                "outputs": outputs,
                "sample_name": sample_name,
                "sample": sample,
                "resolution": resolution,
                "with_borders": with_borders,
                "visualize": visualize,
            }
        )

    process_tasks(
        prepare_lines_sample_task,
        tasks,
        num_workers=num_workers,
        use_indexes=True,
        desc="Prepare lines",
    )


def prepare_lines_sample_task(task: dict):
    prepare_lines_sample(**task)


def prepare_lines_sample(
    inputs, outputs, sample_name, sample, resolution, with_borders, visualize
):
    template_file = sample.parent / "flat_template.png"
    BM_file = sample.parent / "warped_BM.npz"
    template_lines_file = outputs["template_lines"] / f"{sample_name}.json"

    if template_lines_file.exists():
        return  # prevent reprocessing to avoid line id mismatch

    template_lines = find_template_lines(
        template_file=template_file,
        information_delta_file=sample.parent / "flat_information_delta.png",
        words_file=sample.parent / "ground_truth_words.json",
        rescale_size=resolution,
        with_borders=with_borders,
    )

    export_template_lines(
        output_file=template_lines_file,
        lines=template_lines,
        image_path=inputs["template_images"] / f"{sample_name}.jpg",
        is_template=True,
    )

    if visualize:
        visualize_lines(template_lines_file)

    if BM_file.is_file():
        warpedgt_lines = warp_template_lines(BM_file, template_lines, resolution)

        warped_lines_GT_file = outputs["warped_lines_GT"] / f"{sample_name}.json"
        export_template_lines(
            output_file=warped_lines_GT_file,
            lines=warpedgt_lines,
            image_path=inputs["warped_images"] / f"{sample_name}.jpg",
            is_template=False,
        )

        if visualize:
            visualize_lines(warped_lines_GT_file)


def list_samples_inv3d(data_dir: str, split: str) -> Dict[str, Path]:
    split_dir = Path(data_dir) / split

    samples = list(split_dir.rglob("warped_document.png"))
    samples.sort()

    return {sample.parent.stem: sample for sample in samples}


def list_samples_inv3d_real(data_dir: str) -> Dict[str, Path]:
    split_dir = Path(data_dir) / "real"

    samples = list(split_dir.rglob("warped_document_*.jpg"))
    samples.sort()
    assert len(samples) == 360, f"Expected 360 samples, found {len(samples)} samples"

    def extract_name(sample):
        return f"{sample.parent.name}_{sample.stem[len('warped_document_'):]}"

    return {extract_name(sample): sample for sample in samples}
