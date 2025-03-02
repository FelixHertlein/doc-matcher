from tqdm import tqdm
from pathlib import Path
from typing import Dict
from inv3d_util.parallel import process_tasks

from line_detection.prepare_lines.detect_template_text_lines import (
    load_doctr_model_cached,
)

from .export_template_lines import export_template_lines
from .detect_template_lines import detect_template_lines

from ..visualization.visualize_lines import visualize_lines


def prepare_lines_generic(
    inputs: dict,
    outputs: dict,
    data_dir: str,
    split: str,
    resolution: int,
    with_borders: bool,
    num_workers: int,
    visualize: bool = False,
):
    data_dir = Path(data_dir)

    samples = {f.stem: f for f in inputs["warped_images"].glob("*.jpg")}
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
        prepare_lines_generic_sample_task,
        tasks,
        num_workers=num_workers,
        use_indexes=True,
        desc="Prepare lines",
    )

    load_doctr_model_cached.cache_clear()  # clear cache to free memory


def prepare_lines_generic_sample_task(task: dict):
    prepare_lines_generic_sample(**task)


def prepare_lines_generic_sample(
    inputs, outputs, sample_name, sample, resolution, with_borders, visualize
):
    template_file = inputs["template_images_HQ"] / f"{sample_name}.jpg"
    template_lines_file = outputs["template_lines"] / f"{sample_name}.json"

    if template_lines_file.exists():
        return  # prevent reprocessing to avoid line id mismatch

    template_lines = detect_template_lines(
        template_file=template_file,
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
