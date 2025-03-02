import cv2

cv2.setNumThreads(0)

import argparse
import sys
from pathlib import Path

project_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_dir / "src"))

from inv3d_evaluation.score import score


def list_inference_output_dirs():
    return [x.stem for x in (project_dir / "output").iterdir() if x.is_dir()]


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluation script")

    parser.add_argument(
        "--run",
        type=str,
        choices=list_inference_output_dirs(),
        required=True,
        help="Select the data for evaluation. The data must be in the output directory.",
    )

    return parser


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    score(data_dir=project_dir / "output" / args.run)
