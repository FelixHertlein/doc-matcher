from pathlib import Path

from colorama import Fore
from inv3d_util.load import load_image

from .score_core.score import score_all


def score(data_dir: Path):
    # list all dirs in data_dir
    samples = [x for x in data_dir.iterdir() if x.is_dir()]

    # load all samples
    data = [load_sample(sample) for sample in samples]
    data = [x for x in data if x is not None]

    if len(data) == 0:
        print(Fore.RED + "Error: No samples found. Exiting!" + Fore.RESET)
        return

    print(Fore.GREEN + f"Loaded {len(data)} samples!" + Fore.RESET)

    df = score_all(data)
    df.to_csv(data_dir / "results_details.csv", index=False)

    df_summary = df.drop(columns=["sample"]).mean()
    df_summary.to_csv(data_dir / "results_summary.csv", header=False)

    print(Fore.GREEN)
    print("----------------------------------------------------------------")
    print("------------------------   Results   ---------------------------")
    print("----------------------------------------------------------------")
    print(df_summary)
    print(Fore.RESET)

    # print("Matlab version: ", df.iloc[0].version)

    print("Success!")


def load_sample(sample: Path):
    true_image_files = list(sample.glob("true_image.*"))
    norm_image_files = list(sample.glob("norm_image.*"))

    if len(true_image_files) != 1:
        print(
            Fore.YELLOW
            + f"WARNING: Could not find true_image in {sample}. Skipping!"
            + Fore.RESET
        )
        return None

    [true_image_file] = true_image_files
    [norm_image_file] = norm_image_files

    return {
        "true_image_file": true_image_file,
        "norm_image_file": norm_image_file,
        "true_image": load_image(true_image_file),
        "norm_image": load_image(norm_image_file),
        "text_evaluation": True,
        "output_dir": sample,
        "results": {"sample": sample.stem},
    }
