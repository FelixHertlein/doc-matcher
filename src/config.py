import yaml
from pathlib import Path
from copy import deepcopy


def parse_config(
    config_file: str, pipeline: str, split: str, export_parsed_config: bool = False
) -> None:
    config_file = Path(config_file)
    stages = yaml.safe_load(config_file.read_text())

    # resolve input template
    stages["preparation"] = resolve_input_template(stages["preparation"])

    instance2category = {}
    instance2details = {}

    for category_name, category_details in stages.items():
        for instance_name, instance_details in category_details.items():
            assert (
                instance_name not in instance2category
            ), f"Instance '{instance_name}' is duplicated!"
            assert (
                instance_name not in instance2details
            ), f"Instance '{instance_name}' is duplicated!"

            instance2category[instance_name] = category_name
            instance2details[instance_name] = instance_details

    parsed_pipeline = []
    assets = {}

    for instance_name in pipeline.split("_"):
        stage_category = instance2category[instance_name]
        stage_details = deepcopy(instance2details[instance_name])

        stage_name = (
            parsed_pipeline[-1]["stage_name"] + "_" + instance_name
            if len(parsed_pipeline) >= 1
            else instance_name
        )

        stage_inputs = deepcopy(assets)
        stage_outputs = {}

        for create_asset in stage_details.get("creates", []):
            new_asset = (
                Path(stage_category) / stage_name / create_asset / split
            ).as_posix()
            assets[create_asset] = new_asset
            stage_outputs[create_asset] = new_asset

        for update_asset in stage_details.get("updates", []):
            if update_asset in assets:
                updated_asset = (
                    Path(stage_category) / stage_name / update_asset / split
                ).as_posix()
                assets[update_asset] = updated_asset
                stage_outputs[update_asset] = updated_asset

        for remove_asset in stage_details.get("removes", []):
            if remove_asset in assets:
                del assets[remove_asset]

        if "creates" in stage_details:
            del stage_details["creates"]

        if "updates" in stage_details:
            del stage_details["updates"]

        if "removes" in stage_details:
            del stage_details["removes"]

        stage_details["stage_name"] = stage_name
        stage_details["stage_instance"] = instance_name
        stage_details["stage_category"] = stage_category
        stage_details["inputs"] = deepcopy(stage_inputs)
        stage_details["outputs"] = deepcopy(stage_outputs)
        stage_details["split"] = split
        stage_details["base_dir"] = (Path(stage_category) / stage_name).as_posix()

        if len(parsed_pipeline) >= 1:
            stage_details["data_dir"] = parsed_pipeline[-1]["data_dir"]

        parsed_pipeline.append(stage_details)

    parsed_pipeline = {
        stage_info["stage_name"]: {
            key: value for key, value in stage_info.items() if key != "stage_name"
        }
        for stage_info in parsed_pipeline
    }

    if export_parsed_config:
        output_file = config_file.parent / f"{config_file.stem}_parsed.yaml"
        output_file.write_text(yaml.dump(parsed_pipeline, sort_keys=False))

    return parsed_pipeline


def resolve_input_template(preparation_stage):
    input_dir = Path("input/custom_datasets")
    input_data_dirs = [f for f in input_dir.iterdir() if f.is_dir()]

    new_instance_names = {
        input_data_dir.stem.replace(" ", "").replace("_", ""): input_data_dir
        for input_data_dir in input_data_dirs
    }
    if (
        "inv3d" in new_instance_names
        or "inv3d_real" in new_instance_names
        or len(new_instance_names) != len(input_data_dirs)
    ):
        raise ValueError(
            "Input data directories have duplicated names! Avoid spaces and underscores in the names."
        )

    template = preparation_stage["GENERIC_DATA_TEMPLATE"]

    for new_instance_name, input_data_dir in new_instance_names.items():
        new_instance = deepcopy(template)
        new_instance["data_dir"] = input_data_dir.as_posix()

        preparation_stage[new_instance_name] = new_instance

    del preparation_stage["GENERIC_DATA_TEMPLATE"]

    return preparation_stage
