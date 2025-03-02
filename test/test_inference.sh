#!/bin/bash

gpu=0
models=("identity" "dewarpnet@inv3d" "geotr@inv3d" "geotr_template@inv3d" "geotr_template_large@inv3d" "docmatcher@inv3d")
datasets=("example" "inv3d_real")
limit_samples=10

cd /workspaces/doc-matcher/

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do

        # print the current model and dataset
        echo "---------------------------------------------------------"
        echo "Running inference for model $model with dataset $dataset"
        echo "---------------------------------------------------------"

        # run the inference script
        python inference.py --model "$model" --dataset "$dataset" --gpu "$gpu" --limit_samples "$limit_samples"

        # check if the command was successful, otherwise stop the script
        if [ $? -ne 0 ]; then
            echo "Command failed for model $model with dataset $dataset"
            exit 1
        fi

        # check if the output folder exits, otherwise stop the script
        if [ ! -d "output/$dataset-$model" ]; then
            echo "Output folder not found for model $model with dataset $dataset"
            exit 1
        fi

    done
done