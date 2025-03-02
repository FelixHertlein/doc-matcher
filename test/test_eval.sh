#!/bin/bash

gpu=0
model="identity"
dataset="inv3d_real"
limit_samples=1

cd /workspaces/doc-matcher/

# print the model and dataset
echo "---------------------------------------------------------"
echo "Running inference for model $model with dataset $dataset" for the evaluation test
echo "---------------------------------------------------------"

# run the inference script
python inference.py --model "$model" --dataset "$dataset" --gpu "$gpu" --limit_samples "$limit_samples"

# print the evaluation info
echo "---------------------------------------------------------"
echo "Running evaluation for model $model with dataset $dataset"
echo "---------------------------------------------------------"

# run the evaluation script
python eval.py --run "$dataset-$model"
