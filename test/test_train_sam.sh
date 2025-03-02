#!/bin/bash

gpu=0
limit_samples=2

cd /workspaces/doc-matcher/


# print the current model and dataset
echo "---------------------------------------------------------"
echo "Running training for model sam"
echo "---------------------------------------------------------"

# run the training script
python train.py --model-part sam --gpu "$gpu" --limit_samples "$limit_samples"
