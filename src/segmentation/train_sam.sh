#!/bin/bash

# Accept GPU as input
GPU=$1

# Change directory
cd /workspaces/doc-matcher/src/segmentation/finetune_anything

# echo pwd
echo "Current directory: $(pwd)"

# Use the provided GPU
CUDA_VISIBLE_DEVICES=$GPU python train.py --task_name semantic_seg_inv3d