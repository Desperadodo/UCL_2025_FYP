#!/bin/bash
# Cleaned version of CLIP training script
# Removed hardcoded paths and Chinese comments for public repository

conda activate BigModel
cd /path/to/open_clip_train  # Update with actual path
wandb offline

export PYTHONPATH=$PYTHONPATH:/path/to/code  # Update with actual path
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_CACHE="/path/to/cache/huggingface/hub"  # Update with actual path
export TORCH_HOME="/path/to/cache/pretrained/"  # Update with actual path

# Example training commands with placeholder paths
torchrun --nproc_per_node 2 -m main \
    --model customize-clip \
    --dataset-type postgres \
    --train-data /path/to/train_data.pkl  # Update with actual path
    --val-data /path/to/val_data.pkl  # Update with actual path
    --image-folder /path/to/images  # Update with actual path
    --batch-size 2 \
    --epochs 100 \
    --lr 1e-4 \
    --workers 4 \
    --precision amp \
    --name "clip_custom_training_$(date +%Y%m%d_%H%M%S)"

# GigaPath WSI training example
torchrun --nproc_per_node 2 -m main \
    --model customize-clip-wsi-gigapath \
    --dataset-type unpuzzle \
    --train-data /path/to/unpuzzle_train.csv  # Update with actual path
    --val-data /path/to/unpuzzle_val.csv  # Update with actual path
    --seed 42 \
    --patches_per_wsi 2048 \
    --batch-size 4 \
    --workers 8 \
    --epochs 50 \
    --lr 5e-5 \
    --warmup 250 \
    --precision amp \
    --name "clip_wsi_gigapath_training_$(date +%Y%m%d_%H%M%S)"

# Multi-positive loss training example
torchrun --nproc_per_node 4 -m main \
    --model customize-clip-wsi-gigapath \
    --dataset-type unpuzzle \
    --train-data /path/to/unpuzzle_train.csv  # Update with actual path
    --val-data /path/to/unpuzzle_val.csv  # Update with actual path
    --seed 42 \
    --multi-positive-loss \
    --patches_per_wsi 2048 \
    --batch-size 8 \
    --accum-freq 2 \
    --workers 8 \
    --epochs 50 \
    --lr 5e-5 \
    --warmup 250 \
    --precision amp \
    --name "clip_wsi_gigapath_multi_positive_$(date +%Y%m%d_%H%M%S)"