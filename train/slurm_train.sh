#!/bin/bash
#SBATCH --job-name=limo_train
#SBATCH --partition=eng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/tianruny/LIMO/logs/train_%j.out
#SBATCH --error=/home/tianruny/LIMO/logs/train_%j.err

# Usage:
#   sbatch slurm_train.sh <train_data> <output_dir>
# Example:
#   sbatch slurm_train.sh \
#     /home/tianruny/LIMO/data/Q2/qwen2.5-math-7b/rsr/chi2_tau0.5_B5/train.jsonl \
#     /home/tianruny/LIMO/results/checkpoints/qwen2.5-math-7b/rsr/chi2_tau0.5_B5

TRAIN_DATA=$1
OUTPUT_DIR=$2

mkdir -p /home/tianruny/LIMO/logs
mkdir -p "$OUTPUT_DIR"

source /home/tianruny/miniconda3/etc/profile.d/conda.sh
conda activate LIMO

torchrun --nproc_per_node=4 /home/tianruny/LIMO/train/train_lora.py \
    --train_data  "$TRAIN_DATA" \
    --model_path  /home/tianruny/LIMO/models/students/qwen2.5-math-7b \
    --output_dir  "$OUTPUT_DIR" \
    --batch_size  2 \
    --grad_accum  8
