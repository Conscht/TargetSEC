#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J plot_style_plasma
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH -C "GPU_SKU:A100"

cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate emoldm

pip install -q umap-learn

srun python plot_style_shift_plasma.py
