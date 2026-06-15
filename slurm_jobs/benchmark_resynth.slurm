#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J benchmark_resynth
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH -C "GPU_SKU:A100"

cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate emoldm

srun python benchmark_resynth.py && python vmos_calc_resynth.py
