#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J LDM_Ablation
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ------------------------------------------------------------------
# CRITICAL FIX: Request A100 (Blackwell RTX6000 causes crash)
# ------------------------------------------------------------------
#SBATCH --gpus=a100:1

cd "/sc/home/constantin.auga/New folder/Code/EmoConv-LDM"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate emoldm

# Run your new diffusion script
srun --cpu-bind=none python train2_diffusion.py