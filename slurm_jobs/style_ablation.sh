#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J MLP_Ablation
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gpus=a100:1
# Optional: Keep A100 if you want, but this runs fast on any GPU

# Navigate to your project folder
cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate emoldm
export PYTHONPATH="$PWD"

# Run the MLP Ablation script
# NOTE: Make sure you set USE_SPEAKER_COND = True/False inside this file before submitting!
srun python train/train_mlp_ablation.py