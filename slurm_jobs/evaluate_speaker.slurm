#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J evaluate_speaker
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH -C "GPU_SKU:A100"

set -euo pipefail

cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"

mkdir -p logs

# env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate emoldm

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job running on: $(hostname)"
which python
python -V
nvidia-smi || true

srun python evaluate_speaker.py
