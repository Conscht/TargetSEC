#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J hifigan_baseline
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-seed%a-%j.out
#SBATCH --error=logs/%x-seed%a-%j.err
#SBATCH -C "GPU_SKU:A100"
#SBATCH --array=1234-1236

# Three seeds as a job array, one A100 each:
#   sbatch slurm_jobs/training_hifigan_baseline.slurm
# Single seed:
#   sbatch --array=1234 slurm_jobs/training_hifigan_baseline.slurm
# Detached L_SER (no closed loop with the scorer):
#   sbatch --array=1234 --export=ALL,SER_LOSS=detached slurm_jobs/training_hifigan_baseline.slurm
# Resume:
#   sbatch --array=1234 --export=ALL,RESUME=path/to.ckpt slurm_jobs/training_hifigan_baseline.slurm

cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate emoldm
export PYTHONPATH="$PWD"

SEED="${SLURM_ARRAY_TASK_ID:-1234}"

# Gate on the loss/gradient checks -- cheap, and exactly what would have caught
# 45*L_recon and L_SER being dead constants before 400 epochs of training.
echo "=== Pre-flight checks (seed $SEED) ==="
srun --ntasks=1 python train/verify_hifigan_losses.py || exit 1

echo "=== Training (seed $SEED) ==="
EXTRA=""
if [ -n "$RESUME" ]; then EXTRA="--resume $RESUME"; fi
srun --ntasks=1 python train/train_hifigan_baseline.py --seed "$SEED" \
    --ser_loss "${SER_LOSS:-differentiable}" $EXTRA
