#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J synth_speaker
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-seed%a-%j.out
#SBATCH --error=logs/%x-seed%a-%j.err
# If you know A100 is in gpu-batch and you're allowed:
#SBATCH -C "GPU_SKU:A100"
#SBATCH --array=1234

# TargetSEC base synthesizer, matched to the HiFiGAN baseline: MSP-Podcast Train
# for gradients, Development for validation/model selection, Test1 untouched.
#
# Single seed:  sbatch slurm_jobs/training.slurm
# Three seeds:  sbatch --array=1234-1236 slurm_jobs/training.slurm
# Resume:       sbatch --array=1234 --export=ALL,RESUME=path/to.ckpt slurm_jobs/training.slurm

cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"

# env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate emoldm

SEED="${SLURM_ARRAY_TASK_ID:-1234}"

EXTRA=""
if [ -n "$RESUME" ]; then EXTRA="--resume $RESUME"; fi
srun --ntasks=1 python train_synthesizer.py --seed "$SEED" \
    --ser_loss "${SER_LOSS:-detached}" $EXTRA
