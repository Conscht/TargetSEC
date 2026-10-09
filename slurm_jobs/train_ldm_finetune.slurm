#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J ldm_finetune
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-13:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH -C "GPU_SKU:A100"

# TargetSEC stage 4. Both paths are required now -- they used to be hardcoded to
# a June, Test1-trained checkpoint.
#   sbatch --export=ALL,FT_CKPT=<stage2.ckpt>,STATS=<style_stats.pt> slurm_jobs/train_ldm_finetune.slurm

cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate emoldm
export PYTHONPATH="$PWD"

srun --ntasks=1 python -u train/train_ldm_finetune.py \
    --finetune_checkpoint "$FT_CKPT" --style_stats "$STATS" \
    --use_speaker "${USE_SPEAKER:-1}" --max_epochs "${MAX_EPOCHS:-400}" \
    ${RESUME:+--resume "$RESUME"}
