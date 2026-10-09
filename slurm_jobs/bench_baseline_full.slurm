#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J bench_base_full
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Blackwell RTX 6000 (sm_120) is not supported by this torch build -- pin A100.
#SBATCH -C "GPU_SKU:A100"

# Full Test1 arousal evaluation (all 16903 utts x 7 targets), no WAVs.
# Pass CKPT= and SAVE_ROOT= via --export.

cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate emoldm
export PYTHONPATH="$PWD"

srun --ntasks=1 python eval/benchmark_hifigan_baseline.py \
    --checkpoint "$CKPT" --save_root "$SAVE_ROOT" --no_wav
