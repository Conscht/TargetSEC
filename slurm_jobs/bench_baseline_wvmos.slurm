#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J bench_wvmos
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Blackwell RTX 6000 (sm_120) is unsupported by this torch build -- pin A100.
#SBATCH -C "GPU_SKU:A100"

# Random-sample Test1 eval WITH WAVs, then WVMOS. Pass CKPT=, SAVE_ROOT=, N=.
cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate emoldm

srun --ntasks=1 python benchmark_hifigan_baseline.py \
    --checkpoint "$CKPT" --save_root "$SAVE_ROOT" --sample_n "${N:-2000}" --sample_seed 0
srun --ntasks=1 python vmos_calc_hifigan_baseline.py --wav_root "$SAVE_ROOT/wav"
