#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J wvmos_hifigan
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate emoldm

CKPT="checkpoints_hifigan_baseline_annotated/hifigan_baseline_annotated-06-24_12-09-24-epoch=283-val_loss=19.56.ckpt"
SAVE_ROOT="eval_outputs/hifigan_epoch283"

echo "=== Generating wavs ==="
srun --ntasks=1 python benchmark_hifigan_baseline.py --checkpoint "$CKPT" --save_root "$SAVE_ROOT"

echo "=== Running WVMOS ==="
srun --ntasks=1 python vmos_calc_hifigan_baseline.py --wav_root "$SAVE_ROOT/wav"
