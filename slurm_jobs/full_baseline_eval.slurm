#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J full_base_eval
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH -C "GPU_SKU:A100"

# Full Test1 generation + WVMOS for the frozen baseline checkpoint.
# Speaker sim / WER / IEMOCAP run afterwards off the same wavs.
cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate emoldm
export PYTHONPATH="$PWD"

echo "=== generate all 16903 x 7 ==="
srun --ntasks=1 python eval/benchmark_hifigan_baseline.py \
    --checkpoint "$CKPT" --save_root "$SAVE_ROOT"

echo "=== WVMOS ==="
srun --ntasks=1 python eval/vmos_calc_hifigan_baseline.py --wav_root "$SAVE_ROOT/wav"
