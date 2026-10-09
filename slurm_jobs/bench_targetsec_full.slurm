#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J tsec_full
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Blackwell RTX 6000 (sm_120) is unsupported by this torch build -- pin A100.
#SBATCH -C "GPU_SKU:A100"

# TargetSEC full Test1 benchmark: 16903 x 7 = 118321 conversions, then WVMOS.
# ~100 diffusion steps per utterance, so this is far slower than the baseline's
# single forward pass.
cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate emoldm
export PYTHONPATH="$PWD"

srun --ntasks=1 python -u eval/benchmark_finetune.py --synth "$SY" --ldm "$LD" \
    --save_root "$SAVE_ROOT"
srun --ntasks=1 python eval/vmos_calc_hifigan_baseline.py --wav_root "$SAVE_ROOT/wav"
