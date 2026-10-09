#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J bench_mlp
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/%x-spk%a-%j.out
#SBATCH --error=logs/%x-spk%a-%j.err
#SBATCH -C "GPU_SKU:A100"
#SBATCH --array=0-1

cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate emoldm
export PYTHONPATH="$PWD"

SPK="$SLURM_ARRAY_TASK_ID"
if [ "$SPK" = "0" ]; then MLP="$MLP_NOSPK"; else MLP="$MLP_WSPK"; fi
SR="eval_outputs/ABL_mlp_spk${SPK}"

srun --ntasks=1 python -u eval/benchmark_mlp.py --synth "$SYNTH" --mlp "$MLP" \
    --use_speaker "$SPK" --save_root "$SR"
srun --ntasks=1 python eval/vmos_calc_hifigan_baseline.py --wav_root "$SR/wav"
