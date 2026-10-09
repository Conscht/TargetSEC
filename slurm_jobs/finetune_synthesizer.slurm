#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J synth_finetune
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH -C "GPU_SKU:A100"

# TargetSEC stage 2: unfreeze the style encoder, drop LRs.
#   sbatch --export=ALL,SYNTH_CKPT=<stage1.ckpt>,MAX_EPOCHS=20 slurm_jobs/finetune_synthesizer.slurm
# Resume an interrupted stage 2:
#   sbatch --export=ALL,RESUME=<finetune.ckpt> slurm_jobs/finetune_synthesizer.slurm

cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate emoldm
export PYTHONPATH="$PWD"

if [ -n "$RESUME" ]; then
  srun --ntasks=1 python train/train_synthesizer_finetune.py --resume "$RESUME" \
       --max_epochs "${MAX_EPOCHS:-20}"
else
  srun --ntasks=1 python train/train_synthesizer_finetune.py \
       --synth_checkpoint "$SYNTH_CKPT" --max_epochs "${MAX_EPOCHS:-20}" \
       --ser_loss "${SER_LOSS:-detached}"
fi
