#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J synth_finetune2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH -C "GPU_SKU:A100"

cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate emoldm

# Continue from best finetune checkpoint (epoch=48, val_loss=16.24)
srun python train_synthesizer_finetune.py \
  --resume "checkpoints_synthesizer_finetune/synthesizer_finetune-06-13_18-22-18-epoch=48-val_loss=16.24.ckpt"
