#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J abl_mlp
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/%x-spk%a-%j.out
#SBATCH --error=logs/%x-spk%a-%j.err
#SBATCH -C "GPU_SKU:A100"
#SBATCH --array=0-1

# MLP ablation, rebuilt on the CURRENT stage-2 encoder + stats. The old runs used
# the LibriTTS style encoder and December stats, so the MLP was regressing to a
# different latent space than the LDM -- not an apples-to-apples ablation.
cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate emoldm
export PYTHONPATH="$PWD"

SPK="$SLURM_ARRAY_TASK_ID"
srun --ntasks=1 python -u train/train_mlp_ablation.py \
    --style_stats style_stats_finetune_noser.pt \
    --synth_checkpoint "$SYNTH" \
    --use_speaker "$SPK" --max_epochs 300
