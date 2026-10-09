#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J grid_search
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH -C "GPU_SKU:A100"

set -euo pipefail

cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"

mkdir -p logs

# env
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate emoldm

# (optional but nice) avoid multi-thread oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --cpu-bind=cores python grid_search.py \
  --save_dir eval_grid_ser \
  --n_utts 1000 \
  --steps 25,50,100 \
  --guidance 2,3,4 \
  --k 0.3,0.7,0.9 \
  --emotion_embedding_dir "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/MSP-Podcast-1.10/avgclass_emo_embeds" \
  --checkpoint_synth "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/checkpoints_synthesizer/synthesizer_training_speakr-12-14_15-51-55-epoch=89-val_loss=17.24.ckpt" \
  --checkpoint_ldm "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/processing/diffusion_model_training-12-17_15-25-20-epoch=560-val_loss=0.54.ckpt" \
  --style_ckpt "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/MSP-Podcast-1.10/pre-trained_models/pre-trained_style"
