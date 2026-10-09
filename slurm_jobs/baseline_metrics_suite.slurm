#!/bin/bash
#SBATCH -A sci-demelo-mpws2025gd1
#SBATCH -p gpu-batch
#SBATCH -J base_metrics
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH -C "GPU_SKU:A100"

# Speaker similarity + independent SER + WER for a system whose wavs already
# exist at $WAV_ROOT/class_{1..7}/utt_%06d.wav.
#   sbatch --export=ALL,WAV_ROOT=<...>/wav,NAME=HiFiGAN slurm_jobs/baseline_metrics_suite.slurm

cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate emoldm
export PYTHONPATH="$PWD"

echo "########## ECAPA speaker similarity"
srun --ntasks=1 python eval/evaluate_speaker.py --wav_root "$WAV_ROOT"

echo "########## Independent SER (SpeechBrain IEMOCAP)"
srun --ntasks=1 python eval/second_emotion_eval.py --wav_root "$WAV_ROOT" --name "${NAME:-system}"

echo "########## WER (Whisper medium, random 2000/class)"
srun --ntasks=1 python -u eval/wer_eval_perclass.py --wav_root "$WAV_ROOT" --sample_n 2000 --sample_seed 0

echo "SUITE DONE"
