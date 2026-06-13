#!/bin/bash
# Quick start examples for ECAPA-TDNN speaker similarity evaluation

# Make sure you're in the correct directory
cd "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"

# Activate the environment
conda activate emoldm

echo "======================================"
echo "ECAPA-TDNN Speaker Similarity Eval"
echo "======================================"
echo ""

# OPTION 1: Run directly with default settings (intra-class only)
echo "OPTION 1: Direct execution with default settings"
echo "Command:"
echo "python evaluate_speaker_similarity.py"
echo ""
echo "This will evaluate speaker similarity within each emotion class (1-7)"
echo ""
echo ""

# OPTION 2: Run with inter-class evaluation
echo "OPTION 2: Direct execution with inter-class evaluation"
echo "Command:"
echo "python evaluate_speaker_similarity.py --compute_inter_class --samples_per_class 10"
echo ""
echo "This includes inter-class speaker similarity comparison"
echo ""
echo ""

# OPTION 3: Submit as SLURM job
echo "OPTION 3: Submit as SLURM job (recommended for long runs)"
echo "Command:"
echo "sbatch speaker_similarity_eval.slurm"
echo ""
echo "Check job status:"
echo "squeue -u \$USER"
echo ""
echo "Monitor job output:"
echo "tail -f logs/speaker_similarity_eval_<JOB_ID>.log"
echo ""
echo ""

# OPTION 4: Custom paths
echo "OPTION 4: Custom evaluation paths"
echo "Command:"
echo "python evaluate_speaker_similarity.py \\"
echo "    --wav_dir 'eval_outputs/your_eval_dir/wav' \\"
echo "    --output_dir 'eval_outputs/your_eval_dir' \\"
echo "    --compute_inter_class \\"
echo "    --samples_per_class 10"
echo ""
echo ""

echo "======================================"
echo "Results will be saved as:"
echo "  - speaker_similarity_results.json"
echo "  - speaker_similarity_summary.txt"
echo "======================================"
