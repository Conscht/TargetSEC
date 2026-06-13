#!/bin/bash
# Quick verification script to test ECAPA-TDNN setup

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     ECAPA-TDNN Setup Verification Test                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

PROJECT_DIR="/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM"
cd "$PROJECT_DIR" || exit 1

echo "✓ Location: $PROJECT_DIR"
echo ""

# Check files exist
echo "Checking core files..."
files=(
    "evaluate_speaker_similarity.py"
    "speaker_similarity_eval.slurm"
    "analyze_results.py"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        all_exist=false
    fi
done
echo ""

# Check audio directory structure
echo "Checking audio directory structure..."
wav_dir="eval_outputs/test1_768crossatt_synth_long_final_eval_guidance4__gs07/wav"
if [ -d "$wav_dir" ]; then
    echo "  ✓ Directory found: $wav_dir"
    
    # Count class directories
    class_count=0
    for i in {1..7}; do
        if [ -d "$wav_dir/class_$i" ]; then
            file_count=$(find "$wav_dir/class_$i" -name "*.wav" 2>/dev/null | wc -l)
            echo "    ├─ class_$i: $file_count files"
            ((class_count++))
        fi
    done
    echo "    └─ Total: $class_count classes"
else
    echo "  ✗ Directory not found: $wav_dir"
fi
echo ""

# Test Python environment and imports
echo "Testing Python environment (emoldm)..."
if conda activate emoldm 2>/dev/null; then
    echo "  ✓ Environment 'emoldm' activated"
    
    # Test imports
    python3 << 'PYEOF'
import sys
print("  ✓ Python version:", sys.version.split()[0])

try:
    import torch
    print("  ✓ torch:", torch.__version__)
except ImportError:
    print("  ✗ torch not installed")

try:
    import torchaudio
    print("  ✓ torchaudio:", torchaudio.__version__)
except ImportError:
    print("  ✗ torchaudio not installed")

try:
    import numpy
    print("  ✓ numpy:", numpy.__version__)
except ImportError:
    print("  ✗ numpy not installed")

import torch
if torch.cuda.is_available():
    print("  ✓ CUDA available")
    print("    - Device:", torch.cuda.get_device_name(0))
    print("    - Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
else:
    print("  ⚠ CUDA not available (will use CPU)")
PYEOF
else
    echo "  ✗ Environment 'emoldm' not found"
fi
echo ""

# Test script syntax
echo "Testing Python script syntax..."
python3 -m py_compile evaluate_speaker_similarity.py 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ evaluate_speaker_similarity.py: Syntax OK"
else
    echo "  ✗ evaluate_speaker_similarity.py: Syntax error"
fi

python3 -m py_compile analyze_results.py 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ analyze_results.py: Syntax OK"
else
    echo "  ✗ analyze_results.py: Syntax error"
fi
echo ""

# Test SLURM
echo "Testing SLURM availability..."
if command -v sbatch &>/dev/null; then
    echo "  ✓ sbatch command found"
else
    echo "  ⚠ sbatch not found (may not be on compute node)"
fi
echo ""

echo "╔════════════════════════════════════════════════════════════╗"
if [ "$all_exist" = true ]; then
    echo "║            ✅ SETUP VERIFICATION PASSED                   ║"
else
    echo "║            ⚠️  SOME FILES MISSING                         ║"
fi
echo "╚════════════════════════════════════════════════════════════╝"
