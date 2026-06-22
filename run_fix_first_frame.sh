#!/bin/bash
#SBATCH --job-name=vbench_eval
#SBATCH --partition=dgx-b200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --account=jgu32-lab
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=36:00:00
#SBATCH --output=logs/run_rtx6000_%j.out
#SBATCH --error=logs/run_rtx6000_%j.err


echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Navigate to project
cd /vast/projects/jgu32/lab/lulu/prj1/Inference_OP
mkdir -p logs

# Load CUDA module
module load cuda/12.8.1
export HF_HOME=/vast/projects/jgu32/lab/lulu/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/vast/projects/jgu32/lab/lulu/huggingface_cache

# Activate conda environment
source /vast/projects/jgu32/lab/lulu/miniconda3/etc/profile.d/conda.sh
conda activate vbch

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Check GPU
nvidia-smi
nvidia-smi --list-gpus
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Install flash-attn if needed
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "Installing flash-attn..."
    MAX_JOBS=4 pip install flash-attn --no-build-isolation
fi

# Download models if not present
echo "Checking for models..."
mkdir -p wan_models checkpoints

if [ ! -d "wan_models/Wan2.1-T2V-1.3B" ]; then
    echo "Downloading Wan2.1-T2V-1.3B model..."
    huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
        --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
else
    echo "Model already exists"
fi

if [ ! -f "checkpoints/self_forcing_dmd.pt" ]; then
    echo "Downloading Self-Forcing checkpoint..."
    huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
else
    echo "Checkpoint already exists"
fi

# Verify required files
echo "Verifying files..."
for file in configs/self_forcing_dmd.yaml checkpoints/self_forcing_dmd.pt vbench_proper_evaluation.py; do
    if [ -e "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file missing!"
        exit 1
    fi
done
# ============================================
# Load Gemini API key from secure file
if [ -f ~/.gemini_api_key ]; then
    export GEMINI_API_KEY=$(cat ~/.gemini_api_key)
    echo "✓ Loaded Gemini API key"
elif [ -f /home/lululiu/.gemini_api_key ]; then
    export GEMINI_API_KEY=$(cat /home/lululiu/.gemini_api_key)
    echo "✓ Loaded Gemini API key from home directory"
else
    echo "❌ ERROR: GEMINI_API_KEY not found!"
    echo "Please create ~/.gemini_api_key with your API key"
    exit 1
fi

# Install Google Generative AI SDK if not already installed
if ! python -c "import google.genai" 2>/dev/null; then
    echo "Installing google-genai..."
    pip install -q google-genai
else
    echo "✓ google-genai already installed"
fi
# ===============================================

# Run evaluation
echo "============================================"
echo "Starting VBench evaluation"
echo "Time: $(date)"
echo "============================================"

export PYTHONUNBUFFERED=1
python -u vbench_proper_evaluation.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --vbench_path ./VBench \
    --profile \
    --num_candidates 20 \
    --videos_dir vbench_official_eval/test/use_guidance_20 \
    --use_free_llm_for_checklist \
    --use_stage1_filter \
    --select_only \
    --dimensions test \
    --skip_baseline \
    --output_dir vbench_official_eval \
    --noise_generation_type pure_random \
    --use_ema \
    --num_gpus 2 \
    --vbench2 \
    --use_gemini \
    --seed 42 2>&1 | tee logs/debug_output_sp.log

echo "============================================"
echo "Completed at: $(date)"
echo "============================================"
