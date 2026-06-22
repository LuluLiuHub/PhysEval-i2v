#!/bin/bash
#SBATCH --partition=dgx-b200-mig90
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:90gb:1
#SBATCH --time=36:00:00
#SBATCH --output=logs/run_mig90_%j.out
#SBATCH --error=logs/run_mig90_%j.err
#SBATCH --account=jgu32-lab

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

# ============================================
# FIX DEPENDENCIES
# ============================================
echo "============================================"
echo "Setting up dependencies..."
echo "============================================"

# Check PyTorch version and downgrade if needed
TORCH_VERSION=$(python -c 'import torch; print(torch.__version__.split("+")[0])')
if [ "$TORCH_VERSION" != "2.8.0" ]; then
    echo "Downgrading PyTorch from $TORCH_VERSION to 2.8.0..."
    pip install torch==2.8.0 torchvision==0.19.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu124
fi

# Install VideoLLaMA3 dependencies
echo "Installing VideoLLaMA3 dependencies..."
pip install -q transformers>=4.45.0 accelerate av decord ffmpeg-python qwen-vl-utils pillow opencv-python

# Check and reinstall flash-attn if needed
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "Installing flash-attn for PyTorch 2.8.0..."
    pip uninstall flash-attn -y 2>/dev/null
    MAX_JOBS=4 pip install flash-attn --no-build-isolation
else
    # Test if flash-attn works with current PyTorch
    if ! python -c "from flash_attn import flash_attn_func" 2>/dev/null; then
        echo "flash-attn incompatible, reinstalling..."
        pip uninstall flash-attn -y
        MAX_JOBS=4 pip install flash-attn --no-build-isolation
    else
        echo "flash-attn already installed and working"
    fi
fi

# Verify environment
echo "============================================"
echo "Environment verification:"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Flash-attn: $(python -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null || echo 'Not installed')"
echo "============================================"

# Check GPU
nvidia-smi

# ============================================
# DOWNLOAD MODELS
# ============================================
echo "Checking for models..."
mkdir -p wan_models checkpoints

if [ ! -d "wan_models/Wan2.1-T2V-1.3B" ]; then
    echo "Downloading Wan2.1-T2V-1.3B model..."
    huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
        --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
else
    echo "✓ Wan2.1-T2V-1.3B model exists"
fi

if [ ! -f "checkpoints/self_forcing_dmd.pt" ]; then
    echo "Downloading Self-Forcing checkpoint..."
    huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
else
    echo "✓ Self-Forcing checkpoint exists"
fi

# Download VideoLLaMA3 model if not cached
echo "Checking VideoLLaMA3 model..."
if ! python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('DAMO-NLP-SG/VideoLLaMA3-7B', trust_remote_code=True)" 2>/dev/null; then
    echo "Downloading VideoLLaMA3-7B model..."
    huggingface-cli download DAMO-NLP-SG/VideoLLaMA3-7B
else
    echo "✓ VideoLLaMA3-7B model cached"
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
# RUN EVALUATION
# ============================================
echo "============================================"
echo "Starting VBench evaluation with VideoLLaMA3"
echo "Time: $(date)"
echo "============================================"

export PYTHONUNBUFFERED=1
python -u vbench_proper_evaluation.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --vbench_path ./VBench \
    --num_candidates 2 \
    --profile \
    --output_dir vbench_official_eval \
    --max_prompts_per_category 20 \
    --noise_generation_type mixed \
    --use_ema \
    --skip_baseline \
    --seed 42 2>&1 | tee logs/debug_output.log

echo "============================================"
echo "Completed at: $(date)"
echo "============================================"
