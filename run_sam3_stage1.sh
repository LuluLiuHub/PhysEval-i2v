#!/bin/bash
#SBATCH --job-name=sam3_stage1
#SBATCH --partition=dgx-b200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --account=jgu32-lab
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=36:00:00
#SBATCH --output=logs/sam3_stage1_%j.out
#SBATCH --error=logs/sam3_stage1_%j.err


echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Redirect caches away from home (home quota nearly full)
export LAB=/vast/projects/jgu32/lab/lulu
export PIP_CACHE_DIR=$LAB/pip_cache
export CONDA_PKGS_DIRS=$LAB/conda_pkgs
export TMPDIR=$LAB/tmp
mkdir -p $LAB/pip_cache $LAB/conda_pkgs $LAB/tmp

# Navigate to project
cd $LAB/prj1/Inference_OP
mkdir -p logs sam3_masks

# Load CUDA module
module load cuda/12.8.1
export HF_HOME=$LAB/huggingface_cache
export HUGGINGFACE_HUB_CACHE=$LAB/huggingface_cache

# Activate vbch conda environment (has wan/diffusers; SAM3 runs as subprocess in sam3 env)
source $LAB/miniconda3/etc/profile.d/conda.sh
conda activate vbch

# Direct path to sam3 env python — avoids 'conda run' resolving the wrong interpreter
export SAM3_PYTHON=$LAB/miniconda3/envs/sam3/bin/python

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Check GPU
nvidia-smi
nvidia-smi --list-gpus
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ============================================
# Verify sam3 conda env has SAM3 installed
# ============================================
echo "Checking SAM3 in sam3 conda env ($SAM3_PYTHON)..."
$SAM3_PYTHON -c "from transformers import Sam3VideoModel, Sam3VideoProcessor; print('✓ SAM3 ready in sam3 env')" || {
    echo "SAM3 not found in sam3 env — installing dependencies..."
    SAM3_PIP=$LAB/miniconda3/envs/sam3/bin/pip
    $SAM3_PIP install einops ninja easydict Pillow matplotlib opencv-python-headless
    $SAM3_PIP install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
    $SAM3_PIP install git+https://github.com/ronghanghu/cc_torch.git
    $SAM3_PIP install git+https://github.com/facebookresearch/sam3.git
    $SAM3_PYTHON -c "from transformers import Sam3VideoModel, Sam3VideoProcessor; print('✓ SAM3 installed successfully')"
}

# Ensure sam3 env has image/plot dependencies (even if SAM3 itself was already installed)
SAM3_PIP=$LAB/miniconda3/envs/sam3/bin/pip
$SAM3_PYTHON -c "from PIL import Image; import matplotlib; import cv2" 2>/dev/null || {
    echo "Installing image deps in sam3 env..."
    $SAM3_PIP install Pillow matplotlib opencv-python-headless
}
echo "✓ sam3 env image deps OK"

# ============================================
# Install vbch dependencies if needed
# ============================================
echo "Checking vbch dependencies..."

if ! python -c "import flash_attn" 2>/dev/null; then
    echo "Installing flash-attn..."
    MAX_JOBS=4 pip install flash-attn --no-build-isolation
else
    echo "✓ flash-attn already installed"
fi

if ! python -c "import qwen_vl_utils" 2>/dev/null; then
    echo "Installing qwen-vl-utils..."
    pip install qwen-vl-utils
else
    echo "✓ qwen-vl-utils already installed"
fi

# ============================================
# Hugging Face Authentication
# ============================================
echo "Checking Hugging Face authentication..."

if [ -f $LAB/.hf_token ]; then
    export HF_TOKEN=$(cat $LAB/.hf_token)
    echo "✓ Found HF token in lab directory"
elif [ -f ~/.hf_token ]; then
    export HF_TOKEN=$(cat ~/.hf_token)
    echo "✓ Found HF token file"
else
    echo "❌ ERROR: HF token not found!"
    echo "Please create $LAB/.hf_token with your Hugging Face access token"
    exit 1
fi

python -c "
from huggingface_hub import login
login(token='$HF_TOKEN')
print('✓ Logged in to Hugging Face')
"

# ============================================
# Load Gemini API key from secure file
# ============================================
if [ -f $LAB/.gemini_api_key ]; then
    export GEMINI_API_KEY=$(cat $LAB/.gemini_api_key)
    echo "✓ Loaded Gemini API key from lab directory"
elif [ -f ~/.gemini_api_key ]; then
    export GEMINI_API_KEY=$(cat ~/.gemini_api_key)
    echo "✓ Loaded Gemini API key"
else
    echo "❌ ERROR: GEMINI_API_KEY not found!"
    echo "Please create $LAB/.gemini_api_key with your API key"
    exit 1
fi

if ! python -c "import google.genai" 2>/dev/null; then
    echo "Installing google-genai..."
    pip install -q google-genai
else
    echo "✓ google-genai already installed"
fi

# ============================================
# Run Stage 0 + Stage 1 (SAM3) Pipeline
# SAM3 contact detection runs as a subprocess
# in the sam3 conda env via pipeline/run_sam3_contact.py
# ============================================
echo "============================================"
echo "Starting Stage 0 + Stage 1 (SAM3) Pipeline"
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
    --seed 42 2>&1 | tee logs/sam3_stage1_output.log

echo "============================================"
echo "Completed at: $(date)"
echo "============================================"
