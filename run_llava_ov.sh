#!/bin/bash
#SBATCH --job-name=vbench_eval
#SBATCH --partition=b200-mig90
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
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

# Verify environment
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Check GPU
nvidia-smi
nvidia-smi --list-gpus
echo $CUDA_VISIBLE_DEVICES
# After activating conda, add:
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

# Run evaluation
echo "============================================"
echo "Starting VBench evaluation"
echo "Time: $(date)"
echo "============================================"

export PYTHONUNBUFFERED=1

echo "============================================"
echo "Completed at: $(date)"
echo "============================================"
python video_comparison_llava.py \
    --video_paths vbench_official_eval/Mechanics/use_guidance/"A small rubber beach ball is thrown towards the ground, showing the ball's high bounce after impact with the surface.-1.mp4"  vbench_official_eval/Mechanics/use_guidance/"A small rubber beach ball is thrown towards the ground, showing the ball's high bounce after impact with the surface.-2.mp4" \
    --prompt "A small rubber beach ball is thrown towards the ground, showing the ball's high bounce after impact with the surface." \
    --num_frames 16 \
    --max_new_tokens 500 \
    --output results.txt
