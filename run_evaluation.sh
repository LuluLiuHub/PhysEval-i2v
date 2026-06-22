#!/bin/bash
#SBATCH --job-name=vbench_eval
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
nvidia-smi --gpu-reset

# Navigate to project
cd /vast/projects/jgu32/lab/lulu/prj1/Inference_OP
mkdir -p logs

# Load CUDA module
module load cuda/12.8.1
export HF_HOME=/vast/projects/jgu32/lab/lulu/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/vast/projects/jgu32/lab/lulu/huggingface_cache

# Activate conda environment
source /vast/projects/jgu32/lab/lulu/miniconda3/etc/profile.d/conda.sh
conda activate vbch1

# Verify environment
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Check GPU
nvidia-smi

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
python -u vbench_proper_evaluation.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --vbench_path ./VBench \
    --num_candidates 10 \
    --profile \
    --evaluation_only \
    --dimensions multiple_objects \
    --output_dir vbench_official_eval \
    --noise_generation_type mixed \
    --num_gpus 2 \
    --use_ema \
    --seed 42 2>&1 | tee logs/debug_output.log 

echo "============================================"
echo "Completed at: $(date)"
echo "============================================"
