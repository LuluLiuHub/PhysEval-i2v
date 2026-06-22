#!/bin/bash
#SBATCH --partition=dgx-b200-mig90
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:90gb:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/setup_deps_%j.out
#SBATCH --error=logs/setup_deps_%j.err
#SBATCH --account=jgu32-lab
#SBATCH --job-name=setup_vlm_deps

echo "============================================"
echo "Setting up VideoLLaMA3 Dependencies"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================"

# Navigate to project
cd /vast/projects/jgu32/lab/lulu/prj1/Inference_OP
mkdir -p logs

# Load CUDA module
module load cuda/12.8.1

# Set CUDA_HOME
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
echo "CUDA_HOME: $CUDA_HOME"
echo "NVCC: $(which nvcc)"
nvcc --version

# Set cache directories
export HF_HOME=/vast/projects/jgu32/lab/lulu/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/vast/projects/jgu32/lab/lulu/huggingface_cache

# Activate conda environment
source /vast/projects/jgu32/lab/lulu/miniconda3/etc/profile.d/conda.sh
conda activate vbch

echo "============================================"
echo "Current Environment:"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================"

# Install VideoLLaMA3 dependencies
echo "Installing VideoLLaMA3 dependencies..."
pip install -q transformers>=4.45.0 accelerate av decord ffmpeg-python qwen-vl-utils pillow opencv-python

# Uninstall old flash-attn
echo "Removing old flash-attn..."
pip uninstall flash-attn -y 2>/dev/null

# Install ninja for faster compilation
echo "Installing ninja for faster compilation..."
pip install -q ninja

# Install flash-attn (this will take 5-10 minutes)
echo "============================================"
echo "Installing flash-attn (this may take 5-10 minutes)..."
echo "============================================"
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Verify installation
echo "============================================"
echo "Verifying installations..."
echo "============================================"

python -c "
import torch
import transformers
import flash_attn
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ Transformers: {transformers.__version__}')
print(f'✓ Flash-Attn: {flash_attn.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')

# Test flash-attn import
from flash_attn import flash_attn_func
print('✓ flash_attn_func imported successfully')
"

# Download VideoLLaMA3 model
echo "============================================"
echo "Downloading VideoLLaMA3 model..."
echo "============================================"
huggingface-cli download DAMO-NLP-SG/VideoLLaMA3-7B

echo "============================================"
echo "Setup completed successfully!"
echo "Completed at: $(date)"
echo "============================================"
echo ""
echo "You can now run your evaluation with:"
echo "sbatch slurm_run_evaluation.sh"
