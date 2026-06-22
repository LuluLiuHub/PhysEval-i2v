#!/bin/bash

# VBench Parallel Evaluation Setup Script
# Update these paths according to your setup

echo "🔧 Setting up VBench Parallel Evaluation"
echo "=" * 50

# Configuration - UPDATE THESE PATHS
VBENCH_PATH="/path/to/VBench"  # Update this!
CONFIG_PATH="configs/self_forcing_dmd.yaml"
CHECKPOINT_PATH="checkpoints/self_forcing_dmd.pt"

echo "📝 Current configuration:"
echo "  VBench path: $VBENCH_PATH"
echo "  Config path: $CONFIG_PATH"
echo "  Checkpoint path: $CHECKPOINT_PATH"
echo ""

# Update GPU scripts with correct paths
echo "🔄 Updating GPU scripts with your paths..."

# Update GPU 0 script
sed -i "s|/path/to/VBench|$VBENCH_PATH|g" vbench_gpu0_temporal.py
sed -i "s|configs/self_forcing_dmd.yaml|$CONFIG_PATH|g" vbench_gpu0_temporal.py
sed -i "s|checkpoints/self_forcing_dmd.pt|$CHECKPOINT_PATH|g" vbench_gpu0_temporal.py

# Update GPU 1 script
sed -i "s|/path/to/VBench|$VBENCH_PATH|g" vbench_gpu1_nontemporal.py
sed -i "s|configs/self_forcing_dmd.yaml|$CONFIG_PATH|g" vbench_gpu1_nontemporal.py
sed -i "s|checkpoints/self_forcing_dmd.pt|$CHECKPOINT_PATH|g" vbench_gpu1_nontemporal.py

echo "✅ GPU scripts updated"

# Make scripts executable
chmod +x vbench_gpu0_temporal.py
chmod +x vbench_gpu1_nontemporal.py
chmod +x run_vbench_parallel.py

echo "✅ Scripts made executable"

echo ""
echo "🚀 Ready to run! Use one of these commands:"
echo ""
echo "# Run both GPUs in parallel:"
echo "python run_vbench_parallel.py"
echo ""
echo "# Run individual GPUs:"
echo "python vbench_gpu0_temporal.py      # Temporal dimensions on GPU 0"
echo "python vbench_gpu1_nontemporal.py   # Non-temporal dimensions on GPU 1"
echo ""
echo "# With Qwen optimization:"
echo "python run_vbench_parallel.py --use_qwen_optimization"
echo ""
echo "📁 Results will be saved to:"
echo "  • ./eval_results_gpu0_temporal/"
echo "  • ./eval_results_gpu1_nontemporal/"
echo "  • ./eval_results_merged/"