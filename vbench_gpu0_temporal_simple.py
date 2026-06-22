#!/usr/bin/env python3
"""
VBench Evaluation - GPU 0: Temporal Dimensions (Simple)
Generate videos with self-forcing, no guidance comparison
"""

import subprocess
import sys
import os

def main():
    # GPU 0 handles temporal dimensions
    temporal_dimensions = [
        "temporal_style",
        "temporal_flickering",
        "motion_smoothness",
        "dynamic_degree",
        "overall_consistency"
    ]

    print("=" * 60)
    print("VBench GPU 0: Temporal Dimensions")
    print("=" * 60)
    print(f"Processing {len(temporal_dimensions)} temporal dimensions:")
    for dim in temporal_dimensions:
        print(f"  • {dim}")
    print("=" * 60)

    # Extract arguments from command line
    vbench_path = "/path/to/VBench"
    config_path = "configs/self_forcing_dmd.yaml"
    checkpoint_path = "checkpoints/self_forcing_dmd.pt"

    # Parse arguments
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--vbench_path" and i + 1 < len(args):
            vbench_path = args[i + 1]
            i += 2
        elif args[i] == "--config_path" and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
        elif args[i] == "--checkpoint_path" and i + 1 < len(args):
            checkpoint_path = args[i + 1]
            i += 2
        else:
            i += 1

    # Process each dimension
    for i, dimension in enumerate(temporal_dimensions):
        print(f"\n🎬 Processing dimension {i+1}/{len(temporal_dimensions)}: {dimension}")
        print("-" * 50)

        # Get dimension prompts file
        dimension_file = os.path.join(vbench_path, "prompts", "prompts_per_dimension", f"{dimension}.txt")

        if not os.path.exists(dimension_file):
            print(f"⚠️ Dimension file not found: {dimension_file}")
            continue

        # Create command for this dimension
        cmd = [
            "python", "vbench_proper_evaluation.py",
            "--config_path", config_path,
            "--checkpoint_path", checkpoint_path,
            "--vbench_path", vbench_path,
            "--output_dir", f"./eval_results_gpu0_temporal",
            "--use_ema",
            # "--max_prompts_per_category", "50",  # Removed for full evaluation
            # Only process this dimension's file
            "--evaluate_all_dimensions",  # Use single file mode
            "--data_path", dimension_file  # Process this dimension only
        ]

        # Set GPU environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env, check=True)
            print(f"✅ Completed {dimension}")

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed {dimension}: {e}")
            continue
        except KeyboardInterrupt:
            print(f"\n⏹️ Interrupted during {dimension}")
            break

    print("\n" + "=" * 60)
    print("🎬 GPU 0 (Temporal) - COMPLETED")
    print("=" * 60)
    print("Results: ./eval_results_gpu0_temporal/")

if __name__ == "__main__":
    main()