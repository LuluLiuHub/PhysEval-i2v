#!/usr/bin/env python3
"""
Split Extended Prompts into Per-Dimension Files
Reads the extended prompts from prompts/vbench/all_dimension_extended.txt and splits them
into separate files per dimension, matching the VBench structure.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

def split_extended_prompts():
    """Split extended prompts into per-dimension files"""

    # Paths
    project_root = Path("/Users/lululiu/ml/Inference_OP")
    vbench_root = project_root / "VBench"

    vbench_info_path = vbench_root / "vbench" / "VBench_full_info.json"
    extended_prompts_path = project_root / "prompts" / "vbench" / "all_dimension_extended.txt"
    output_dir = project_root / "prompts" / "vbench" / "extended_prompts_per_dimension"

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("Splitting Extended Prompts into Per-Dimension Files")
    print("=" * 60)
    print(f"VBench info: {vbench_info_path}")
    print(f"Extended prompts: {extended_prompts_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Load VBench info (prompt-to-dimension mapping)
    with open(vbench_info_path, 'r') as f:
        vbench_info = json.load(f)

    # Load extended prompts
    with open(extended_prompts_path, 'r') as f:
        extended_prompts = [line.strip() for line in f if line.strip()]

    print(f"\nLoaded {len(vbench_info)} prompt mappings")
    print(f"Loaded {len(extended_prompts)} extended prompts")

    # Verify counts match
    if len(vbench_info) != len(extended_prompts):
        print(f"❌ ERROR: Mismatch in counts!")
        print(f"   VBench info: {len(vbench_info)} prompts")
        print(f"   Extended prompts: {len(extended_prompts)} prompts")
        return False

    # Group prompts by dimension
    dimension_prompts = defaultdict(list)

    for i, (info, extended_prompt) in enumerate(zip(vbench_info, extended_prompts)):
        # Each prompt can belong to multiple dimensions
        dimensions = info.get('dimension', [])

        for dimension in dimensions:
            dimension_prompts[dimension].append(extended_prompt)

    # Save prompts to per-dimension files
    print(f"\nSplitting into {len(dimension_prompts)} dimension files:")
    print("-" * 60)

    dimension_counts = {}
    for dimension in sorted(dimension_prompts.keys()):
        prompts = dimension_prompts[dimension]
        dimension_counts[dimension] = len(prompts)

        # Save to file
        output_file = output_dir / f"{dimension}.txt"
        with open(output_file, 'w') as f:
            for prompt in prompts:
                f.write(f"{prompt}\n")

        print(f"  ✓ {dimension:<30} {len(prompts):>4} prompts -> {output_file.name}")

    print("-" * 60)
    print(f"Total: {sum(dimension_counts.values())} prompt assignments across {len(dimension_prompts)} dimensions")
    print(f"\n✓ Extended prompts split successfully!")
    print(f"Output directory: {output_dir}")

    # Compare with original prompts_per_dimension to verify
    original_dir = vbench_root / "prompts" / "prompts_per_dimension"
    if original_dir.exists():
        print("\n" + "=" * 60)
        print("Comparison with Original VBench Prompts:")
        print("=" * 60)
        print(f"{'Dimension':<30} {'Original':>10} {'Extended':>10} {'Match':>10}")
        print("-" * 60)

        for dimension in sorted(dimension_counts.keys()):
            original_file = original_dir / f"{dimension}.txt"
            if original_file.exists():
                with open(original_file, 'r') as f:
                    original_count = len([line for line in f if line.strip()])
                extended_count = dimension_counts[dimension]
                match = "✓" if original_count == extended_count else "✗"
                print(f"{dimension:<30} {original_count:>10} {extended_count:>10} {match:>10}")
            else:
                print(f"{dimension:<30} {'N/A':>10} {dimension_counts[dimension]:>10} {'N/A':>10}")

        print("-" * 60)

    return True

if __name__ == "__main__":
    success = split_extended_prompts()
    exit(0 if success else 1)
