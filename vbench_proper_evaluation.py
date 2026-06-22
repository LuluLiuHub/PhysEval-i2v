#!/usr/bin/env python3
"""
VBench Official Evaluation Script following exact VBench sampling protocol
- Sample videos from extended prompts (prompts/vbench/extended_prompts_per_dimension) by default
- Option to use original VBench prompts (prompts/prompts_per_dimension)
- 5 videos per prompt (25 for temporal_flickering)
- Proper random seed management and reproducibility
- Exact naming convention: prompt-index.mp4
"""

import os
import argparse
import json
import subprocess
import torch
from pathlib import Path
from tqdm import tqdm
import tempfile
import random
import sys
import socket

# No longer need VBench aggregate scoring - focusing on per-dimension comparison

def find_free_port(start_port=29500, max_attempts=100):
    """Find a free port by attempting to bind to it"""
    for offset in range(max_attempts):
        port = start_port + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port after {max_attempts} attempts starting from {start_port}")

def parse_args():
    parser = argparse.ArgumentParser(description="VBench evaluation following exact official protocol")
    parser.add_argument("--config_path", type=str, required=True, help="Path to model config")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vbench_path", type=str, required=True, help="Path to VBench installation directory")
    parser.add_argument("--output_dir", type=str, default="vbench_official_eval", help="Output directory")
    parser.add_argument("--num_candidates", type=int, default=10,
                        help="Number of noise candidates for guidance")
    parser.add_argument("--noise_generation_type", type=str, default="mixed",
                        choices=["mixed", "scale_variations", "pure_random", "edge_cases",
                                "frequency_variations", "spatial_correlations", "temporal_correlations",
                                "low_discrepancy"],
                        help="Noise generation type to use for guidance")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility")
    parser.add_argument("--evaluate_all_dimensions", action="store_true",
                        help="Evaluate using all_dimension.txt instead of per-dimension files")
    parser.add_argument("--max_prompts_per_category", type=int, default=None,
                        help="Maximum number of prompts to generate per category/dimension (default: None = all prompts)")
    parser.add_argument("--use_ema", action="store_true",
                        help="Use EMA parameters from checkpoint")
    parser.add_argument("--evaluation_only", action="store_true",
                        help="Skip video generation and only run evaluation on existing videos")
    parser.add_argument("--dimensions", type=str, nargs='+', default=None,
                        help="Specific dimension(s) to evaluate (e.g., 'temporal_flickering' or 'temporal_flickering color'). If not specified, all dimensions are evaluated.")
    parser.add_argument("--profile", action="store_true",
                        help="Enable profiling to measure VAE decoding and denoising time")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline generation and only generate videos with guidance")
    parser.add_argument("--skip_guided", action="store_true",
                        help="Skip guided generation and only generate baseline videos")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of worker GPUs for parallel candidate generation. Total GPUs needed: num_gpus+1 (GPU 0 reserved for scorer). Example: --num_gpus=4 requires 5 GPUs total.")
    parser.add_argument("--vbench2", action="store_true",
                        help="Use VBench 2.0 prompts from prompts/prompt and prompts/prompt_aug/wanx_aug_prompt")
    parser.add_argument("--use_fixed_first_frame", action="store_true",
                        help="Fix the first frame and only vary subsequent frames across candidates")
    parser.add_argument("--use_gemini", action="store_true",
                        help="Use Gemini API for physics-aware video comparison instead of VideoLLaMA3")
    parser.add_argument("--use_smc", action="store_true",
                        help="Use SMC (Sequential Monte Carlo) sampling instead of regular inference")
    parser.add_argument("--smc_block_sizes", type=int, nargs="+", default=[6, 6, 6],
                        help="SMC block sizes (default: 6 6 6 for num_frame_per_block=3 models)")
    parser.add_argument("--smc_initial_particles", type=int, default=50,
                        help="SMC: number of particles in Block 1 (default: 50)")
    parser.add_argument("--smc_top_k", type=int, default=10,
                        help="SMC: number of top particles to keep after each stage (default: 10)")
    parser.add_argument("--smc_branch_factor", type=int, default=5,
                        help="SMC: branching factor for Block 2+ (default: 5)")
    parser.add_argument("--max_gemini_stage3_workers", type=int, default=5,
                        help="Max concurrent Gemini API calls for Stage 3 tournament (default: 5)")
    parser.add_argument("--use_free_llm_for_checklist", action="store_true",
                        help="Use FREE open-source LLM (Qwen) for prompt checklist generation instead of Gemini (saves API costs)")
    parser.add_argument("--free_llm_model", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct",
                        help="Hugging Face model for free LLM (default: Qwen/Qwen2.5-Coder-32B-Instruct, lighter: Qwen/Qwen2.5-7B-Instruct or Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--use_qwen_vl", action="store_true",
                        help="Use FREE Qwen3-VL-30B for video evaluation (Stage 1 & 2) instead of Gemini - makes entire pipeline 100%% free")
    parser.add_argument("--qwen_vl_model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        help="Qwen VL model for video understanding (default: Qwen/Qwen3-VL-30B-A3B-Instruct)")
    parser.add_argument("--qwen_vl_fps", type=float, default=1.0,
                        help="FPS for Qwen3-VL video processing (lower = less memory, default: 1.0)")
    parser.add_argument("--qwen_vl_max_pixels", type=int, default=480*720,
                        help="Max pixels for Qwen3-VL video frames (lower = less memory, default: 480*720)")

    # Stage 1 filter parameters
    parser.add_argument("--use_stage1_filter", action="store_true",
                        help="Enable Stage 1 hallucination filter (Phase 1a + 1b) to pre-filter bad videos before expensive Stage 2 forensics")
    parser.add_argument("--stage1_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="VL model for Stage 1 filtering (default: Qwen/Qwen3-VL-8B-Instruct)")
    parser.add_argument("--stage1_num_gpus", type=int, default=None,
                        help="Number of GPUs for Stage 1 parallel processing (default: None = auto-detect all available GPUs)")
    parser.add_argument("--stage1_workers_per_gpu", type=int, default=5,
                        help="Number of concurrent worker threads per GPU for Stage 1 (default: 5)")
    parser.add_argument("--stage1_max_iterations", type=int, default=5,
                        help="Max iterations for Stage 1 convergence loop (default: 5)")

    # Workflow control
    parser.add_argument("--generate_only", action="store_true",
                        help="Only generate candidate videos without selection (saves to output_dir/videos/)")
    parser.add_argument("--select_only", action="store_true",
                        help="Only select best from pre-generated videos (requires --videos_dir)")
    parser.add_argument("--videos_dir", type=str, default=None,
                        help="Directory containing pre-generated candidate videos (for --select_only mode)")

    return parser.parse_args()

def get_all_dimension_files(vbench_path, use_extended_prompts=True):
    """Get all dimension txt files from extended prompts or original VBench prompts"""
    if use_extended_prompts:
        # Use extended prompts from prompts/vbench/extended_prompts_per_dimension
        # Get path relative to current working directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dimension_dir = os.path.join(script_dir, "prompts", "vbench", "extended_prompts_per_dimension")
        print(f"Using EXTENDED prompts from: {dimension_dir}")
    else:
        # Use original VBench prompts
        dimension_dir = os.path.join(vbench_path, "prompts", "prompts_per_dimension")
        print(f"Using ORIGINAL VBench prompts from: {dimension_dir}")

    if not os.path.exists(dimension_dir):
        print(f"✗ Dimension directory not found: {dimension_dir}")
        return []

    dimension_files = []
    for file in os.listdir(dimension_dir):
        if file.endswith('.txt'):
            dimension_name = file.replace('.txt', '')
            file_path = os.path.join(dimension_dir, file)
            dimension_files.append((dimension_name, file_path))

    print(f"✓ Found {len(dimension_files)} dimension files")
    return dimension_files

def sample_videos_for_dimension(config_path, checkpoint_path, dimension_name, prompts_file,
                              base_output_dir, use_guidance=False, num_candidates=10,
                              noise_type="mixed", base_seed=42, max_prompts_per_category=50,
                              use_ema=False, profile=False, extended_prompts_file=None, num_gpus=1,
                              use_fixed_first_frame=False, use_gemini=False, use_smc=False,
                              smc_block_sizes=[6, 6, 6], smc_initial_particles=50,
                              smc_top_k=10, smc_branch_factor=5,
                              max_gemini_stage3_workers=5,
                              use_free_llm_for_checklist=False, free_llm_model='Qwen/Qwen2.5-Coder-32B-Instruct',
                              use_qwen_vl=False, qwen_vl_model='Qwen/Qwen3-VL-30B-A3B-Instruct',
                              qwen_vl_fps=1.0, qwen_vl_max_pixels=480*720,
                              use_stage1_filter=False, stage1_model='Qwen/Qwen3-VL-8B-Instruct',
                              stage1_num_gpus=None, stage1_workers_per_gpu=5, stage1_max_iterations=5,
                              generate_only=False):
    """Sample videos following exact VBench protocol with organized directory structure.
    Optimized version: Makes single call to inference.py for all prompts (loads model once)."""

    # Set random seed at the beginning
    torch.manual_seed(base_seed)
    random.seed(base_seed)

    # Read prompt list
    with open(prompts_file, 'r') as f:
        prompt_list = f.readlines()
    prompt_list = [prompt.strip() for prompt in prompt_list if prompt.strip()]

    # Read extended prompt list if provided
    if extended_prompts_file:
        with open(extended_prompts_file, 'r') as f:
            extended_prompt_list = f.readlines()
        extended_prompt_list = [prompt.strip() for prompt in extended_prompt_list if prompt.strip()]
        assert len(extended_prompt_list) == len(prompt_list), \
            f"Extended prompts ({len(extended_prompt_list)}) must match original prompts ({len(prompt_list)})"
    else:
        extended_prompt_list = None

    # Limit prompts per category if specified
    if max_prompts_per_category and len(prompt_list) > max_prompts_per_category:
        prompt_list = prompt_list[:max_prompts_per_category]
        if extended_prompt_list:
            extended_prompt_list = extended_prompt_list[:max_prompts_per_category]
        print(f"Limited {dimension_name} to {max_prompts_per_category} prompts")

    # Create temporary file with limited prompts if needed
    if max_prompts_per_category and len(prompt_list) <= max_prompts_per_category:
        # Need to create truncated file for inference.py
        temp_prompts_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        for prompt in prompt_list:
            temp_prompts_file.write(f"{prompt}\n")
        temp_prompts_file.close()
        actual_prompts_file = temp_prompts_file.name

        # Create temp file for extended prompts too
        if extended_prompt_list:
            temp_extended_file = tempfile.NamedTemporaryFile(mode='w', suffix='_extended.txt', delete=False)
            for prompt in extended_prompt_list:
                temp_extended_file.write(f"{prompt}\n")
            temp_extended_file.close()
            actual_extended_file = temp_extended_file.name
        else:
            temp_extended_file = None
            actual_extended_file = None
    else:
        # Use original files
        actual_prompts_file = prompts_file
        temp_prompts_file = None
        actual_extended_file = extended_prompts_file
        temp_extended_file = None

    # Determine number of videos per prompt
    # VBench 1.0: 5 videos per prompt (25 for temporal_flickering)
    # VBench 2.0: 3 videos per prompt (20 for Diversity)
    if dimension_name == "temporal_flickering":
        videos_per_prompt = 25  # VBench 1.0
    elif dimension_name == "Diversity":
        videos_per_prompt = 20  # VBench 2.0
    else:
        videos_per_prompt = 3  # VBench 2.0 default

    # Limit batch size to avoid OOM (especially for temporal_flickering with 25 videos)
    # Multi-GPU worker processes have ~774MB overhead vs single-GPU mode
    # 90GB MIG: Model (88.18GB) + overhead (774MB) + batch=1 activations (~200MB) ≈ 89.2GB
    # batch_size=2 needs ~400MB which causes OOM
    max_batch_size = 5

    # Create organized directory structure: base_output_dir/dimension_name/condition_num_candidates/
    condition_name = "use_guidance" if use_guidance else "baseline"
    folder_name = f"{condition_name}_{num_candidates}"
    output_folder = os.path.join(base_output_dir, dimension_name, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    print(f"Sampling {videos_per_prompt} videos per prompt for dimension: {dimension_name}")
    print(f"Condition: {condition_name}")
    print(f"Total prompts: {len(prompt_list)}")
    print(f"Total videos to generate: {len(prompt_list) * videos_per_prompt}")
    print(f"Output folder: {output_folder}")

    # Calculate number of batches needed (ceiling division to avoid OOM)
    num_batches = (videos_per_prompt + max_batch_size - 1) // max_batch_size

    if num_batches > 1:
        print(f"⚠️  Splitting into {num_batches} batches of up to {max_batch_size} videos to avoid OOM")

    # ========== GENERATE VIDEOS IN BATCHES ==========
    all_success = True
    for batch_idx in range(num_batches):
        # Calculate batch range
        batch_start = batch_idx * max_batch_size
        batch_end = min(batch_start + max_batch_size, videos_per_prompt)
        batch_size = batch_end - batch_start

        if num_batches > 1:
            print(f"\n🚀 Batch {batch_idx + 1}/{num_batches}: Generating {batch_size} videos per prompt (indices {batch_start}-{batch_end-1})")
        else:
            print(f"\n🚀 Generating all {len(prompt_list)} prompts in one batch (model loads once)...")

        # Use the active conda env's Python for inference subprocess.
        # Priority: VBCH_PYTHON env var > CONDA_PREFIX/bin/python > sys.executable
        import os as _os
        _vbch_python = (
            _os.environ.get("VBCH_PYTHON")
            or (f"{_os.environ['CONDA_PREFIX']}/bin/python" if _os.environ.get("CONDA_PREFIX") else None)
            or sys.executable
        )
        print(f"   inference Python: {_vbch_python}  (CONDA_PREFIX={_os.environ.get('CONDA_PREFIX','not set')})")

        # Build command to call inference.py OR smc_inference.py for this batch
        if use_smc:
            # ── SMC mode: always uses torchrun for multi-GPU ──────────────────
            script_name = "smc_inference.py"
            master_port = find_free_port(start_port=29500, max_attempts=100)
            print(f"  🔬 Using SMC sampling with {num_gpus} GPUs (port {master_port})")
            cmd = [
                _vbch_python, "-m", "torch.distributed.run",
                f"--nproc_per_node={num_gpus}",
                f"--master_port={master_port}",
                script_name,
                "--config_path", config_path,
                "--checkpoint_path", checkpoint_path,
                "--data_path", actual_prompts_file,  # Original prompts (for filenames)
                "--output_folder", output_folder,
                "--num_samples", str(batch_size),  # Generate videos for this batch only
                "--start_index", str(batch_start),  # Offset saved indices to avoid overwriting
                "--seed", str(base_seed + batch_start),  # Offset seed for each batch
                "--num_output_frames", "21",
                "--height", "480",
                "--width", "832",
                "--use_gemini",  # SMC requires Gemini for evaluation
            ]
            # SMC-specific args
            cmd.extend([
                "--smc_block_sizes"] + [str(bs) for bs in smc_block_sizes])
            cmd.extend([
                "--smc_initial_particles", str(smc_initial_particles),
                "--smc_top_k", str(smc_top_k),
                "--smc_branch_factor", str(smc_branch_factor)
            ])
        else:
            # ── Regular inference.py mode ──────────────────────────────────────
            script_name = "inference.py"
            # Use torchrun for DDP-based multi-GPU candidate generation
            if use_guidance and num_gpus > 1:
                master_port = find_free_port(start_port=29500, max_attempts=100)
                print(f"  Using available port: {master_port}")
                cmd = [
                    _vbch_python, "-m", "torch.distributed.run",
                    f"--nproc_per_node={num_gpus}",
                    f"--master_port={master_port}",
                    script_name,
                    "--config_path", config_path,
                    "--checkpoint_path", checkpoint_path,
                    "--data_path", actual_prompts_file,  # Original prompts (for filenames)
                    "--output_folder", output_folder,
                    "--num_samples", str(batch_size),  # Generate videos for this batch only
                    "--start_index", str(batch_start),  # Offset saved indices to avoid overwriting
                    "--seed", str(base_seed + batch_start),  # Offset seed for each batch
                    "--num_output_frames", "21"
                ]
            else:
                # Single-GPU mode: use python directly
                cmd = [
                    sys.executable, script_name,
                    "--config_path", config_path,
                    "--checkpoint_path", checkpoint_path,
                    "--data_path", actual_prompts_file,  # Original prompts (for filenames)
                    "--output_folder", output_folder,
                    "--num_samples", str(batch_size),  # Generate videos for this batch only
                    "--start_index", str(batch_start),  # Offset saved indices to avoid overwriting
                    "--seed", str(base_seed + batch_start),  # Offset seed for each batch
                    "--num_output_frames", "21"
                ]

        # Add extended prompts if provided (for generation quality)
        if actual_extended_file:
            cmd.extend(["--extended_prompt_path", actual_extended_file])

        # Add optional flags (shared between inference.py and smc_inference.py)
        if use_ema:
            cmd.append("--use_ema")

        if profile:
            cmd.append("--profile")

        # inference.py-specific guidance flags (not used in SMC mode)
        if not use_smc and use_guidance:
            cmd.extend([
                "--use_inference_guidance",
                "--num_candidates", str(num_candidates),
                "--noise_generation_type", noise_type,
                "--num_gpus", str(num_gpus)
            ])
            if use_fixed_first_frame:
                cmd.append("--use_fixed_first_frame")

            # If generate_only, save all candidates without selection
            if generate_only:
                cmd.append("--save_all_candidates")
            elif use_gemini:
                cmd.append("--use_gemini")

        # Free LLM flags (works for both SMC and regular inference)
        if use_free_llm_for_checklist:
            cmd.append("--use_free_llm_for_checklist")
            cmd.extend(["--free_llm_model", free_llm_model])

        # Qwen VL flags (FREE alternative to Gemini for video evaluation)
        if use_qwen_vl:
            cmd.append("--use_qwen_vl")
            cmd.extend(["--qwen_vl_model", qwen_vl_model])
            cmd.extend(["--qwen_vl_fps", str(qwen_vl_fps)])
            cmd.extend(["--qwen_vl_max_pixels", str(qwen_vl_max_pixels)])

        # Stage 1 filter flags (multi-GPU hallucination pre-filter)
        if use_stage1_filter:
            cmd.append("--use_stage1_filter")
            cmd.extend(["--stage1_model", stage1_model])
            if stage1_num_gpus is not None:
                cmd.extend(["--stage1_num_gpus", str(stage1_num_gpus)])
            cmd.extend(["--stage1_workers_per_gpu", str(stage1_workers_per_gpu)])
            cmd.extend(["--stage1_max_iterations", str(stage1_max_iterations)])

        # Stage 3 workers (only for Gemini mode)
        if use_gemini or use_smc:
            cmd.extend(["--max_gemini_stage3_workers", str(max_gemini_stage3_workers)])

        # Run inference for this batch
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()

        # CRITICAL: Set NCCL environment variables for MIG device support
        # MIG slices on same physical GPU cannot use P2P - must use shared memory
        # This applies to both SMC mode and regular multi-GPU guidance
        if (use_smc and num_gpus > 1) or (use_guidance and num_gpus > 1):
            env['NCCL_P2P_DISABLE'] = '1'   # Disable P2P for MIG on same GPU
            env['NCCL_SHM_DISABLE'] = '0'   # Enable shared memory instead
            env['NCCL_DEBUG'] = 'WARN'      # Show NCCL warnings
            print(f"🚀 Running DDP with torchrun ({num_gpus} GPUs): {' '.join(cmd)}")
            print(f"   ⚙️  NCCL_P2P_DISABLE=1 (required for MIG devices on same physical GPU)")
        else:
            print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env)

        if result.returncode != 0:
            print(f"✗ Batch {batch_idx + 1}/{num_batches} failed")
            all_success = False
            break
        else:
            print(f"✓ Batch {batch_idx + 1}/{num_batches} completed successfully")

    # Clean up temporary files if created
    if temp_prompts_file is not None:
        try:
            os.unlink(temp_prompts_file.name)
        except:
            pass
    if temp_extended_file is not None:
        try:
            os.unlink(temp_extended_file.name)
        except:
            pass

    if not all_success:
        print(f"✗ Failed to generate videos for {dimension_name}")
        return False

    print(f"✓ Generated {len(prompt_list) * videos_per_prompt} videos for {dimension_name}")
    return True

def create_custom_vbench_info(video_folder, vbench_path, dimension_name):
    """Create a custom VBench info file that matches only the videos we have"""
    import json
    import glob

    def clean_prompt_for_filename(prompt):
        """Apply the same cleaning logic as inference.py"""
        return prompt.replace('/', '_').replace('\\', '_').replace(':', '_').replace('\n', ' ')

    # Get list of actual video files
    video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
    all_video_basenames = [os.path.splitext(os.path.basename(f))[0] for f in video_files]

    # Extract videos with proper format: prompt-N.mp4
    # Works for both short original prompts and long extended prompts
    video_basenames = []
    for basename in all_video_basenames:
        parts = basename.rsplit('-', 1)
        if len(parts) == 2 and parts[1].isdigit():
            prompt_part = parts[0]
            # Include all videos with proper format (prompt-N.mp4)
            # No length filter needed - works for both short and extended prompts
            video_basenames.append(basename)

    print(f"Found {len(video_files)} total video files, {len(video_basenames)} with proper format in {video_folder}")

    # Load the original VBench info file for metadata
    original_info_file = os.path.join(vbench_path, "vbench", "VBench_full_info.json")
    if not os.path.exists(original_info_file):
        print(f"✗ Original VBench info file not found: {original_info_file}")
        return None

    with open(original_info_file, 'r') as f:
        original_info = json.load(f)

    # Load extended prompts for this dimension
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extended_prompts_file = os.path.join(script_dir, "prompts", "vbench", "extended_prompts_per_dimension", f"{dimension_name}.txt")

    if not os.path.exists(extended_prompts_file):
        print(f"⚠️ Extended prompts file not found: {extended_prompts_file}")
        print(f"   Falling back to original prompts from VBench_full_info.json")
        extended_prompts = []
    else:
        with open(extended_prompts_file, 'r') as f:
            extended_prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(extended_prompts)} extended prompts for dimension {dimension_name}")

    # Create info entries for all videos we have
    # Strategy: Match videos against ORIGINAL prompts (since videos are named with original prompts)
    # Extended prompts are only used for generation, not for filename matching
    custom_info = []
    matched_videos = set()

    # Always match against original prompts from VBench_full_info.json
    for prompt_entry in original_info:
        if isinstance(prompt_entry, dict) and "prompt_en" in prompt_entry:
            if "dimension" not in prompt_entry or dimension_name not in prompt_entry["dimension"]:
                continue

            original_prompt = prompt_entry["prompt_en"]
            cleaned_original = clean_prompt_for_filename(original_prompt)[:100]

            # Look for videos matching this prompt
            matching_videos = [vb for vb in video_basenames if vb.rsplit('-', 1)[0] == cleaned_original]

            if matching_videos:
                if prompt_entry not in custom_info:
                    custom_info.append(prompt_entry)
                    matched_videos.update(matching_videos)

    print(f"Matched {len(custom_info)} prompts from {len(video_files)} videos for dimension {dimension_name}")

    if len(custom_info) == 0:
        print(f"⚠️ Warning: No videos matched to VBench prompts. This might cause evaluation to fail.")
        print(f"   Sample video names: {video_basenames[:3]}")

    # Save custom info file
    custom_info_path = os.path.join(vbench_path, f"custom_{dimension_name}_info.json")
    with open(custom_info_path, 'w') as f:
        json.dump(custom_info, f, indent=2)

    print(f"✓ Created custom info file: {custom_info_path}")
    return custom_info_path

def run_vbench2_evaluation_on_folder(video_folder, vbench_path, output_name, dimension_name):
    """Run VBench 2.0 evaluation on a folder of videos using Python API"""
    import sys
    import torch

    # Convert to absolute paths to avoid path issues
    vbench2_base_path = os.path.join(os.path.abspath(vbench_path), "VBench-2.0")
    video_abs_path = os.path.abspath(video_folder)

    print(f"Running VBench 2.0 evaluation using Python API")
    print(f"Videos path: {video_abs_path}")
    print(f"Dimension: {dimension_name}")
    print(f"Output name: {output_name}")

    # Change to VBench 2.0 directory for evaluation
    original_cwd = os.getcwd()
    original_path = sys.path.copy()

    try:
        os.chdir(vbench2_base_path)
        # Add VBench 2.0 to Python path
        if vbench2_base_path not in sys.path:
            sys.path.insert(0, vbench2_base_path)

        # Import VBench2 after adding to path
        from vbench2 import VBench2

        # Setup VBench2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        full_info_json = os.path.join(vbench2_base_path, "vbench2", "VBench2_full_info.json")
        output_path = os.path.join(vbench2_base_path, "evaluation_results")
        os.makedirs(output_path, exist_ok=True)

        print(f"\n🔄 Starting VBench 2.0 evaluation for dimension: {dimension_name}")
        print(f"   Videos path: {video_abs_path}")
        print(f"   Output name: {output_name}")
        print(f"   Device: {device}")
        print(f"   This may take several minutes...")

        # Initialize VBench2
        my_vbench = VBench2(
            device=device,
            full_info_dir=full_info_json,
            output_path=output_path
        )

        # Evaluate the specified dimension
        my_vbench.evaluate(
            videos_path=video_abs_path,
            name=output_name,
            dimension_list=[dimension_name],
            mode='vbench_standard'
        )

        print(f"\n✓ VBench 2.0 evaluation completed")

        # Look for the expected results file
        expected_eval_results = os.path.join(output_path, f"{output_name}_eval_results.json")

        # Check if file exists
        if os.path.exists(expected_eval_results):
            print(f"✓ Found eval_results file: {expected_eval_results}")
            return expected_eval_results
        else:
            print(f"✗ Expected results file not found: {expected_eval_results}")

            # List what's actually in the evaluation_results directory
            if os.path.exists(output_path):
                print(f"Evaluation results directory contents: {os.listdir(output_path)}")
            else:
                print(f"Evaluation results directory doesn't exist: {output_path}")
            return None

    except Exception as e:
        print(f"✗ VBench 2.0 evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        os.chdir(original_cwd)
        sys.path = original_path


def run_vbench_evaluation_on_folder(video_folder, vbench_path, output_name, dimension_name):
    """Run VBench evaluation on a folder of videos using Python API"""
    import sys
    import torch

    # Convert to absolute paths to avoid path issues
    vbench_abs_path = os.path.abspath(vbench_path)
    video_abs_path = os.path.abspath(video_folder)

    print(f"Running VBench evaluation using Python API")
    print(f"Videos path: {video_abs_path}")
    print(f"Dimension: {dimension_name}")
    print(f"Output name: {output_name}")  # Fixed: was showing misleading double name

    # Change to VBench directory for evaluation
    original_cwd = os.getcwd()
    original_path = sys.path.copy()

    try:
        os.chdir(vbench_abs_path)
        # Add VBench to Python path
        if vbench_abs_path not in sys.path:
            sys.path.insert(0, vbench_abs_path)

        # Import VBench after adding to path
        from vbench import VBench

        # First, run static filter for temporal_flickering dimension only
        if dimension_name == "temporal_flickering":
            print("Running static filter for temporal_flickering dimension...")
            filter_cmd = [
                "python", "static_filter.py",
                "--videos_path", video_abs_path
            ]
            filter_result = subprocess.run(filter_cmd, capture_output=True, text=True)
            if filter_result.returncode != 0:
                print(f"Warning: Static filter failed: {filter_result.stderr}")
            else:
                print("✓ Static filter completed")

        # Use VBench command-line directly
        output_path = os.path.join(vbench_abs_path, "evaluation_results")
        custom_name = output_name

        print(f"\n🔄 Starting VBench evaluation via command-line for dimension: {dimension_name}")
        print(f"   Videos path: {video_abs_path}")
        print(f"   Output name: {custom_name}")
        print(f"   This may take several minutes...")

        # Use VBench's original full JSON (not custom - VBench will match videos by filename)
        original_json = os.path.join(vbench_abs_path, "vbench", "VBench_full_info.json")
        print(f"   Using VBench full info: {original_json}")

        # Build VBench command using evaluate.py script
        evaluate_script = os.path.join(vbench_abs_path, "evaluate.py")
        vbench_cmd = [
            "python", evaluate_script,
            "--videos_path", video_abs_path,
            "--dimension", dimension_name,
            "--output_path", output_path,
            "--full_json_dir", original_json,
            "--mode", "vbench_standard",
            "--custom_name", custom_name
        ]

        print(f"   Running command: {' '.join(vbench_cmd)}")
        result = subprocess.run(
            vbench_cmd,
            capture_output=True,
            text=True,
            cwd=vbench_abs_path,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode == 0:
            print(f"   ✓ VBench command completed successfully")
            if result.stdout:
                print(f"   Output:\n{result.stdout}")
        else:
            print(f"   ✗ VBench command failed with return code {result.returncode}")
            if result.stdout:
                print(f"   stdout:\n{result.stdout}")
            if result.stderr:
                print(f"   stderr:\n{result.stderr}")
            return None

        print(f"\n✓ VBench evaluation completed")

        # Look for the expected files with our custom naming
        eval_results_dir = os.path.join(vbench_abs_path, "evaluation_results")
        expected_eval_results = os.path.join(eval_results_dir, f"{custom_name}_eval_results.json")
        expected_full_info = os.path.join(eval_results_dir, f"{custom_name}_full_info.json")

        # Check if files exist
        if os.path.exists(expected_eval_results):
            print(f"✓ Found eval_results file: {expected_eval_results}")
            return expected_eval_results
        elif os.path.exists(expected_full_info):
            print(f"✓ Found full_info file: {expected_full_info}")
            return expected_full_info
        else:
            print(f"✗ Expected results files not found:")
            print(f"  Expected eval_results: {expected_eval_results}")
            print(f"  Expected full_info: {expected_full_info}")

            # List what's actually in the evaluation_results directory
            if os.path.exists(eval_results_dir):
                print(f"Evaluation results directory contents: {os.listdir(eval_results_dir)}")
            else:
                print(f"Evaluation results directory doesn't exist: {eval_results_dir}")
            return None

    except Exception as e:
        print(f"✗ VBench evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        os.chdir(original_cwd)
        sys.path = original_path

def parse_vbench_results(results_file):
    """Parse VBench results file"""
    if not os.path.exists(results_file):
        print(f"✗ Results file not found: {results_file}")
        return None

    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"✓ Parsed results from {os.path.basename(results_file)}")
        print(f"  Keys: {list(results.keys()) if isinstance(results, dict) else f'Type: {type(results)}'}")
        if isinstance(results, dict) and len(results) > 0:
            first_key = list(results.keys())[0]
            print(f"  Sample content: {first_key} -> {type(results[first_key])}")
        return results
    except Exception as e:
        print(f"✗ Error parsing results: {e}")
        return None

def generate_comprehensive_report(all_results, dimensions, noise_types, videos_per_prompt_info):
    """Generate comprehensive VBench comparison report"""

    print(f"Debug: Generating report with {len(all_results)} result sets")
    print(f"Debug: all_results keys: {list(all_results.keys())}")
    print(f"Debug: dimensions: {dimensions}")
    print(f"Debug: noise_types: {noise_types}")

    # Available dimensions in our evaluation
    available_dims = ['appearance_style', 'color', 'human_action', 'multiple_objects', 'object_class',
                     'overall_consistency', 'scene', 'spatial_relationship', 'subject_consistency',
                     'temporal_flickering', 'temporal_style']

    report = []
    report.append("VBench Official Evaluation Report")
    report.append("=" * 60)
    report.append("Following exact VBench sampling protocol:")
    report.append("- Sample videos from all txt files in prompts/prompts_per_dimension")
    report.append("- 5 videos per prompt (25 for temporal_flickering)")
    report.append("- Random seeds recorded for reproducibility")
    report.append("- Naming: prompt-index.mp4")
    report.append("=" * 60)
    report.append("")

    # Sample statistics
    report.append("Sampling Statistics:")
    for dim, count in videos_per_prompt_info.items():
        report.append(f"  • {dim}: {count} videos per prompt")
    report.append("")

    # VBench standard metrics for comparison
    vbench_metrics = [
        'subject_consistency', 'background_consistency', 'temporal_flickering',
        'motion_smoothness', 'dynamic_degree', 'aesthetic_quality',
        'imaging_quality', 'object_class', 'multiple_objects',
        'human_action', 'color', 'spatial_relationship', 'scene'
    ]

    # Per-dimension analysis
    report.append("Per-Dimension Results:")
    report.append("-" * 80)
    report.append(f"{'Dimension':<20} {'Baseline':<12} {'Best Guided':<12} {'Improvement':<12} {'Best Type':<15}")
    report.append("-" * 80)

    improvements = []
    for dimension in dimensions:
        baseline_key = f"baseline_{dimension}"
        if baseline_key not in all_results:
            continue

        baseline_results = all_results[baseline_key]

        # Handle VBench result formats
        if isinstance(baseline_results, dict) and dimension in baseline_results:
            # Case 1: Direct dimension key in dict
            dimension_data = baseline_results[dimension]
            if isinstance(dimension_data, list) and len(dimension_data) > 0:
                # VBench format: dimension_data[0] is the overall score
                baseline_score = dimension_data[0]
            else:
                baseline_score = dimension_data
        elif isinstance(baseline_results, dict):
            # Case 2: Fallback to overall_score
            baseline_score = baseline_results.get('overall_score', 0)
        elif isinstance(baseline_results, list) and len(baseline_results) > 0:
            # Case 3: Results is directly a list (shouldn't happen with current format)
            baseline_score = baseline_results[0] if isinstance(baseline_results[0], (int, float)) else 0
        else:
            print(f"Warning: Unexpected baseline_results format for {dimension}: {type(baseline_results)}")
            baseline_score = 0

        # Ensure baseline_score is a number
        if isinstance(baseline_score, (list, tuple)) and len(baseline_score) > 0:
            baseline_score = baseline_score[0]
        elif not isinstance(baseline_score, (int, float)):
            print(f"Warning: baseline_score is not numeric for {dimension}: {baseline_score} ({type(baseline_score)})")
            baseline_score = 0
        best_score = baseline_score
        best_type = "baseline"

        # Find best performing noise type for this dimension
        for noise_type in noise_types:
            guided_key = f"guided_{noise_type}_{dimension}"
            if guided_key in all_results:
                guided_results = all_results[guided_key]

                # Handle VBench result formats for guided results
                if isinstance(guided_results, dict) and dimension in guided_results:
                    # Case 1: Direct dimension key in dict
                    dimension_data = guided_results[dimension]
                    if isinstance(dimension_data, list) and len(dimension_data) > 0:
                        # VBench format: dimension_data[0] is the overall score
                        score = dimension_data[0]
                    else:
                        score = dimension_data
                elif isinstance(guided_results, dict):
                    # Case 2: Fallback to overall_score
                    score = guided_results.get('overall_score', 0)
                elif isinstance(guided_results, list) and len(guided_results) > 0:
                    # Case 3: Results is directly a list
                    score = guided_results[0] if isinstance(guided_results[0], (int, float)) else 0
                else:
                    score = 0

                # Ensure score is a number
                if isinstance(score, (list, tuple)) and len(score) > 0:
                    score = score[0]
                elif not isinstance(score, (int, float)):
                    score = 0
                if score > best_score:
                    best_score = score
                    best_type = noise_type

        improvement = ((best_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
        improvements.append(improvement)

        report.append(f"{dimension:<20} {baseline_score:<12.4f} {best_score:<12.4f} {improvement:<12.2f}% {best_type:<15}")

    report.append("-" * 80)
    report.append("")

    # Simple comparison focused on guidance improvements
    report.append("Guidance vs Baseline Comparison:")
    report.append("-" * 80)
    report.append("")

    # Summary statistics
    positive_improvements = [imp for imp in improvements if imp > 0]
    negative_improvements = [imp for imp in improvements if imp < 0]

    report.append("🎯 Guidance Impact Summary:")
    report.append("-" * 50)
    report.append(f"📊 Total dimensions evaluated: {len(improvements)}")
    report.append(f"✅ Dimensions improved by guidance: {len(positive_improvements)}")
    report.append(f"❌ Dimensions degraded by guidance: {len(negative_improvements)}")

    if positive_improvements:
        avg_improvement = sum(positive_improvements)/len(positive_improvements)
        max_improvement = max(positive_improvements)
        report.append(f"📈 Average improvement: {avg_improvement:.2f}%")
        report.append(f"🔥 Maximum improvement: {max_improvement:.2f}%")

        # Find which dimension had the best improvement
        best_dim_idx = improvements.index(max_improvement)
        best_dim = dimensions[best_dim_idx] if best_dim_idx < len(dimensions) else "unknown"
        report.append(f"🏆 Best performing dimension: {best_dim} (+{max_improvement:.2f}%)")

    if len(positive_improvements) > len(negative_improvements):
        report.append(f"✨ Overall result: Guidance IMPROVED performance on {len(positive_improvements)}/{len(improvements)} dimensions")
    else:
        report.append(f"⚠️ Overall result: Guidance needs improvement - helped only {len(positive_improvements)}/{len(improvements)} dimensions")

    report.append("")

    # Best performing noise types
    noise_type_wins = {noise_type: 0 for noise_type in noise_types}
    for dimension in dimensions:
        baseline_key = f"baseline_{dimension}"
        if baseline_key not in all_results:
            continue

        # Extract baseline score (handle VBench result format)
        baseline_results = all_results[baseline_key]
        if isinstance(baseline_results, dict) and dimension in baseline_results:
            baseline_data = baseline_results[dimension]
            if isinstance(baseline_data, list) and len(baseline_data) > 0:
                baseline_score = baseline_data[0]
            else:
                baseline_score = baseline_data
        else:
            baseline_score = 0

        # Ensure baseline_score is numeric
        if not isinstance(baseline_score, (int, float)):
            baseline_score = 0

        best_score = baseline_score
        best_type = None

        for noise_type in noise_types:
            guided_key = f"guided_{noise_type}_{dimension}"
            if guided_key in all_results:
                guided_results = all_results[guided_key]

                # Extract guided score (handle VBench result format)
                if isinstance(guided_results, dict) and dimension in guided_results:
                    guided_data = guided_results[dimension]
                    if isinstance(guided_data, list) and len(guided_data) > 0:
                        score = guided_data[0]
                    else:
                        score = guided_data
                else:
                    score = 0

                # Ensure score is numeric
                if not isinstance(score, (int, float)):
                    score = 0

                if score > best_score:
                    best_score = score
                    best_type = noise_type

        if best_type:
            noise_type_wins[best_type] += 1

    report.append("Best Performing Noise Types:")
    for noise_type, wins in sorted(noise_type_wins.items(), key=lambda x: x[1], reverse=True):
        report.append(f"  • {noise_type}: best in {wins} dimensions")

    return "\n".join(report)

def main():
    args = parse_args()

    # Validate workflow flags
    if args.generate_only and args.select_only:
        print("❌ Error: Cannot use both --generate_only and --select_only")
        sys.exit(1)

    if args.select_only and not args.videos_dir:
        print("❌ Error: --select_only requires --videos_dir to be specified")
        sys.exit(1)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Define VBench 2.0 dimensions (physics and advanced reasoning)
    vbench2_dimensions = [
        "Mechanics", "Thermotics", "Material",  # Physics Understanding
        "Human_Anatomy", "Human_Identity", "Human_Clothes",  # Human Capability
        "Complex_Landscape", "Complex_Plot", "Composition",  # Scene Understanding
        "Motion_Rationality", "Motion_Order_Understanding", "Human_Interaction",  # Motion Reasoning
        "Dynamic_Spatial_Relationship", "Dynamic_Attribute", "Multi-View_Consistency",
        "Instance_Preservation", "Diversity",  # Visual Consistency
        "Camera_Motion"  # Camera Control
    ]

    # Print workflow mode
    if args.generate_only:
        print("🎬 GENERATE ONLY MODE: Will generate candidates and save to videos/ directory")
        print("=" * 60)
    elif args.select_only:
        print("🎯 SELECT ONLY MODE: Will select best from pre-generated videos")
        print("=" * 60)
        print(f"Reading from: {args.videos_dir}")
    else:
        print("VBench Official Evaluation Following Exact Protocol")
        print("=" * 60)
        print("Protocol:")
        print(f"- Using ORIGINAL prompts for filenames (short, clean)")
        print(f"- Using EXTENDED prompts for generation (detailed, high quality)")
        if args.vbench2:
            print("- VBench 2.0: 3 videos per prompt (20 for Diversity)")
        else:
            print("- VBench 1.0: 5 videos per prompt (25 for temporal_flickering)")

        # Show which conditions are being generated
        if args.skip_baseline and args.skip_guided:
            print("⚠️  WARNING: Both baseline and guided generation are skipped!")
        elif args.skip_baseline:
            print("- ONLY use_guidance condition (baseline skipped)")
        elif args.skip_guided:
            print("- ONLY baseline condition (guided skipped)")
        else:
            print("- Both baseline and use_guidance conditions")

    print("- Random seeds recorded for reproducibility")
    print("- Naming: prompt-index.mp4")
    print(f"Max prompts per category: {args.max_prompts_per_category}")

    # Show sampling method
    if args.use_smc:
        print(f"🔬 Sampling method: SMC (Sequential Monte Carlo)")
        print(f"   - Block sizes: {args.smc_block_sizes}")
        print(f"   - Initial particles: {args.smc_initial_particles}")
        print(f"   - Top-k: {args.smc_top_k}, Branch factor: {args.smc_branch_factor}")
        print(f"   - Uses Gemini for intermediate + final evaluation")
    else:
        print(f"Sampling method: Regular inference")
        print(f"   - Noise generation type: {args.noise_generation_type}")
        print(f"   - Number of candidates: {args.num_candidates}")

    profiling_status = "ENABLED" if args.profile else "DISABLED"
    print(f"Profiling: {profiling_status}")
    print("=" * 60)

    all_results = {}
    videos_per_prompt_info = {}

    if args.evaluate_all_dimensions:
        # Evaluate using all_dimension.txt (946 prompts)
        print("\n🎯 Using all_dimension.txt for evaluation")
        all_dimension_file = os.path.join(args.vbench_path, "prompts", "all_dimension.txt")
        all_dimension_extended_file = os.path.join(args.vbench_path, "prompts", "all_dimension_extended.txt")
        dimensions = ["all_dimensions"]
        # Store as (name, original_file, extended_file)
        dimension_files = [("all_dimensions", all_dimension_file, all_dimension_extended_file)]
    else:
        # Get both original and extended files for each dimension
        # We ALWAYS want to use original names for filenames but extended for generation
        script_dir = Path(__file__).parent

        if args.vbench2:
            # VBench 2.0 paths - look in VBench installation directory
            vbench_base = Path(args.vbench_path)
            original_prompts_dir = vbench_base / "VBench-2.0" / "prompts" / "prompt"
            extended_prompts_dir = vbench_base / "VBench-2.0" / "prompts" / "prompt_aug" / "wanx_aug_prompt"
        else:
            # VBench 1.0 paths (original)
            original_prompts_dir = script_dir / "prompts" / "vbench" / "prompts"
            extended_prompts_dir = script_dir / "prompts" / "vbench" / "extended_prompts_per_dimension"

        # Get all dimension names from original prompts directory
        original_files = sorted(original_prompts_dir.glob("*.txt"))
        dimension_files = []

        for original_file in original_files:
            dim_name = original_file.stem
            extended_file = extended_prompts_dir / f"{dim_name}.txt"

            if extended_file.exists():
                # Store as (name, original_file, extended_file)
                dimension_files.append((dim_name, str(original_file), str(extended_file)))
            else:
                print(f"⚠️ Warning: No extended prompts for {dim_name}, using original only")
                dimension_files.append((dim_name, str(original_file), None))

        dimensions = [dim_name for dim_name, _, _ in dimension_files]

    # Filter dimensions if specific ones are requested
    if args.dimensions:
        filtered_files = [
            (name, orig, ext) for name, orig, ext in dimension_files
            if name in args.dimensions
        ]
        if len(filtered_files) == 0:
            print(f"❌ Error: None of the requested dimensions {args.dimensions} were found.")
            print(f"   Available dimensions: {dimensions}")
            return
        dimension_files = filtered_files
        dimensions = [dim_name for dim_name, _, _ in dimension_files]
        print(f"\n📌 Filtering to specific dimensions: {dimensions}")

    # Process each dimension file
    for dimension_name, original_file, extended_file in dimension_files:
        print(f"\n🎯 Processing dimension: {dimension_name}")
        print("-" * 60)

        videos_per_prompt = 25 if dimension_name == "temporal_flickering" else 5
        videos_per_prompt_info[dimension_name] = videos_per_prompt

        # Handle different workflow modes
        if args.select_only:
            # SELECT ONLY MODE: Read pre-generated videos from --videos_dir
            print(f"\n🎯 SELECT ONLY MODE: Selecting best from {args.videos_dir}")

            # Find all candidate videos in the directory
            import glob
            candidate_pattern = os.path.join(args.videos_dir, "rank*_candidate_*.mp4")
            all_candidates = sorted(glob.glob(candidate_pattern))

            if not all_candidates:
                print(f"❌ No candidate videos found in {args.videos_dir}")
                print(f"   Expected pattern: rank*_candidate_*.mp4")
                continue

            print(f"   Found {len(all_candidates)} candidate videos")

            # Group candidates by prompt (batch_idx)
            from collections import defaultdict
            candidates_by_batch = defaultdict(list)

            for video_path in all_candidates:
                basename = os.path.basename(video_path)
                parts = basename.split('_')
                batch_idx = 0
                for i, part in enumerate(parts):
                    if part == 'batch' and i + 1 < len(parts):
                        try:
                            batch_idx = int(parts[i + 1])
                            break
                        except:
                            pass
                candidates_by_batch[batch_idx].append(video_path)

            print(f"   Grouped into {len(candidates_by_batch)} batch(es) by filename")

            # Read prompts
            with open(original_file, 'r') as f:
                prompts_list = [line.strip() for line in f if line.strip()]

            if extended_file and os.path.exists(extended_file):
                with open(extended_file, 'r') as f:
                    extended_prompts_list = [line.strip() for line in f if line.strip()]
            else:
                extended_prompts_list = None

            # Initialize scorer
            import torch
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            from pipeline.vlm_scorer import MultiObjectiveScorer
            scorer = MultiObjectiveScorer(
                device=device,
                use_gemini=args.use_gemini,
                use_qwen_vl=args.use_qwen_vl,
                qwen_vl_model=args.qwen_vl_model,
                qwen_vl_fps=args.qwen_vl_fps,
                qwen_vl_max_pixels=args.qwen_vl_max_pixels,
                use_stage1_model=args.use_stage1_filter,
                stage1_model=args.stage1_model,
                use_free_llm_for_checklist=args.use_free_llm_for_checklist,
                free_llm_model=args.free_llm_model
            )

            # Process each batch
            # Use modulo so multiple generation batches for the same prompt all use the correct prompt
            for batch_idx, candidate_paths in sorted(candidates_by_batch.items()):
                prompt_idx = batch_idx % len(prompts_list)
                prompt = prompts_list[prompt_idx]
                evaluation_prompt = extended_prompts_list[prompt_idx] if extended_prompts_list else prompt

                print(f"\n   📝 Batch {batch_idx} (prompt {prompt_idx}): {prompt[:80]}...")
                print(f"      Candidates: {len(candidate_paths)}")

                # Generate checklist
                if args.use_gemini:
                    checklist = scorer.generate_prompt_alignment_checklist(
                        text_prompt=evaluation_prompt,
                        model_name="gemini-2.5-flash",
                        max_output_tokens=1500,
                        temperature=0.1
                    )
                else:
                    checklist = scorer.generate_prompt_alignment_checklist(text_prompt=evaluation_prompt)

                # Free the 32B text LLM — checklist is already a string, no longer needed.
                # Reclaims ~20GB VRAM before Stage 1 loads Qwen2.5-VL-7B + 14B tournament model.
                # Will lazy-reload next iteration if there are more prompts.
                if args.use_stage1_filter:
                    scorer.free_text_llm()

                # Run selection
                results = scorer.multi_stage_candidate_selection(
                    candidate_paths=candidate_paths,
                    text_prompt=evaluation_prompt,
                    prompt_alignment_checklist=checklist,
                    use_stage0=False,
                    use_stage1_filter=args.use_stage1_filter,
                    stage1_model_name=args.stage1_model,
                    stage1_num_gpus=args.stage1_num_gpus,
                    stage1_workers_per_gpu=args.stage1_workers_per_gpu,
                    stage1_max_iterations=args.stage1_max_iterations
                )

                winner_path = results['winner_path']
                winner_idx = results['winner_index']

                # Save best video
                model = "regular" if not args.use_ema else "ema"
                clean_prompt = prompt[:180].replace('/', '_').replace('\\', '_').replace(':', '_')
                best_filename = f'{clean_prompt}_{model}.mp4'
                best_output_path = os.path.join(args.output_dir, dimension_name, "selected", best_filename)
                os.makedirs(os.path.dirname(best_output_path), exist_ok=True)

                import shutil
                shutil.copy2(winner_path, best_output_path)

                print(f"      🏆 Winner: candidate_{winner_idx}")
                print(f"      📁 Saved: {best_output_path}")

            print(f"\n✅ Selection complete for {dimension_name}")
            continue

        if args.generate_only:
            # GENERATE ONLY MODE: Generate candidates and save to videos/ directory
            print(f"\n🎬 GENERATE ONLY MODE: Generating candidates for {dimension_name}...")
            # Set a flag to skip evaluation later
            # The normal generation code below will run, but we'll exit before evaluation

        if not args.evaluation_only:
            # 1. Generate baseline videos (no guidance)
            if not args.skip_baseline:
                print(f"\n1. Generating baseline videos for {dimension_name}...")
                print(f"   Using original prompts for filenames: {Path(original_file).name}")
                if extended_file:
                    print(f"   Using extended prompts for generation: {Path(extended_file).name}")
                success_baseline = sample_videos_for_dimension(
                    config_path=args.config_path,
                    checkpoint_path=args.checkpoint_path,
                    dimension_name=dimension_name,
                    prompts_file=original_file,
                    base_output_dir=str(output_dir),
                    use_guidance=False,
                    base_seed=args.seed,
                    max_prompts_per_category=args.max_prompts_per_category,
                    use_ema=args.use_ema,
                    profile=args.profile,
                    extended_prompts_file=extended_file,
                    use_fixed_first_frame=args.use_fixed_first_frame,
                    use_gemini=args.use_gemini,
                    use_smc=args.use_smc,
                    smc_block_sizes=args.smc_block_sizes,
                    smc_initial_particles=args.smc_initial_particles,
                    smc_top_k=args.smc_top_k,
                    smc_branch_factor=args.smc_branch_factor,
                    max_gemini_stage3_workers=args.max_gemini_stage3_workers,
                    use_free_llm_for_checklist=args.use_free_llm_for_checklist,
                    free_llm_model=args.free_llm_model,
                    use_qwen_vl=args.use_qwen_vl,
                    qwen_vl_model=args.qwen_vl_model,
                    qwen_vl_fps=args.qwen_vl_fps,
                    qwen_vl_max_pixels=args.qwen_vl_max_pixels,
                    use_stage1_filter=args.use_stage1_filter,
                    stage1_model=args.stage1_model,
                    stage1_num_gpus=args.stage1_num_gpus,
                    stage1_workers_per_gpu=args.stage1_workers_per_gpu,
                    stage1_max_iterations=args.stage1_max_iterations,
                    generate_only=args.generate_only
                )
            else:
                print(f"\n1. ⏭️  Skipping baseline generation (--skip_baseline)")
                success_baseline = True

            # 2. Generate guided videos (with guidance)
            if not args.skip_guided:
                print(f"\n2. Generating guided videos for {dimension_name}...")
                print(f"   Using original prompts for filenames: {Path(original_file).name}")
                if extended_file:
                    print(f"   Using extended prompts for generation: {Path(extended_file).name}")
                success_guided = sample_videos_for_dimension(
                    config_path=args.config_path,
                    checkpoint_path=args.checkpoint_path,
                    dimension_name=dimension_name,
                    prompts_file=original_file,
                    base_output_dir=str(output_dir),
                    use_guidance=True,
                    num_candidates=args.num_candidates,
                    noise_type=args.noise_generation_type,
                    base_seed=args.seed,
                    max_prompts_per_category=args.max_prompts_per_category,
                    use_ema=args.use_ema,
                    profile=args.profile,
                    extended_prompts_file=extended_file,
                    num_gpus=args.num_gpus,
                    use_fixed_first_frame=args.use_fixed_first_frame,
                    use_gemini=args.use_gemini,
                    use_smc=args.use_smc,
                    smc_block_sizes=args.smc_block_sizes,
                    smc_initial_particles=args.smc_initial_particles,
                    smc_top_k=args.smc_top_k,
                    smc_branch_factor=args.smc_branch_factor,
                    max_gemini_stage3_workers=args.max_gemini_stage3_workers,
                    use_free_llm_for_checklist=args.use_free_llm_for_checklist,
                    free_llm_model=args.free_llm_model,
                    use_qwen_vl=args.use_qwen_vl,
                    qwen_vl_model=args.qwen_vl_model,
                    qwen_vl_fps=args.qwen_vl_fps,
                    qwen_vl_max_pixels=args.qwen_vl_max_pixels,
                    use_stage1_filter=args.use_stage1_filter,
                    stage1_model=args.stage1_model,
                    stage1_num_gpus=args.stage1_num_gpus,
                    stage1_workers_per_gpu=args.stage1_workers_per_gpu,
                    stage1_max_iterations=args.stage1_max_iterations,
                    generate_only=args.generate_only
                )
            else:
                print(f"\n2. ⏭️  Skipping guided generation (--skip_guided)")
                success_guided = True

            if not success_baseline or not success_guided:
                print(f"✗ Failed to generate videos for {dimension_name}")
                continue
        else:
            print(f"\n⏭️ Skipping video generation for {dimension_name} (evaluation_only=True)")
            success_baseline = True
            success_guided = True

        # If generate_only mode, skip evaluation
        if args.generate_only:
            print(f"\n✅ GENERATE ONLY: Videos generated for {dimension_name}")
            print(f"   📁 Videos saved to: {output_dir / dimension_name}")
            print(f"\n💡 To select best videos, run with --select_only --videos_dir <path>")
            continue

        # 3. Evaluate baseline
        if not args.skip_baseline:
            print(f"\n3. Evaluating baseline for {dimension_name}...")
            baseline_dir = os.path.join(str(output_dir), dimension_name, f"baseline_{args.num_candidates}")

            # Check if this is a VBench 2.0 dimension
            if dimension_name in vbench2_dimensions:
                print(f"   Using VBench 2.0 evaluation for {dimension_name}")
                baseline_results_file = run_vbench2_evaluation_on_folder(
                    video_folder=baseline_dir,
                    vbench_path=args.vbench_path,
                    output_name=f"baseline_{dimension_name}",
                    dimension_name=dimension_name
                )
            else:
                print(f"   Using VBench 1.0 evaluation for {dimension_name}")
                baseline_results_file = run_vbench_evaluation_on_folder(
                    video_folder=baseline_dir,
                    vbench_path=args.vbench_path,
                    output_name=f"baseline_{dimension_name}",
                    dimension_name=dimension_name
                )

            if baseline_results_file:
                baseline_results = parse_vbench_results(baseline_results_file)
                all_results[f'baseline_{dimension_name}'] = baseline_results
                print(f"✓ Baseline evaluation completed for {dimension_name}")
        else:
            print(f"\n3. ⏭️  Skipping baseline evaluation (--skip_baseline)")

        # 4. Evaluate guided videos
        if not args.skip_guided:
            print(f"\n4. Evaluating guided videos for {dimension_name}...")
            guided_dir = os.path.join(str(output_dir), dimension_name, f"use_guidance_{args.num_candidates}")

            # Check if this is a VBench 2.0 dimension
            if dimension_name in vbench2_dimensions:
                print(f"   Using VBench 2.0 evaluation for {dimension_name}")
                guided_results_file = run_vbench2_evaluation_on_folder(
                    video_folder=guided_dir,
                    vbench_path=args.vbench_path,
                    output_name=f"guided_{dimension_name}",
                    dimension_name=dimension_name
                )
            else:
                print(f"   Using VBench 1.0 evaluation for {dimension_name}")
                guided_results_file = run_vbench_evaluation_on_folder(
                    video_folder=guided_dir,
                    vbench_path=args.vbench_path,
                    output_name=f"guided_{dimension_name}",
                    dimension_name=dimension_name
                )

            if guided_results_file:
                guided_results = parse_vbench_results(guided_results_file)
                all_results[f'guided_{args.noise_generation_type}_{dimension_name}'] = guided_results
                print(f"✓ Guided evaluation completed for {dimension_name}")
        else:
            print(f"\n4. ⏭️  Skipping guided evaluation (--skip_guided)")

    # 5. Generate comprehensive report
    print("\n5. Generating comprehensive report...")
    comparison_report = generate_comprehensive_report(
        all_results, dimensions, [args.noise_generation_type], videos_per_prompt_info
    )

    # Save results
    results_file = output_dir / "vbench_official_evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    report_file = output_dir / "vbench_official_evaluation_report.txt"
    with open(report_file, 'w') as f:
        f.write(comparison_report)

    print(f"\n✓ Evaluation completed!")
    print(f"Results: {results_file}")
    print(f"Report: {report_file}")
    print("\n" + "=" * 60)
    print("VBENCH OFFICIAL EVALUATION SUMMARY")
    print("=" * 60)
    print(comparison_report)

if __name__ == "__main__":
    main()