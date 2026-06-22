import argparse
import os
import sys

# ═══════════════════════════════════════════════════════════════════════════
# CRITICAL: Set NCCL environment variables BEFORE any torch imports
# This must be done at the very top, before torch.distributed is imported
# MIG slices on the same physical GPU cannot use P2P communication
# ═══════════════════════════════════════════════════════════════════════════
os.environ.setdefault('NCCL_P2P_DISABLE', '1')   # Disable P2P for MIG on same GPU
os.environ.setdefault('NCCL_SHM_DISABLE', '0')   # Enable shared memory instead
os.environ.setdefault('NCCL_DEBUG', 'WARN')      # Show NCCL warnings

import torch
import time
import threading
from queue import Queue
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from pipeline.sampling import NoiseSampler
from pipeline.vlm_scorer import MultiObjectiveScorer
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--start_index", type=int, default=0, help="Starting index for saved video filenames (for batch processing)")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--use_inference_guidance", action="store_true",
                    help="Whether to use inference guidance with random noise search")
parser.add_argument("--num_candidates", type=int, default=5,
                    help="Number of candidate pairs to generate during random noise search")
parser.add_argument("--save_intermediates", action="store_true",
                    help="Save intermediate denoising predictions for analysis")
parser.add_argument("--save_all_candidates", action="store_true",
                    help="Save all candidate videos without selection (for generate-only mode)")
parser.add_argument("--noise_generation_type", type=str, default="mixed",
                    choices=["mixed", "scale_variations", "pure_random", "edge_cases",
                            "frequency_variations", "spatial_correlations", "temporal_correlations",
                            "low_discrepancy"],
                    help="Type of noise generation pattern to use")
parser.add_argument("--profile", action="store_true",
                    help="Enable profiling to measure VAE decoding and denoising time")
parser.add_argument("--num_gpus", type=int, default=1,
                    help="Number of worker GPUs for parallel candidate generation. Total GPUs needed: num_gpus+1 (GPU 0 reserved for scorer). Example: --num_gpus=4 requires 5 GPUs total.")
parser.add_argument("--use_fixed_first_frame", action="store_true",
                    help="Fix the first frame and only vary subsequent frames across candidates")
parser.add_argument("--use_gemini", action="store_true",
                    help="Use Gemini API for physics-aware video comparison instead of VideoLLaMA3")
parser.add_argument("--max_gemini_stage1_workers", type=int, default=5,
                    help="Max concurrent Gemini API calls for Stage 1 (default: 5)")
parser.add_argument("--max_gemini_stage2_workers", type=int, default=5,
                    help="Max concurrent Gemini API calls for Stage 2 (default: 3)")
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
                    help="Frame sampling rate for Qwen3-VL video processing (default: 1.0 fps)")
parser.add_argument("--qwen_vl_max_pixels", type=int, default=480*720,
                    help="Maximum pixels per frame for Qwen3-VL (default: 345600 = 480*720)")
parser.add_argument("--use_stage1_filter", action="store_true",
                    help="Enable Stage 1 hallucination filter (frame checking + temporal consistency) with multi-GPU support")
parser.add_argument("--stage1_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                    help="Model for Stage 1 filtering (default: Qwen/Qwen3-VL-8B-Instruct)")
parser.add_argument("--stage1_num_gpus", type=int, default=None,
                    help="Number of GPUs for Stage 1 filtering (default: auto-detect all available GPUs)")
parser.add_argument("--stage1_workers_per_gpu", type=int, default=5,
                    help="Worker threads per GPU for Stage 1 (default: 5)")
parser.add_argument("--stage1_max_iterations", type=int, default=5,
                    help="Max convergence iterations for Stage 1 filtering (default: 5)")

def gemini_consumer_thread(gemini_queue, result_queue, scorer, args):
    """
    Background thread that consumes video batches and runs Gemini evaluation.
    Runs on CPU (rank 0 only), independent of GPU/NCCL operations.
    """
    print(f"[Rank 0 - CPU] 🔄 Gemini consumer thread started")

    while True:
        # Wait for next batch
        batch_info = gemini_queue.get()

        if batch_info is None:  # Poison pill to stop
            print(f"[Rank 0 - CPU] 🛑 Gemini consumer thread stopping")
            break

        prompt_idx, batch_idx, video_paths, prompt, prompt_alignment_checklist, final_output_path = batch_info

        try:
            print(f"\n[Rank 0 - CPU] {'='*80}")
            print(f"[Rank 0 - CPU] 🎬 Prompt {prompt_idx}, Batch {batch_idx}: Evaluating {len(video_paths)} candidates")
            print(f"[Rank 0 - CPU] {'='*80}\n")

            # Run parallel 3-stage evaluation with detailed reporting
            result = scorer.run_3stage_pipeline_parallel(
                video_paths=video_paths,
                text_prompt=prompt,
                prompt_alignment_checklist=prompt_alignment_checklist,
                max_stage1_workers=args.max_gemini_stage1_workers,
                max_stage2_workers=args.max_gemini_stage2_workers,
                max_stage3_workers=args.max_gemini_stage3_workers,
                return_detailed=True
            )

            winner_path = result['winner_path']

            # === GENERATE STATISTICS REPORT ===
            # Include both prompt_idx and batch_idx to avoid overwriting when batch_size > 1
            stats_output_dir = os.path.join(os.path.dirname(video_paths[0]), "pipeline_stats")
            # Combine prompt and batch indices: prompt0000_batch0, prompt0000_batch1, etc.
            combined_idx = f"{prompt_idx:04d}_batch{batch_idx}"
            scorer.generate_pipeline_statistics_report(
                detailed_results=result,
                total_videos=len(video_paths),
                output_dir=stats_output_dir,
                prompt_idx=combined_idx
            )

            # === STAGE TIMINGS ===
            print(f"\n[Rank 0 - CPU] ⏱️  Stage Timings:")
            print(f"[Rank 0 - CPU]    Stage 1 (Filter):     {result.get('stage1_time', 0):.2f}s")
            print(f"[Rank 0 - CPU]    Stage 2 (SCRIBE):     {result.get('stage2_time', 0):.2f}s")
            print(f"[Rank 0 - CPU]    Stage 3 (Tournament): {result.get('stage3_time', 0):.2f}s")
            total_time = result.get('stage1_time', 0) + result.get('stage2_time', 0) + result.get('stage3_time', 0)
            print(f"[Rank 0 - CPU]    Total:                {total_time:.2f}s\n")

            # === STAGE 1 DETAILED RESULTS ===
            retained_videos = result.get('retained_videos', [])
            rejected_videos = result.get('rejected_videos', [])

            print(f"[Rank 0 - CPU] 📊 Stage 1 Results: {len(retained_videos)} PASSED, {len(rejected_videos)} FAILED\n")

            # Show PASSED videos with FULL responses
            if retained_videos:
                print(f"[Rank 0 - CPU] {'='*80}")
                print(f"[Rank 0 - CPU] ✅ PASSED Videos (Full Reports)")
                print(f"[Rank 0 - CPU] {'='*80}\n")
                for rv in retained_videos:
                    print(f"[Rank 0 - CPU] --- candidate_{rv['candidate_num']} ---")
                    print(f"{rv.get('response', 'No response')}")
                    print(f"\n")

            # Show FAILED videos with FULL responses
            if rejected_videos:
                print(f"[Rank 0 - CPU] {'='*80}")
                print(f"[Rank 0 - CPU] ❌ FAILED Videos (Full Reports)")
                print(f"[Rank 0 - CPU] {'='*80}\n")
                for rv in rejected_videos:
                    print(f"[Rank 0 - CPU] --- candidate_{rv['candidate_num']} ---")
                    print(f"{rv.get('response', 'No response')}")
                    print(f"\n")

            # === STAGE 2 FORENSIC REPORTS (FULL) ===
            forensic_reports = result.get('forensic_reports', [])
            if forensic_reports:
                print(f"[Rank 0 - CPU] {'='*80}")
                print(f"[Rank 0 - CPU] 📝 Stage 2 Forensic Reports (Full)")
                print(f"[Rank 0 - CPU] {'='*80}\n")
                for report in forensic_reports:
                    print(f"[Rank 0 - CPU] --- candidate_{report['candidate_num']} ({len(report['forensic_log'])} chars) ---")
                    print(f"{report['forensic_log']}")
                    print(f"\n")

            # Delete losers (keep only winner)
            # TODO: Could optimize with multithreading if deletion becomes bottleneck
            deleted_count = 0
            for video_path in video_paths:
                if video_path != winner_path and os.path.exists(video_path):
                    os.remove(video_path)
                    deleted_count += 1

            # Rename winner to final output path
            if winner_path and os.path.exists(winner_path):
                os.rename(winner_path, final_output_path)
                print(f"[Rank 0 - CPU] 📝 Renamed: {os.path.basename(winner_path)} → {os.path.basename(final_output_path)}")

            print(f"\n[Rank 0 - CPU] {'='*80}")
            print(f"[Rank 0 - CPU] ✅ Batch {batch_idx} COMPLETE")
            print(f"[Rank 0 - CPU]    Output: {os.path.basename(final_output_path)}")
            print(f"[Rank 0 - CPU]    Deleted: {deleted_count} losing videos")
            print(f"[Rank 0 - CPU] {'='*80}\n")

            # Put result back
            result_queue.put((batch_idx, final_output_path))
        except Exception as e:
            print(f"[Rank 0 - CPU] ❌ Error processing batch {batch_idx}: {e}")
            # Put error result
            result_queue.put((batch_idx, None))

        finally:
            # Mark this batch as done
            gemini_queue.task_done()

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize distributed inference
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        # Set device BEFORE init_process_group - this is the correct way for MIG devices
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        # Debug: Print device info before initialization
        print(f"[Rank {local_rank}] Using device: cuda:{local_rank} = {torch.cuda.get_device_name(local_rank)}")
        print(f"[Rank {local_rank}] Current device: {torch.cuda.current_device()}")

        # For Gemini API scoring with many candidates (e.g., 100), rank 0 may take 15-20 minutes
        # Increase NCCL timeout to prevent other ranks from timing out while waiting
        # Default is 10 minutes (600s), we set to 30 minutes (1800s)
        import datetime
        timeout_duration = datetime.timedelta(seconds=1800)  # 30 minutes

        # Initialize process group
        # CRITICAL: Specify device_id explicitly to prevent NCCL from auto-detecting topology
        # Without this, NCCL sees both MIG slices as "duplicate" on same PCI bus (1b000)
        # Note: NCCL environment variables are also set at the top of this file
        dist.init_process_group(
            backend='nccl',
            timeout=timeout_duration,
            device_id=local_rank  # Explicitly tell NCCL which device this rank uses
        )
        world_size = dist.get_world_size()
        print(f"[Rank {local_rank}] Successfully initialized process group")
        set_seed(args.seed + local_rank)
    else:
        device = torch.device("cuda")
        local_rank = 0
        world_size = 1
        set_seed(args.seed)

    print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
    low_memory = get_cuda_free_memory_gb(device) < 40

    torch.set_grad_enabled(False)

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    # Check if using inference guidance
    use_guidance = getattr(args, 'use_inference_guidance', False) or getattr(config, 'use_inference_guidance', False)
    num_candidates = getattr(args, 'num_candidates', getattr(config, 'num_candidates', 5))


    # Initialize pipeline on current rank
    if hasattr(config, 'denoising_step_list'):
        pipeline = CausalInferencePipeline(config, device=device)
    else:
        pipeline = CausalDiffusionInferencePipeline(config, device=device)

    if args.checkpoint_path:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        if args.use_ema and 'generator_ema' in state_dict:
            print("Loading EMA generator weights")
            pipeline.generator.load_state_dict(state_dict['generator_ema'])
        elif 'generator' in state_dict:
            print("Loading generator weights")
            pipeline.generator.load_state_dict(state_dict['generator'])
        elif 'state_dict' in state_dict:
            print("Loading weights from 'state_dict' key")
            pipeline.generator.load_state_dict(state_dict['state_dict'])
        elif 'model' in state_dict:
            print("Loading weights from 'model' key")
            pipeline.generator.load_state_dict(state_dict['model'])
        else:
            print("Loading checkpoint as raw state_dict")
            print(f"Checkpoint keys: {list(state_dict.keys())[:5]}...")
            pipeline.generator.load_state_dict(state_dict)

    pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    # Initialize scorer
    scorer = None
    use_gemini = getattr(args, 'use_gemini', False)

    # Initialize Gemini queue and consumer thread (only for rank 0)
    gemini_queue = None
    result_queue = None
    gemini_consumer = None

    if use_guidance:
        if use_gemini:
            # Gemini mode: Only rank 0 needs scorer, runs in separate CPU thread
            if local_rank == 0:
                print(f"[Rank 0] Initializing Gemini scorer (API-based, no GPU model)...")
                scorer = MultiObjectiveScorer(
                    device=device,
                    use_gemini=True,
                    use_qwen_vl=args.use_qwen_vl,
                    qwen_vl_model=args.qwen_vl_model,
                    qwen_vl_fps=args.qwen_vl_fps,
                    qwen_vl_max_pixels=args.qwen_vl_max_pixels,
                    use_stage1_model=args.use_stage1_filter,
                    stage1_model=args.stage1_model,
                    use_free_llm_for_checklist=args.use_free_llm_for_checklist,
                    free_llm_model=args.free_llm_model
                )

                # Create queues for producer-consumer pattern
                gemini_queue = Queue(maxsize=10)  # Buffer up to 10 batches
                result_queue = Queue()
                total_batches_queued = 0  # Track total batches for final collection
                gemini_start_time = time.time()  # Track total Gemini evaluation time

                # Start background consumer thread (CPU-only, rank 0)
                gemini_consumer = threading.Thread(
                    target=gemini_consumer_thread,
                    args=(gemini_queue, result_queue, scorer, args),
                    daemon=True
                )
                gemini_consumer.start()

                print(f"✅ Gemini mode enabled with background evaluation thread")
                print(f"   Stage 1 workers: {args.max_gemini_stage1_workers}")
                print(f"   Stage 2 workers: {args.max_gemini_stage2_workers}")
                print(f"   Stage 3 workers: {args.max_gemini_stage3_workers}")
        else:
            # VideoLLaMA3 mode: all ranks load scorer for parallel scoring
            if dist.is_initialized():
                print(f"[Rank {local_rank}] Loading VLM scorer on GPU {local_rank}...")
            scorer = MultiObjectiveScorer(
                device=device,
                use_gemini=False,
                use_qwen_vl=args.use_qwen_vl,
                qwen_vl_model=args.qwen_vl_model,
                qwen_vl_fps=args.qwen_vl_fps,
                qwen_vl_max_pixels=args.qwen_vl_max_pixels,
                use_stage1_model=args.use_stage1_filter,
                stage1_model=args.stage1_model,
                use_free_llm_for_checklist=args.use_free_llm_for_checklist,
                free_llm_model=args.free_llm_model
            )
            if dist.is_initialized():
                print(f"[Rank {local_rank}] ✓ VLM scorer loaded")


    # Create dataset
    if args.i2v:
        assert not dist.is_initialized(), "I2V does not support distributed inference yet"
        transform = transforms.Compose([
            transforms.Resize((480, 832)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = TextImagePairDataset(args.data_path, transform=transform)
    else:
        dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
    num_prompts = len(dataset)
    print(f"Number of prompts: {num_prompts}")

    # When using guidance with DDP, all ranks must process the same prompts
    # so they can collaborate on candidate generation (candidate-level parallelism).
    # DistributedSampler is only used when NOT using guidance (prompt-level parallelism).
    if dist.is_initialized() and not use_guidance:
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
        print(f"[Rank {local_rank}] Using DistributedSampler - each rank processes different prompts")
    else:
        sampler = SequentialSampler(dataset)
        if dist.is_initialized() and use_guidance:
            print(f"[Rank {local_rank}] Using SequentialSampler - all ranks process same prompts for candidate generation")
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

    # Create output directory (only on main process to avoid race conditions)
    if local_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)

    if dist.is_initialized():
        # Barrier uses current device automatically for NCCL backend
        dist.barrier()


    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        device, dtype = videos[0].device, videos[0].dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in videos
        ]

        output = torch.stack(output, dim=0)
        return output


    for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
        idx = batch_data['idx'].item()

        # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
        # Unpack the batch data for convenience
        if isinstance(batch_data, dict):
            batch = batch_data
        elif isinstance(batch_data, list):
            batch = batch_data[0]  # First (and only) item in the batch

        all_video = []
        num_generated_frames = 0  # Number of generated (latent) frames

        if args.i2v:
            # For image-to-video, batch contains image and caption
            prompt = batch['prompts'][0]  # Get caption from batch
            prompts = [prompt] * args.num_samples

            # Process the image
            image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
            )
        else:
            # For text-to-video, batch is just the text prompt
            prompt = batch['prompts'][0]
            extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
            if extended_prompt is not None:
                prompts = [extended_prompt] * args.num_samples  # Use extended prompt for generation
                original_prompts = [prompt] * args.num_samples  # Keep original prompt for scoring
            else:
                prompts = [prompt] * args.num_samples
                original_prompts = None  # Will default to prompts in pipeline
            initial_latent = None
            # Generate noise candidates for inference guidance
            if use_guidance:
                # Set seed based on prompt index for deterministic but varied noise per prompt
                torch.manual_seed(args.seed + idx)

                # Generate separate candidate sets for each batch item
                # Each batch item gets its own set of candidates
                noise_sampler = NoiseSampler(device=device)

                # Check if we should use fixed first frame strategy
                use_fixed_first_frame = hasattr(args, 'use_fixed_first_frame') and args.use_fixed_first_frame

                # ONLY RANK 0 generates noise candidates, then broadcasts to all other GPUs
                if local_rank == 0:
                    if use_fixed_first_frame:
                        print(f"  🎯 Using fixed first frame strategy...")

                        # Generate one random first frame for each batch item
                        batch_candidate_sets = []
                        for batch_idx in range(args.num_samples):
                            # Generate random first frame
                            first_frame_noise = torch.randn(
                                1, 1, 16, 60, 104, device=device, dtype=torch.bfloat16
                            )

                            # Generate num_candidates with this fixed first frame, varying only subsequent frames
                            candidates_for_batch = noise_sampler.generate_fixed_first_frame_candidates(
                                first_frame_noise=first_frame_noise,
                                batch_size=1,
                                num_frames=args.num_output_frames,
                                num_channels=16,
                                height=60,
                                width=104,
                                device=device,
                                dtype=torch.bfloat16,
                                num_candidates=num_candidates,
                                noise_generation_type=args.noise_generation_type
                            )
                            batch_candidate_sets.append(candidates_for_batch)
                    else:
                        # Original strategy: generate full noise from scratch
                        batch_candidate_sets = []
                        for batch_idx in range(args.num_samples):
                            # Generate num_candidates maximally separated noises for this batch item
                            candidates_for_batch = noise_sampler.generate_widespread_noise_candidates(
                                1, args.num_output_frames, 16, 60, 104, device, torch.bfloat16,
                                num_candidates, args.noise_generation_type
                            )
                            batch_candidate_sets.append(candidates_for_batch)

                    # Reorganize: from [batch][candidate][1,T,C,H,W] to [candidate][batch,T,C,H,W]
                    sampled_noise_candidates = []
                    for cand_idx in range(num_candidates):
                        # Stack all batch items for this candidate index
                        cand_noise = torch.cat([
                            batch_candidate_sets[batch_idx][cand_idx]
                            for batch_idx in range(args.num_samples)
                        ], dim=0)  # Concatenate along batch dimension
                        sampled_noise_candidates.append(cand_noise)
                else:
                    # Other ranks: create empty placeholder tensors that will be filled by broadcast
                    sampled_noise_candidates = []
                    for cand_idx in range(num_candidates):
                        cand_noise = torch.empty(
                            args.num_samples, args.num_output_frames, 16, 60, 104,
                            device=device, dtype=torch.bfloat16
                        )
                        sampled_noise_candidates.append(cand_noise)

                # Broadcast noise candidates from rank 0 to all other ranks
                if dist.is_initialized():
                    for cand_idx in range(num_candidates):
                        dist.broadcast(sampled_noise_candidates[cand_idx], src=0)
                    if local_rank == 0:
                        print(f"  📡 Broadcasted {num_candidates} noise candidates to all {world_size} GPUs")

                sampled_noise = sampled_noise_candidates[0]  # Use first candidate as default
            else:
                # Set seed based on prompt index for deterministic but varied noise per prompt
                torch.manual_seed(args.seed + idx)
                sampled_noise = torch.randn(
                    [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
                )
                sampled_noise_candidates = None

        # Generate frames with DDP-based candidate generation
        if use_guidance and dist.is_initialized() and sampled_noise_candidates is not None:
            # DDP-based multi-GPU candidate generation
            if local_rank == 0:
                print(f"\n📹 Generating {num_candidates} candidates across {world_size} GPUs for prompt {idx}...")

            # All ranks already generated identical noise candidates (same seed at line 214)
            # Broadcast from rank 0 to ensure bit-for-bit identical values across all ranks
            for cand_idx in range(num_candidates):
                dist.broadcast(sampled_noise_candidates[cand_idx], src=0)

            dist.barrier()  # Ensure all ranks have synchronized candidates

            # Determine which candidates this rank will process (round-robin)
            my_candidate_indices = list(range(local_rank, num_candidates, world_size))

            if local_rank == 0:
                print(f"  Rank assignments: {[f'Rank {r}: {list(range(r, num_candidates, world_size))}' for r in range(world_size)]}")

            # Generate videos for my assigned candidates
            my_video_results = []  # List of dict with candidate_idx, batch_idx, video_path, rank

            for cand_idx in my_candidate_indices:
                if local_rank == 0:
                    print(f"  [Rank {local_rank}] Processing candidate {cand_idx}/{num_candidates-1}...")

                # Set seed for this candidate
                # When using fixed first frame, use same seed for all candidates to ensure
                # identical first frame denoising across all GPUs
                if use_fixed_first_frame:
                    set_seed(args.seed + idx * 1000)
                else:
                    set_seed(args.seed + idx * 1000 + cand_idx)

                # Generate video for this candidate
                cand_noise = sampled_noise_candidates[cand_idx]

                video, latents = pipeline.inference(
                    noise=cand_noise,
                    text_prompts=prompts,
                    return_latents=True,
                    initial_latent=initial_latent,
                    low_memory=low_memory,
                    profile=args.profile,
                    save_intermediates=False,  # Don't save intermediates for candidates
                    output_folder=args.output_folder,
                    fix_first_frame_noise=use_fixed_first_frame,
                )

                # Decode and save videos for each batch item
                video_decoded = rearrange(video, 'b t c h w -> b t h w c').cpu()
                video_final = 255.0 * video_decoded

                batch_size = video_final.shape[0]
                # Use augmented/extended prompt for scoring (same as generation prompt)
                # This ensures checklist includes all entities mentioned in the detailed prompt
                scoring_prompt = prompts[0]  # Extended prompt if available, otherwise pure prompt
                clean_prompt = scoring_prompt[:50].replace('/', '_').replace('\\', '_').replace(':', '_')

                for batch_idx in range(batch_size):
                    video_path = os.path.join(
                        args.output_folder,
                        f"rank{local_rank}_candidate_{cand_idx}_batch_{batch_idx}_{clean_prompt}.mp4"
                    )
                    write_video(video_path, video_final[batch_idx], fps=16)
                    my_video_results.append({
                        'candidate_idx': cand_idx,
                        'batch_idx': batch_idx,
                        'video_path': video_path,
                        'rank': local_rank
                    })

                # Clear cache after each candidate
                pipeline.vae.model.clear_cache()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            dist.barrier()  # Wait for all ranks to finish generating

            # Gather all video results to rank 0
            all_video_results = [None] * world_size
            dist.all_gather_object(all_video_results, my_video_results)

            # Rank 0: Organize video paths
            if local_rank == 0:
                # Flatten the gathered results
                all_results_flat = [item for sublist in all_video_results for item in sublist]

                # Organize by [candidate][batch]
                candidate_video_paths = [[None] * args.num_samples for _ in range(num_candidates)]
                for result in all_results_flat:
                    candidate_video_paths[result['candidate_idx']][result['batch_idx']] = result['video_path']
            else:
                candidate_video_paths = None

            # Broadcast candidate_video_paths to all ranks for parallel scoring
            # Wrap in list for broadcast_object_list (expects a list, broadcasts each element)
            candidate_video_paths = [candidate_video_paths]
            dist.broadcast_object_list(candidate_video_paths, src=0)
            candidate_video_paths = candidate_video_paths[0]  # Unwrap the broadcasted object

            batch_size = args.num_samples

            # If save_all_candidates mode, skip selection and keep all candidates
            if args.save_all_candidates:
                if local_rank == 0:
                    print(f"\n✅ GENERATE ONLY MODE: All {num_candidates} candidates saved for each prompt")
                    print(f"   📁 Location: {args.output_folder}")
                    print(f"   💡 Use select_best.py or vbench_proper_evaluation.py --select_only to choose the best")
                # Skip to next prompt
                continue

            if use_gemini:
                # Gemini mode: Distributed tournament - all ranks process different batches in parallel
                if local_rank == 0:
                    print(f"  🎯 Scoring {batch_size} batch items using 3-Stage Gemini Evaluation (Stage 1: Visual Hallucination Filter → Stage 2: Forensic Observation → Stage 3: Tournament)...")
                    print(f"  📊 Processing {num_candidates} candidates per batch")

                scoring_start_time = time.time()
                all_scoring_results_list = [] if local_rank == 0 else None

                # Generate prompt alignment checklist once for this prompt (rank 0 only)
                if local_rank == 0:
                    print(f"\n  📋 Generating Prompt Alignment Checklist ONCE for all {batch_size} batches...")
                    prompt_alignment_checklist = scorer.generate_prompt_alignment_checklist(
                        text_prompt=scoring_prompt,
                        model_name="gemini-2.5-flash",
                        max_output_tokens=1500,
                        temperature=0.1
                    )

                    print(f"\n  🚀 Queueing {batch_size} batches for background Gemini evaluation...")

                    # Queue all batches for background processing (non-blocking)
                    model = "regular" if not args.use_ema else "ema"
                    filename_prompt = original_prompts[0] if original_prompts else prompt

                    for batch_idx in range(batch_size):
                        # Extract paths for this batch item across all candidates
                        batch_item_paths = [candidate_paths[batch_idx] for candidate_paths in candidate_video_paths]

                        # Calculate final output path for this batch
                        actual_idx = args.start_index + batch_idx
                        if args.save_with_index:
                            final_output_path = os.path.join(args.output_folder, f'{idx}-{actual_idx}_{model}.mp4')
                        else:
                            final_output_path = os.path.join(args.output_folder, f'{filename_prompt[:180]}-{actual_idx}.mp4')

                        # Push to Gemini consumer thread (non-blocking) - include final output path
                        gemini_queue.put((idx, batch_idx, batch_item_paths, scoring_prompt, prompt_alignment_checklist, final_output_path))
                        print(f"    ✓ Batch {batch_idx + 1}/{batch_size} queued ({len(batch_item_paths)} candidates)")

                    total_batches_queued += batch_size
                    print(f"\n  ⏭️  GPUs moving to next prompt immediately (Gemini evaluates in background)...")

                # No barriers needed! GPUs continue immediately to next prompt
                # The Gemini consumer thread processes batches asynchronously on CPU
                # Results will be collected at the very end (after all prompts processed)
            else:
                # VideoLLaMA3 mode: parallel scoring across ranks
                # Each rank scores its assigned batch items (round-robin)
                my_batch_indices = list(range(local_rank, batch_size, world_size))

                if local_rank == 0:
                    print(f"  🎯 Scoring {batch_size} batch items across {world_size} GPUs...")
                    print(f"  Scoring assignments: {[f'Rank {r}: batches {list(range(r, batch_size, world_size))}' for r in range(world_size)]}")

                scoring_start_time = time.time()

                my_scoring_results = []
                for batch_idx in my_batch_indices:
                    # Extract paths for this batch item across all candidates
                    batch_item_paths = [candidate_paths[batch_idx] for candidate_paths in candidate_video_paths]

                    if local_rank == 0:
                        print(f"    [Rank {local_rank}] Scoring batch {batch_idx + 1}/{batch_size}...")

                    # Score this batch item's candidates on this GPU
                    scores = scorer.score_complete_videos(batch_item_paths, scoring_prompt)

                    # Find best candidate for this batch item
                    best_candidate_idx = torch.argmax(scores).item()
                    best_score = scores[best_candidate_idx].item()

                    my_scoring_results.append({
                        'batch_idx': batch_idx,
                        'best_candidate_idx': best_candidate_idx,
                        'best_score': best_score
                    })

                    if local_rank == 0:
                        print(f"      ✓ Best: Candidate {best_candidate_idx} (score: {best_score:.4f})")

                scoring_end_time = time.time()
                scoring_duration = scoring_end_time - scoring_start_time

                if local_rank == 0:
                    print(f"  ⏱️  Scoring took {scoring_duration:.2f}s across {world_size} GPUs")

                # Gather scoring results to rank 0
                all_scoring_results = [None] * world_size
                dist.all_gather_object(all_scoring_results, my_scoring_results)

                # Flatten and sort by batch_idx
                if local_rank == 0:
                    all_scoring_results_list = [item for sublist in all_scoring_results for item in sublist]
                    all_scoring_results_list.sort(key=lambda x: x['batch_idx'])
                else:
                    all_scoring_results_list = None

            # Only rank 0 renames final winners (losers already deleted during tournament)
            if local_rank == 0:
                best_candidate_indices = [r['best_candidate_idx'] for r in all_scoring_results_list]
                best_video_paths = [
                    candidate_video_paths[r['best_candidate_idx']][r['batch_idx']]
                    for r in all_scoring_results_list
                ]

                print(f"  Best candidates per batch: {best_candidate_indices}")
                print(f"  (Losers already deleted during tournament)")

                # Rename best videos to final output paths
                model = "regular" if not args.use_ema else "ema"
                filename_prompt = original_prompts[0] if original_prompts else prompt

                for seed_idx, best_path in enumerate(best_video_paths):
                    actual_idx = args.start_index + seed_idx
                    if args.save_with_index:
                        final_output_path = os.path.join(args.output_folder, f'{idx}-{actual_idx}_{model}.mp4')
                    else:
                        # Use prompt directly as filename with index (VBench 2.0 format)
                        final_output_path = os.path.join(args.output_folder, f'{filename_prompt[:180]}-{actual_idx}.mp4')

                    # Rename temp file to final name
                    if os.path.exists(best_path):
                        os.rename(best_path, final_output_path)
                        print(f"  ✓ Saved: {final_output_path}")

            # All ranks wait for rank 0 to finish cleanup before continuing to next prompt
            dist.barrier()

            # Skip the rest of the loop - videos already saved
            continue

        # Single-GPU guidance mode
        elif use_guidance and sampled_noise_candidates is not None:
            print(f"\n📹 Generating {num_candidates} candidates on single GPU for prompt {idx}...")

            # Generate videos for all candidates
            candidate_video_paths = []

            for cand_idx in range(num_candidates):
                print(f"  Generating candidate {cand_idx + 1}/{num_candidates}...")

                # Set seed for this candidate
                set_seed(args.seed + idx * 1000 + cand_idx)

                # Generate video for this candidate
                cand_noise = sampled_noise_candidates[cand_idx]
                video, latents = pipeline.inference(
                    noise=cand_noise,
                    text_prompts=prompts,
                    return_latents=True,
                    initial_latent=initial_latent,
                    low_memory=low_memory,
                    profile=args.profile,
                    save_intermediates=False,  # Don't save intermediates for candidates
                    output_folder=args.output_folder,
                    fix_first_frame_noise=use_fixed_first_frame,
                )

                # Decode and save videos for each batch item
                video_decoded = rearrange(video, 'b t c h w -> b t h w c').cpu()
                video_final = 255.0 * video_decoded

                batch_size = video_final.shape[0]
                # Use augmented/extended prompt for scoring (same as generation prompt)
                # This ensures checklist includes all entities mentioned in the detailed prompt
                scoring_prompt = prompts[0]  # Extended prompt if available, otherwise pure prompt
                clean_prompt = scoring_prompt[:50].replace('/', '_').replace('\\', '_').replace(':', '_')

                batch_video_paths = []
                for batch_idx in range(batch_size):
                    video_path = os.path.join(
                        args.output_folder,
                        f"candidate_{cand_idx}_batch_{batch_idx}_{clean_prompt}.mp4"
                    )
                    write_video(video_path, video_final[batch_idx], fps=16)
                    batch_video_paths.append(video_path)

                candidate_video_paths.append(batch_video_paths)

                # Clear cache after each candidate
                pipeline.vae.model.clear_cache()
                torch.cuda.empty_cache()

            print(f"  ✓ All {num_candidates} candidates generated")

            # Scorer already initialized at startup
            # Score per batch item
            batch_size = args.num_samples
            best_candidate_indices = []
            best_video_paths = []

            scoring_start_time = time.time()

            if use_gemini:
                # Gemini mode: use tournament-style comparison
                print(f"  🎯 Scoring candidates using Gemini API...")
                for batch_idx in range(batch_size):
                    # Extract paths for this batch item across all candidates
                    batch_item_paths = [candidate_paths[batch_idx] for candidate_paths in candidate_video_paths]

                    print(f"    Batch item {batch_idx + 1}/{batch_size}: Running Gemini tournament...")

                    # Use Gemini tournament-style comparison
                    best_path = scorer.compare_videos_gemini_tournament(
                        batch_item_paths, scoring_prompt
                    )
                    # Find index of winning video
                    best_candidate_idx = batch_item_paths.index(best_path)
                    best_candidate_indices.append(best_candidate_idx)
                    best_video_paths.append(best_path)

                    print(f"      ✓ Best: Candidate {best_candidate_idx} (Gemini winner)")

                scoring_end_time = time.time()
                scoring_duration = scoring_end_time - scoring_start_time
                print(f"  ⏱️  Gemini scoring took {scoring_duration:.2f}s")
            else:
                # VideoLLaMA3 mode: score with local model
                print(f"  🎯 Scoring candidates per batch item...")
                for batch_idx in range(batch_size):
                    # Extract paths for this batch item across all candidates
                    batch_item_paths = [candidate_paths[batch_idx] for candidate_paths in candidate_video_paths]

                    print(f"    Batch item {batch_idx + 1}/{batch_size}:")
                    # Score this batch item's candidates
                    scores = scorer.score_complete_videos(batch_item_paths, scoring_prompt)

                    # Select best candidate for this batch item
                    best_candidate_idx = torch.argmax(scores).item()
                    best_score = scores[best_candidate_idx].item()
                    best_candidate_indices.append(best_candidate_idx)

                    # Get path of best video for this batch item
                    best_path = batch_item_paths[best_candidate_idx]
                    best_video_paths.append(best_path)

                    print(f"      ✓ Best: Candidate {best_candidate_idx} (score: {best_score:.4f})")

                scoring_end_time = time.time()
                scoring_duration = scoring_end_time - scoring_start_time
                print(f"  ⏱️  Scoring took {scoring_duration:.2f}s on single GPU")

            print(f"  Best candidates per batch: {best_candidate_indices}")

            # Delete all candidate videos except the best ones
            deleted_count = 0
            for candidate_paths in candidate_video_paths:
                for vid_path in candidate_paths:
                    if vid_path not in best_video_paths:
                        if os.path.exists(vid_path):
                            os.remove(vid_path)
                            deleted_count += 1

            print(f"  ✓ Kept {len(best_video_paths)} best videos, deleted {deleted_count} candidates\n")

            # Rename best videos to final output paths
            model = "regular" if not args.use_ema else "ema"
            filename_prompt = original_prompts[0] if original_prompts else prompt

            for seed_idx, best_path in enumerate(best_video_paths):
                actual_idx = args.start_index + seed_idx
                if args.save_with_index:
                    final_output_path = os.path.join(args.output_folder, f'{idx}-{actual_idx}_{model}.mp4')
                else:
                    # Use prompt directly as filename with index (VBench 2.0 format)
                    final_output_path = os.path.join(args.output_folder, f'{filename_prompt[:180]}-{actual_idx}.mp4')

                # Rename temp file to final name
                if os.path.exists(best_path):
                    os.rename(best_path, final_output_path)
                    print(f"  ✓ Saved: {final_output_path}")

            # Skip the rest of the loop - videos already saved
            continue

        # Non-guidance mode (baseline - single video per prompt, no comparison)
        # Fixed first frame has no effect in baseline mode (no candidates to compare)
        if hasattr(args, 'use_fixed_first_frame') and args.use_fixed_first_frame:
            if local_rank == 0:
                print("⚠️  Warning: --use_fixed_first_frame has no effect in baseline mode (only useful with --use_inference_guidance)")

        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent,
            low_memory=low_memory,
            profile=args.profile,
            save_intermediates=args.save_intermediates,
            output_folder=args.output_folder,
            fix_first_frame_noise=False,  # Always False in baseline mode
        )

        # DEBUG: Print actual generated video properties
        print(f"🔍 DEBUG: Expected frames from config: {args.num_output_frames}")
        print(f"🔍 DEBUG: Generated video shape: {video.shape}")
        print(f"🔍 DEBUG: Generated latents shape: {latents.shape}")
        print(f"🔍 DEBUG: Sampled noise shape: {sampled_noise.shape}")
        if initial_latent is not None:
            print(f"🔍 DEBUG: Initial latent shape: {initial_latent.shape}")
        actual_frames = video.shape[1] if len(video.shape) > 1 else video.shape[0]
        calculated_duration = actual_frames / 16.0  # 16 FPS
        print(f"🔍 DEBUG: Actual frames generated: {actual_frames}")
        print(f"🔍 DEBUG: Calculated duration at 16 FPS: {calculated_duration:.2f} seconds")

        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        all_video.append(current_video)
        num_generated_frames += latents.shape[1]

        # Final output video
        video = 255.0 * torch.cat(all_video, dim=1)

        # Clear VAE cache
        pipeline.vae.model.clear_cache()

        # Save the video if the current prompt is not a dummy prompt
        if idx < num_prompts:
            model = "regular" if not args.use_ema else "ema"
            for seed_idx in range(args.num_samples):
                # All processes save their videos
                # Use start_index offset for batch processing (e.g., batch 2 starts at index 5)
                actual_idx = args.start_index + seed_idx
                if args.save_with_index:
                    output_path = os.path.join(args.output_folder, f'{idx}-{actual_idx}_{model}.mp4')
                else:
                    # Use prompt directly as filename with index (VBench 2.0 format)
                    # Note: prompt should be the short version passed for naming
                    output_path = os.path.join(args.output_folder, f'{prompt[:180]}-{actual_idx}.mp4')
                write_video(output_path, video[seed_idx], fps=16)

    # After all prompts processed, collect Gemini evaluation results
    if use_gemini and local_rank == 0 and 'total_batches_queued' in locals() and total_batches_queued > 0:
        print(f"\n{'='*80}")
        print(f"🏁 All {total_batches_queued} batches queued. Waiting for Gemini evaluations to complete...")
        print(f"{'='*80}\n")

        # Collect all results from the queue (blocking until all complete)
        all_gemini_results = []
        for i in range(total_batches_queued):
            batch_idx, final_path = result_queue.get()
            all_gemini_results.append((batch_idx, final_path))
            print(f"  ✅ [{i+1}/{total_batches_queued}] Batch {batch_idx} complete: {os.path.basename(final_path) if final_path else 'ERROR'}")

        # Calculate total Gemini evaluation time
        gemini_end_time = time.time()
        total_gemini_time = gemini_end_time - gemini_start_time

        print(f"\n{'='*80}")
        print(f"✅ All Gemini evaluations complete!")
        print(f"   Total time: {total_gemini_time:.2f}s ({total_gemini_time/60:.1f} minutes)")
        print(f"   Average per batch: {total_gemini_time/total_batches_queued:.2f}s")
        print(f"{'='*80}\n")

    # After all prompts processed, cleanup Gemini consumer thread
    if use_gemini and local_rank == 0 and 'gemini_consumer' in locals() and gemini_consumer is not None:
        print(f"\n[Rank 0] Stopping Gemini consumer thread...")
        gemini_queue.put(None)  # Poison pill to signal thread to stop
        gemini_consumer.join(timeout=300)  # Wait up to 5 minutes for thread to finish
        if gemini_consumer.is_alive():
            print(f"[Rank 0] ⚠️  Warning: Gemini consumer thread did not stop within timeout")
        else:
            print(f"[Rank 0] ✓ Gemini consumer thread stopped successfully")
