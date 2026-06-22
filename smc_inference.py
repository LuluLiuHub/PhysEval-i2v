"""
SMC (Sequential Monte Carlo) + Gemini Flash Guided Video Generation

Architecture (default config: num_frame_per_block=3, block_sizes=[6,6,6]):
  Fixed initial: Frames 1-3   (3 frames, num_frame_per_block, deterministic)
  Block 1:       Frames 4-9   (6 frames) - 50 particles → top 10 via Gemini
  Block 2:       Frames 10-15 (6 frames) - 10×5=50 branches → top 10 via Gemini
  Block 3:       Frames 16-21 (6 frames) - 10×5=50 branches → best 1 via Gemini

  Total: 3 + 6 + 6 + 6 = 21 frames

Gemini evaluation strategy:
  - Intermediate stages (Block 1, 2): Stage 1 visual filter + direct video comparison
  - Final stage (Block 3): Full 3-stage pipeline (filter → SCRIBE → tournament)
"""

import os
import argparse
import time
import json
import shutil
import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from omegaconf import OmegaConf
from einops import rearrange
from torchvision.io import write_video

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from pipeline.vlm_scorer import MultiObjectiveScorer
from utils.misc import set_seed
from demo_utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller


# =========================================================================
# CLI Arguments
# =========================================================================

parser = argparse.ArgumentParser(description="SMC + Gemini Guided Video Generation")

# ── Shared args (same names as inference.py) ──────────────────────────────────
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the prompts file")
parser.add_argument("--extended_prompt_path", type=str,
                    help="Path to extended prompts for generation (raw prompts used for Gemini scoring)")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Total frames: 1 (fixed) + sum(smc_block_sizes)")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--start_index", type=int, default=0,
                    help="Starting index for saved video filenames (for batch processing)")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--save_intermediates", action="store_true",
                    help="Save intermediate denoising predictions for analysis")
parser.add_argument("--noise_generation_type", type=str, default="mixed",
                    choices=["mixed", "scale_variations", "pure_random", "edge_cases",
                             "frequency_variations", "spatial_correlations", "temporal_correlations",
                             "low_discrepancy"],
                    help="Type of noise generation pattern to use for particles")
parser.add_argument("--profile", action="store_true",
                    help="Enable profiling to measure VAE decoding and denoising time")
parser.add_argument("--low_memory", action="store_true",
                    help="Enable low memory mode")
parser.add_argument("--use_gemini", action="store_true",
                    help="Use Gemini API for physics-aware video evaluation (required for SMC)")
parser.add_argument("--max_gemini_stage1_workers", type=int, default=5,
                    help="Max concurrent Gemini API calls for Stage 1 (default: 5)")
parser.add_argument("--max_gemini_stage2_workers", type=int, default=5,
                    help="Max concurrent Gemini API calls for Stage 2 (default: 5)")

# ── SMC-specific args ─────────────────────────────────────────────────────────
# height/width needed here because SMC creates noise tensors directly;
# inference.py derives resolution from dataset images instead.
parser.add_argument("--height", type=int, default=480, help="Video height in pixels")
parser.add_argument("--width", type=int, default=832, help="Video width in pixels")
parser.add_argument("--smc_block_sizes", type=int, nargs="+", default=[6, 6, 6],
                    help="Frames per block after fixed initial block (default: 6 6 6 for num_frame_per_block=3 models)")
parser.add_argument("--smc_initial_particles", type=int, default=50,
                    help="Number of particles to generate in Block 1 (default: 50)")
parser.add_argument("--smc_top_k", type=int, default=10,
                    help="Number of top particles to keep after each intermediate stage (default: 10)")
parser.add_argument("--smc_branch_factor", type=int, default=5,
                    help="Number of branches per winner for Block 2+ (default: 5)")
parser.add_argument("--gemini_api_key", type=str, default=None,
                    help="Gemini API key (or set GEMINI_API_KEY env var)")


# =========================================================================
# Data Structures
# =========================================================================

@dataclass
class SMCParticle:
    """Represents one particle's state at any point in the SMC pipeline."""
    particle_id: int
    # Accumulated latents from all blocks so far: shape (1, frames_so_far, C, H, W)
    # Used for conditioning next block generation (KV cache)
    latents: Optional[torch.Tensor] = None
    # Path to the decoded video on disk (for Gemini scoring)
    # Video is written to disk immediately during generation to save memory
    video_path: Optional[str] = None
    # Gemini evaluation score (higher = better)
    score: float = 0.0
    # Which winner this was branched from (for Stage 2+)
    parent_id: Optional[int] = None


@dataclass
class SMCConfig:
    """Configuration for SMC guided generation."""
    # Frame structure
    block_sizes: List[int] = field(default_factory=lambda: [7, 7, 6])
    # Particle counts
    initial_particles: int = 50    # Stage 1: 50 particles
    top_k: int = 10               # Keep top 10 after each stage
    branch_factor: int = 5         # Branch 5 from each winner → 50 in stages 2+
    # Gemini workers (matches inference.py split)
    max_gemini_stage1_workers: int = 5   # Intermediate stage: Stage 1 filter + direct comparison
    max_gemini_stage2_workers: int = 5   # Final stage: full 3-stage pipeline


# =========================================================================
# SMC Guided Generation
# =========================================================================

class SMCGuidedGeneration:
    """
    Sequential Monte Carlo guided video generation using Gemini Flash.

    Each block is generated for N particles in parallel across GPUs.
    After each block, Gemini evaluates partial videos and selects the top K.
    The top K are then branched into K × branch_factor new particles for the next block.
    """

    def __init__(self, pipeline, vae, scorer, config: SMCConfig,
                 device: torch.device, local_rank: int, world_size: int,
                 output_folder: str, num_frame_channels: int = 16):
        self.pipeline = pipeline
        self.vae = vae
        self.scorer = scorer
        self.config = config
        self.device = device
        self.local_rank = local_rank
        self.world_size = world_size
        self.output_folder = output_folder
        self.num_frame_channels = num_frame_channels

        # Create temp folder for SMC intermediate videos (all ranks need to write)
        self.smc_tmp_dir = os.path.join(output_folder, "_smc_tmp")
        os.makedirs(self.smc_tmp_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def run(self, prompt: str, fixed_first_frame_latent: torch.Tensor,
            noise_template: torch.Tensor, text_prompts: List[str], prompt_idx: int,
            prompt_alignment_checklist: str) -> Optional[str]:
        """
        Run full SMC guided generation for a single prompt.

        Args:
            prompt:                     Short prompt string (for filenames/logging)
            fixed_first_frame_latent:   Latent for the fixed first frames, shape (1, num_fixed_frames, C, H, W)
            noise_template:             Full noise tensor shape (1, total_frames, C, H, W)
                                        Used as a template for generating per-block noise
            text_prompts:               Extended prompts list for the pipeline
            prompt_idx:                 Index of this prompt (for unique file naming)
            prompt_alignment_checklist: Pre-generated Gemini checklist for this prompt

        Returns:
            Path to the best particle's video file, or None on failure
        """
        cfg = self.config
        total_blocks = len(cfg.block_sizes)

        if self.local_rank == 0:
            print(f"\n{'='*80}")
            print(f"SMC Guided Generation: '{prompt[:80]}'")
            print(f"  Blocks: {cfg.block_sizes} frames = {sum(cfg.block_sizes)} generated + 1 fixed")
            print(f"  Stage 1: {cfg.initial_particles} particles → top {cfg.top_k}")
            print(f"  Stage 2+: {cfg.top_k} × {cfg.branch_factor} = {cfg.top_k * cfg.branch_factor} branches → top {cfg.top_k}")
            print(f"{'='*80}\n")

        # ── Stage 1: Generate block 1 for initial_particles ──────────────────
        stage1_start = time.time()
        if self.local_rank == 0:
            print(f"[SMC] Stage 1: Generating {cfg.initial_particles} particles for block 1 ({cfg.block_sizes[0]} frames)...")

        particles = self._generate_block_for_particles(
            particle_ids=list(range(cfg.initial_particles)),
            parent_latents_map={},        # No parent latents in stage 1
            block_idx=0,
            noise_template=noise_template,
            fixed_first_frame_latent=fixed_first_frame_latent,
            text_prompts=text_prompts,
            prompt_idx=prompt_idx,
            stage_idx=1,
        )

        # Videos already written to disk during generation
        # Gemini evaluation: Stage 1 filter + direct comparison
        is_final = False
        top_k_particles = self._evaluate_intermediate(
            particles, prompt, prompt_alignment_checklist, stage_idx=1, is_final=is_final
        )

        if self.local_rank == 0:
            print(f"[SMC] Stage 1 complete in {time.time()-stage1_start:.1f}s | "
                  f"{len(top_k_particles)}/{len(particles)} particles retained")

        # ── Stages 2 to N-1: Branch and evaluate intermediate blocks ─────────
        current_winners = top_k_particles
        for block_idx in range(1, total_blocks):
            is_final = (block_idx == total_blocks - 1)
            stage_idx = block_idx + 1
            stage_start = time.time()
            n_branches = cfg.top_k * cfg.branch_factor

            if self.local_rank == 0:
                print(f"\n[SMC] Stage {stage_idx}: Branching {len(current_winners)} winners × "
                      f"{cfg.branch_factor} = {n_branches} particles "
                      f"for block {block_idx+1} ({cfg.block_sizes[block_idx]} frames)...")

            # Build parent maps: particle_id → accumulated latents
            parent_latents_map = {p.particle_id: p.latents for p in current_winners}

            # Assign new IDs for branches: winner_i spawns branch_factor children
            branch_ids = []
            branch_parents = []
            for winner in current_winners:
                for b in range(cfg.branch_factor):
                    branch_id = winner.particle_id * cfg.branch_factor + b
                    branch_ids.append(branch_id)
                    branch_parents.append(winner.particle_id)

            # Generate this block for all branches
            particles = self._generate_block_for_particles(
                particle_ids=branch_ids,
                parent_latents_map=parent_latents_map,
                block_idx=block_idx,
                noise_template=noise_template,
                fixed_first_frame_latent=fixed_first_frame_latent,
                text_prompts=text_prompts,
                prompt_idx=prompt_idx,
                stage_idx=stage_idx,
                parent_ids=branch_parents,
            )

            # Videos already written to disk during generation

            if is_final:
                # Final stage: full 3-stage Gemini evaluation
                top_k_particles = self._evaluate_final(particles, prompt, prompt_alignment_checklist, prompt_idx)
            else:
                # Intermediate: Stage 1 filter + direct comparison
                top_k_particles = self._evaluate_intermediate(
                    particles, prompt, prompt_alignment_checklist, stage_idx=stage_idx, is_final=False
                )

            if self.local_rank == 0:
                print(f"[SMC] Stage {stage_idx} complete in {time.time()-stage_start:.1f}s | "
                      f"{len(top_k_particles)}/{len(particles)} particles retained")

            current_winners = top_k_particles

        # ── Final winner ──────────────────────────────────────────────────────
        if not current_winners:
            if self.local_rank == 0:
                print(f"[SMC] ❌ No winners found, returning None")
            return None

        best = current_winners[0]
        if self.local_rank == 0:
            print(f"\n[SMC] ✅ Final winner: particle_{best.particle_id} (score={best.score:.3f})")
            print(f"[SMC] Video: {best.video_path}")

        # Return video path (video already on disk)
        return best.video_path

    # -------------------------------------------------------------------------
    # Block generation (distributed across GPUs)
    # -------------------------------------------------------------------------

    def _generate_block_for_particles(
        self,
        particle_ids: List[int],
        parent_latents_map: dict,        # particle_id → accumulated latents
        block_idx: int,
        noise_template: torch.Tensor,
        fixed_first_frame_latent: torch.Tensor,
        text_prompts: List[str],
        prompt_idx: int,
        stage_idx: int,
        parent_ids: Optional[List[int]] = None,
    ) -> List[SMCParticle]:
        """
        Distribute particles across GPU ranks and generate one block per particle.
        Each rank generates its assigned particles sequentially (batch_size=1 each).

        pipeline.inference() decodes all accumulated latents together (initial_latent + new block),
        ensuring temporal consistency. Returns particles with both accumulated latents (for next
        block's KV cache) and decoded video frames (for Gemini scoring).
        """
        block_size = self.config.block_sizes[block_idx]
        _, _, C, H, W = noise_template.shape

        # Assign particles to ranks (round-robin)
        my_particle_ids = [pid for i, pid in enumerate(particle_ids)
                           if i % self.world_size == self.local_rank]
        my_parent_ids = [parent_ids[i] for i in range(len(particle_ids))
                         if i % self.world_size == self.local_rank] if parent_ids else [None] * len(my_particle_ids)

        if self.local_rank == 0:
            print(f"  [Block {block_idx+1}] Distributing {len(particle_ids)} particles across {self.world_size} GPUs...")
            print(f"  [Rank {self.local_rank}] Generating {len(my_particle_ids)} particles...")

        my_particles = []
        for i, (pid, ppid) in enumerate(zip(my_particle_ids, my_parent_ids)):
            # Build fresh block noise for this particle on CPU (deterministic across ranks),
            # then move to device. CUDA randn diverges per-device even with the same seed.
            torch.manual_seed(pid * 10000 + block_idx * 1000 + prompt_idx)
            block_noise = torch.randn(1, block_size, C, H, W,
                                      dtype=noise_template.dtype).to(self.device)

            # Determine initial_latent
            # Stage 1: only fixed first frame
            # Stage 2+: fixed first frame + all previously generated blocks
            if ppid is None or ppid not in parent_latents_map:
                # Stage 1: initial_latent = fixed first frame only
                initial_latent = fixed_first_frame_latent.to(self.device)
            else:
                # Stage 2+: initial_latent = accumulated latents from parent
                initial_latent = parent_latents_map[ppid].to(self.device)

            # Run inference for this block
            # Pipeline decodes all accumulated latents together (initial_latent + new block)
            # Returns video already decoded from all accumulated latents
            with torch.no_grad():
                video, latents = self.pipeline.inference(
                    noise=block_noise,
                    text_prompts=text_prompts,
                    initial_latent=initial_latent,
                    return_latents=True,
                    fix_first_frame_noise=False,
                )

            # Accumulate latents for next block's KV cache conditioning
            accumulated_latents = torch.cat([initial_latent.cpu(), latents.cpu()], dim=1)

            # Convert video to (T, H, W, C) float32 [0, 255] format and write to disk immediately
            # Pipeline returns: (1, T, 3, H, W) already normalized to [0, 1]
            # Just rearrange and scale to [0, 255] (same as inference.py)
            video_frames = rearrange(video[0], 't c h w -> t h w c').cpu() * 255.0

            # Write to disk immediately to save memory
            video_path = os.path.join(
                self.smc_tmp_dir,
                f"prompt{prompt_idx:04d}_stage{stage_idx}_particle{pid:04d}.mp4"
            )
            write_video(video_path, video_frames, fps=16)

            particle = SMCParticle(
                particle_id=pid,
                latents=accumulated_latents,
                video_path=video_path,  # Store only path, not video data
                parent_id=ppid,
            )
            my_particles.append(particle)

            # Clear VAE cache after each particle (same as inference.py after each candidate)
            self.vae.model.clear_cache()
            torch.cuda.empty_cache()
            if self.world_size > 1:
                torch.cuda.synchronize()

            print(f"  [Rank {self.local_rank}] Particle {i+1}/{len(my_particle_ids)} (id={pid}) done")

        # Gather all particles on rank 0
        all_particles_per_rank = [None] * self.world_size
        dist.all_gather_object(all_particles_per_rank, my_particles)

        # Flatten into ordered list
        all_particles = []
        # Rebuild in original particle_ids order
        gathered = {p.particle_id: p for rank_particles in all_particles_per_rank
                    for p in rank_particles}
        for pid in particle_ids:
            if pid in gathered:
                all_particles.append(gathered[pid])

        return all_particles

    # -------------------------------------------------------------------------
    # Gemini evaluation: intermediate (Stage 1 filter + direct comparison)
    # -------------------------------------------------------------------------

    def _evaluate_intermediate(
        self,
        particles: List[SMCParticle],
        prompt: str,
        prompt_alignment_checklist: str,
        stage_idx: int,
        is_final: bool,
    ) -> List[SMCParticle]:
        """
        Evaluate partial videos using:
          1. Stage 1 visual filter (remove obvious glitches)
          2. Direct comparison among survivors to rank by quality

        Returns top_k particles sorted by score (best first).
        Only runs on rank 0; other ranks receive the result via broadcast.
        """
        top_k = self.config.top_k
        top_k_ids = []

        if self.local_rank == 0:
            video_paths = [p.video_path for p in particles if p.video_path]

            print(f"\n  [Gemini] Stage {stage_idx} evaluation: {len(video_paths)} partial videos...")
            eval_start = time.time()

            # Step 1: Stage 1 visual filter (parallel)
            # Use pre-generated checklist instead of generating it again
            retained, rejected = self.scorer.run_parallel_stage1(
                video_paths=video_paths,
                text_prompt=prompt,
                prompt_alignment_checklist=prompt_alignment_checklist,
                max_workers=self.config.max_gemini_stage1_workers,
            )

            print(f"  [Gemini] Stage 1 filter: {len(retained)}/{len(video_paths)} passed, "
                  f"{len(rejected)} rejected")

            # If all rejected, fall back to all videos for comparison
            candidates = retained if retained else [{'path': p} for p in video_paths]
            candidate_paths = [c['path'] for c in candidates]

            # Step 2: Direct video comparison to rank survivors
            # run_direct_video_comparison handles the len <= top_k case internally
            ranked_paths = self.scorer.run_direct_video_comparison(
                video_paths=candidate_paths,
                text_prompt=prompt,
                top_k=top_k,
                max_workers=self.config.max_gemini_stage1_workers,
                comparison_batch_size=5,
            )

            # Map winner paths back to particle IDs
            ranked_path_set = set(ranked_paths)
            path_to_particle = {p.video_path: p for p in particles}
            top_k_ids = [
                path_to_particle[path].particle_id
                for path in ranked_paths
                if path in path_to_particle
            ]

            print(f"  [Gemini] Evaluation done in {time.time()-eval_start:.1f}s | "
                  f"Winners: {top_k_ids}")

            # Cleanup: delete rejected video files
            for p in particles:
                if p.video_path and p.video_path not in ranked_path_set:
                    if os.path.exists(p.video_path):
                        os.remove(p.video_path)

        # Broadcast winner IDs to all ranks
        top_k_ids_container = [top_k_ids]
        dist.broadcast_object_list(top_k_ids_container, src=0)
        top_k_ids = top_k_ids_container[0]

        return [p for p in particles if p.particle_id in top_k_ids]

    # -------------------------------------------------------------------------
    # Gemini evaluation: final (full 3-stage pipeline)
    # -------------------------------------------------------------------------

    def _evaluate_final(self, particles: List[SMCParticle], prompt: str,
                       prompt_alignment_checklist: str, prompt_idx: int) -> List[SMCParticle]:
        """
        Final stage: run the full 3-stage Gemini pipeline (filter → SCRIBE → tournament).
        Returns a single-element list with the best particle.
        """
        best_id = None

        if self.local_rank == 0:
            video_paths = [p.video_path for p in particles if p.video_path]

            print(f"\n  [Gemini] Final evaluation: full 3-stage pipeline on {len(video_paths)} videos...")
            eval_start = time.time()

            result = self.scorer.run_3stage_pipeline_parallel(
                video_paths=video_paths,
                text_prompt=prompt,
                prompt_alignment_checklist=prompt_alignment_checklist,
                max_stage1_workers=self.config.max_gemini_stage1_workers,
                max_stage2_workers=self.config.max_gemini_stage2_workers,
                return_detailed=True,
            )

            winner_path = result.get('winner_path')
            print(f"  [Gemini] Final evaluation done in {time.time()-eval_start:.1f}s")
            print(f"  [Gemini] Stage 1: {result.get('stage1_time', 0):.1f}s | "
                  f"Stage 2: {result.get('stage2_time', 0):.1f}s | "
                  f"Stage 3: {result.get('stage3_time', 0):.1f}s")

            # === GENERATE STATISTICS REPORT ===
            stats_output_dir = os.path.join(self.smc_tmp_dir, "pipeline_stats")
            self.scorer.generate_pipeline_statistics_report(
                detailed_results=result,
                total_videos=len(video_paths),
                output_dir=stats_output_dir,
                prompt_idx=prompt_idx
            )

            # Find the winning particle
            path_to_particle = {p.video_path: p for p in particles}
            if winner_path and winner_path in path_to_particle:
                best_id = path_to_particle[winner_path].particle_id
            elif particles:
                best_id = particles[0].particle_id

            # Cleanup losers
            for p in particles:
                if p.video_path and p.video_path != winner_path and os.path.exists(p.video_path):
                    os.remove(p.video_path)

        # Broadcast winner ID
        best_id_container = [best_id]
        dist.broadcast_object_list(best_id_container, src=0)
        best_id = best_id_container[0]

        return [p for p in particles if p.particle_id == best_id]



# =========================================================================
# Main entrypoint
# =========================================================================

def main():
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # ── Initialize distributed (exact pattern from inference.py) ───────────────
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        # For SMC with Gemini, rank 0 may take longer during evaluation
        # Increase NCCL timeout to prevent other ranks from timing out
        import datetime
        timeout_duration = datetime.timedelta(seconds=3600)  # 60 minutes for SMC

        # Initialize process group with explicit device_id to avoid GPU mapping issues
        dist.init_process_group(backend='nccl', device_id=device, timeout=timeout_duration)
        world_size = dist.get_world_size()
        set_seed(args.seed + local_rank)
    else:
        device = torch.device("cuda")
        local_rank = 0
        world_size = 1
        set_seed(args.seed)

    print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
    low_memory = get_cuda_free_memory_gb(device) < 40

    torch.set_grad_enabled(False)

    # ── Load config (exact pattern from inference.py) ──────────────────────────
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    # ── Initialize pipeline (exact pattern from inference.py) ──────────────────
    if hasattr(config, 'denoising_step_list'):
        pipeline = CausalInferencePipeline(config, device=device)
    else:
        pipeline = CausalDiffusionInferencePipeline(config, device=device)

    # ── Load checkpoint (exact pattern from inference.py) ──────────────────────
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

    # ── Move to device (exact pattern from inference.py) ───────────────────────
    pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    vae = pipeline.vae

    # ── Load prompts ───────────────────────────────────────────────────────────
    with open(args.data_path, "r") as f:
        if args.data_path.endswith(".json"):
            data = json.load(f)
            prompts = [item["prompt"] if isinstance(item, dict) else item for item in data]
        else:
            prompts = [line.strip() for line in f if line.strip()]

    # Extended prompts: used for pipeline generation (richer conditioning).
    # Raw prompts are kept for Gemini scoring and file naming.
    extended_prompts = None
    if args.extended_prompt_path is not None:
        with open(args.extended_prompt_path, "r") as f:
            extended_prompts = [line.strip() for line in f if line.strip()]
        assert len(extended_prompts) == len(prompts), (
            f"extended_prompt_path has {len(extended_prompts)} lines but data_path has {len(prompts)}"
        )

    if local_rank == 0:
        print(f"Loaded {len(prompts)} prompts"
              + (f" + extended prompts" if extended_prompts else ""))

    # ── Load Gemini scorer (rank 0 only) ───────────────────────────────────────
    scorer = None
    if local_rank == 0:
        scorer = MultiObjectiveScorer(device=device, use_gemini=args.use_gemini)

    # ── SMC config ─────────────────────────────────────────────────────────────
    smc_config = SMCConfig(
        block_sizes=args.smc_block_sizes,
        initial_particles=args.smc_initial_particles,
        top_k=args.smc_top_k,
        branch_factor=args.smc_branch_factor,
        max_gemini_stage1_workers=args.max_gemini_stage1_workers,
        max_gemini_stage2_workers=args.max_gemini_stage2_workers,
    )

    smc = SMCGuidedGeneration(
        pipeline=pipeline,
        vae=vae,
        scorer=scorer,
        config=smc_config,
        device=device,
        local_rank=local_rank,
        world_size=world_size,
        output_folder=args.output_folder,
    )

    # ── Run SMC for each prompt ────────────────────────────────────────────────
    # Get num_frame_per_block from loaded config (required for KV cache models)
    num_frame_per_block = getattr(config, 'num_frame_per_block', 1)

    if local_rank == 0:
        print(f"Pipeline requires blocks of {num_frame_per_block} frames (from config.num_frame_per_block)")
        print(f"independent_first_frame: {getattr(config, 'independent_first_frame', False)}")
        import sys
        sys.stdout.flush()

    # Validate that all SMC block sizes are multiples of num_frame_per_block
    for block_size in args.smc_block_sizes:
        if block_size % num_frame_per_block != 0:
            raise ValueError(
                f"SMC block size {block_size} is not a multiple of num_frame_per_block={num_frame_per_block}. "
                f"Adjust --smc_block_sizes to use multiples of {num_frame_per_block}. "
                f"Example for 21 frames with num_frame_per_block=3: --smc_block_sizes 6 6 6 "
                f"(3 fixed + 6 + 6 + 6 = 21)"
            )

    # Fixed initial block must also be a multiple of num_frame_per_block
    num_fixed_frames = num_frame_per_block
    total_frames = num_fixed_frames + sum(args.smc_block_sizes)

    assert total_frames == args.num_output_frames, (
        f"Block sizes {args.smc_block_sizes} sum to {sum(args.smc_block_sizes)}, "
        f"plus {num_fixed_frames} fixed frames = {total_frames} total, "
        f"but num_output_frames={args.num_output_frames}"
    )

    torch.manual_seed(args.seed)
    _, _, C, H, W = (1, total_frames, 16, args.height // 8, args.width // 8)  # latent dims

    for prompt_idx, prompt in enumerate(prompts):
        # prompt:            raw prompt → Gemini scoring + file naming
        # generation_prompt: extended prompt (if provided) → pipeline text conditioning
        generation_prompt = extended_prompts[prompt_idx] if extended_prompts else prompt

        if local_rank == 0:
            print(f"\n{'='*80}")
            print(f"Prompt {prompt_idx + 1}/{len(prompts)}: {prompt}")
            if generation_prompt != prompt:
                print(f"  (generation prompt: {generation_prompt})")

        # Generate prompt alignment checklist once per prompt (rank 0 only)
        # This will be reused for all evaluation stages for this prompt
        # Do this FIRST before expensive latent generation since it only needs the text prompt
        if local_rank == 0:
            print(f"\n[Gemini] Generating prompt alignment checklist...")
            prompt_alignment_checklist = scorer.generate_prompt_alignment_checklist(
                text_prompt=prompt,
                model_name="gemini-2.5-flash",
                max_output_tokens=1000,
                temperature=0.1,
            )
        else:
            prompt_alignment_checklist = ""

        # Generate the fixed initial block on rank 0 only, then broadcast to all ranks.
        # This block has num_fixed_frames frames (must be multiple of num_frame_per_block).
        # fix_first_frame_noise=True uses torch.manual_seed which only controls CPU RNG,
        # but randn_like on CUDA tensors draws from the CUDA RNG — so intermediate
        # denoising noise would still diverge across ranks. Broadcasting is the only
        # reliable way to guarantee all ranks share an identical initial block latent.
        if local_rank == 0:
            torch.manual_seed(args.seed)
            fixed_initial_noise = torch.randn(1, num_fixed_frames, C, H, W, dtype=torch.bfloat16).to(device)

            import sys
            print(f"\n[DEBUG] Generating fixed initial latents:", flush=True)
            print(f"  INPUT noise shape: {fixed_initial_noise.shape}", flush=True)
            print(f"  num_fixed_frames: {num_fixed_frames}", flush=True)
            sys.stdout.flush()

            with torch.no_grad():
                # Generate initial latent only - no decoding during initial generation
                _, fixed_initial_latent = pipeline.inference(
                    noise=fixed_initial_noise,
                    text_prompts=[generation_prompt],
                    return_latents=True,
                )

            print(f"  OUTPUT latent shape: {fixed_initial_latent.shape}", flush=True)
            sys.stdout.flush()
        else:
            fixed_initial_latent = torch.zeros(1, num_fixed_frames, C, H, W, device=device, dtype=torch.bfloat16)

        if world_size > 1:
            # Broadcast latent for next block's conditioning
            dist.broadcast(fixed_initial_latent, src=0)

        # Clear cache after generation/broadcast
        vae.model.clear_cache()
        torch.cuda.empty_cache()

        # Noise template (used for shape reference; per-particle noise generated inside SMC)
        # Remaining frames after fixed initial block
        remaining_frames = total_frames - num_fixed_frames
        noise_template = torch.zeros(
            1, remaining_frames, C, H, W,
            device=device, dtype=torch.bfloat16
        )

        # Run SMC:
        #   prompt            → Gemini evaluation (raw, what the video should show)
        #   text_prompts      → pipeline generation (extended, richer conditioning)
        best_video_path = smc.run(
            prompt=prompt,
            fixed_first_frame_latent=fixed_initial_latent,
            noise_template=noise_template,
            text_prompts=[generation_prompt],
            prompt_idx=prompt_idx + args.start_index,
            prompt_alignment_checklist=prompt_alignment_checklist,
        )

        # Copy best video to final output location
        if local_rank == 0 and best_video_path is not None:
            actual_idx = args.start_index + prompt_idx
            model = "ema" if args.use_ema else "regular"
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f"{actual_idx}_{model}.mp4")
            else:
                output_path = os.path.join(args.output_folder, f"{prompt[:180]}-{actual_idx}.mp4")

            # Copy the video file instead of re-encoding
            shutil.copy2(best_video_path, output_path)
            print(f"\n[SMC] ✅ Saved: {output_path}")

    if local_rank == 0:
        print(f"\n{'='*80}")
        print(f"SMC generation complete for all {len(prompts)} prompts.")
        print(f"{'='*80}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
