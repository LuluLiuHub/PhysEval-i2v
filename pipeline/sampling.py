"""
Noise sampling strategies for video generation inference guidance.
Contains various noise generation methods for candidate evaluation.
"""

import torch
from typing import List, Optional


class NoiseSampler:
    """Noise sampling strategies for inference guidance"""

    def __init__(self, device='cuda'):
        self.device = device

    def generate_widespread_noise_candidates(self, batch_size, num_frames, num_channels, height, width, device, dtype, num_candidates=20, noise_generation_type="mixed"):
        """
        Generate widespread noise candidates covering maximum noise space.

        Args:
            batch_size: Batch size
            num_frames: Number of frames
            num_channels: Number of channels (typically 16 for latent space)
            height: Height dimension
            width: Width dimension
            device: Device to generate noise on
            dtype: Data type for noise
            num_candidates: Number of candidates to generate
            noise_generation_type: Type of noise generation strategy
                - "pure_random": Completely random candidates
                - "mixed": Mix of gaussian/uniform/scaled/structured
                - "low_discrepancy": Quasi-random sequences

        Returns:
            List of noise candidates
        """
        candidates = []

        if noise_generation_type == "pure_random":
            # Generate completely random noise candidates
            for i in range(num_candidates):
                noise = torch.randn(batch_size, num_frames, num_channels, height, width,
                                  device=device, dtype=dtype)
                candidates.append(noise)

        elif noise_generation_type == "mixed":
            # Mix of different noise strategies
            strategies = ["gaussian", "uniform", "scaled_gaussian", "structured"]

            for i in range(num_candidates):
                strategy = strategies[i % len(strategies)]

                if strategy == "gaussian":
                    # Standard Gaussian noise
                    noise = torch.randn(batch_size, num_frames, num_channels, height, width,
                                      device=device, dtype=dtype)
                elif strategy == "uniform":
                    # Uniform noise in [-1, 1]
                    noise = torch.rand(batch_size, num_frames, num_channels, height, width,
                                     device=device, dtype=dtype) * 2 - 1
                elif strategy == "scaled_gaussian":
                    # Gaussian with different scales
                    scale = 0.5 + (i % 4) * 0.5  # Scales: 0.5, 1.0, 1.5, 2.0
                    noise = torch.randn(batch_size, num_frames, num_channels, height, width,
                                      device=device, dtype=dtype) * scale
                elif strategy == "structured":
                    # Structured noise with patterns
                    noise = torch.randn(batch_size, num_frames, num_channels, height, width,
                                      device=device, dtype=dtype)
                    # Add some structure by smoothing in spatial dimensions
                    if height > 4 and width > 4:
                        kernel_size = 3
                        padding = kernel_size // 2
                        # Simple spatial smoothing
                        noise_flat = noise.view(-1, 1, height, width)
                        noise_smooth = torch.nn.functional.avg_pool2d(
                            noise_flat, kernel_size, stride=1, padding=padding
                        )
                        noise = noise_smooth.view(batch_size, num_frames, num_channels, height, width)

                candidates.append(noise)

        elif noise_generation_type == "low_discrepancy":
            # Quasi-random sequences for better coverage
            for i in range(num_candidates):
                # Simple quasi-random approach using different seeds
                generator = torch.Generator(device=device)
                generator.manual_seed(i * 12345)  # Different seed for each candidate
                noise = torch.randn(batch_size, num_frames, num_channels, height, width,
                                  generator=generator, device=device, dtype=dtype)
                candidates.append(noise)

        else:
            # Fallback to pure random
            for i in range(num_candidates):
                noise = torch.randn(batch_size, num_frames, num_channels, height, width,
                                  device=device, dtype=dtype)
                candidates.append(noise)

        print(f"Generated {len(candidates)} noise candidates using '{noise_generation_type}' strategy")
        return candidates

    def generate_refined_noise_candidates(self, best_noise, num_candidates, distance=1.0):
        """
        Generate noise candidates at a specific distance from the best candidate.

        Args:
            best_noise: Best noise candidate from previous iteration
            num_candidates: Number of refined candidates to generate
            distance: L2 distance from the best candidate

        Returns:
            List of refined noise candidates
        """
        refined_candidates = []

        for i in range(num_candidates):
            if i == 0:
                # Always include the best candidate unchanged
                refined_candidates.append(best_noise.clone())
            else:
                # Generate random direction vector
                direction = torch.randn_like(best_noise)

                # Normalize to unit vector
                direction_norm = torch.norm(direction)
                if direction_norm > 0:
                    direction = direction / direction_norm
                else:
                    # Fallback if norm is zero
                    direction = torch.ones_like(best_noise) / torch.sqrt(torch.tensor(best_noise.numel(), dtype=best_noise.dtype))

                # Generate candidate at specified distance
                refined_noise = best_noise + distance * direction
                refined_candidates.append(refined_noise)

        print(f"Generated {len(refined_candidates)} refined candidates at distance {distance:.3f}")
        return refined_candidates

    def generate_spherical_candidates(self, center_noise, num_candidates, radius=1.0):
        """
        Generate candidates uniformly distributed on a sphere around center_noise.

        Args:
            center_noise: Center point for candidate generation
            num_candidates: Number of candidates to generate
            radius: Radius of the sphere

        Returns:
            List of noise candidates on sphere surface
        """
        candidates = []

        for i in range(num_candidates):
            if i == 0:
                # Include center point
                candidates.append(center_noise.clone())
            else:
                # Generate random direction and normalize
                direction = torch.randn_like(center_noise)
                direction = direction / torch.norm(direction)

                # Place on sphere surface
                candidate = center_noise + radius * direction
                candidates.append(candidate)

        return candidates

    def generate_grid_candidates(self, center_noise, num_candidates, step_size=0.5):
        """
        Generate candidates on a regular grid around center_noise.

        Args:
            center_noise: Center point for candidate generation
            num_candidates: Number of candidates to generate
            step_size: Step size for grid spacing

        Returns:
            List of noise candidates on grid
        """
        candidates = [center_noise.clone()]  # Always include center

        # Generate grid offsets in the first few dimensions
        remaining = num_candidates - 1
        dims_to_perturb = min(8, center_noise.numel())  # Limit perturbation dimensions

        for i in range(remaining):
            candidate = center_noise.clone()
            flat_candidate = candidate.flatten()

            # Perturb specific dimensions in a grid pattern
            for dim_idx in range(dims_to_perturb):
                if i & (1 << dim_idx):  # Binary pattern for grid
                    flat_candidate[dim_idx] += step_size
                else:
                    flat_candidate[dim_idx] -= step_size

            candidates.append(flat_candidate.view_as(center_noise))

        return candidates[:num_candidates]

    def adaptive_sampling(self, center_noise, num_candidates, scores_history=None, adaptive_radius=True):
        """
        Adaptive sampling that adjusts strategy based on previous results.

        Args:
            center_noise: Current best noise candidate
            num_candidates: Number of candidates to generate
            scores_history: List of previous scores for adaptation
            adaptive_radius: Whether to adapt radius based on score improvement

        Returns:
            List of adaptively generated candidates
        """
        # Default radius
        radius = 1.0

        # Adapt radius based on score improvement
        if adaptive_radius and scores_history and len(scores_history) > 1:
            recent_improvement = scores_history[-1] - scores_history[-2] if len(scores_history) > 1 else 0
            if recent_improvement > 0.1:
                radius = 0.5  # Smaller radius for fine-tuning
            elif recent_improvement < 0.01:
                radius = 2.0  # Larger radius for exploration

        # Use distance-based sampling with adaptive radius
        return self.generate_refined_noise_candidates(center_noise, num_candidates, radius)

    def particle_filter_importance_sampling(
        self,
        pipeline,
        text_prompt,
        image_scorer,
        batch_size,
        num_frames,
        num_channels,
        height,
        width,
        device,
        dtype,
        num_particles=20,
        num_iterations=5,
        comparison_batch_size=4,
        resample_ratio=0.5,
        perturbation_scale=0.3
    ):
        """
        Particle filter based importance sampling using LLaVA-OneVision for first frame evaluation.

        Process:
        1. Initialize population of noise particles
        2. For each iteration:
           a. Generate first frames for all particles
           b. Evaluate particles in batches using LLaVA-OneVision comparison
           c. Assign weights based on ranking
           d. Resample particles (keep best, resample worst)
           e. Perturb particles for exploration
        3. Return final particle population

        Args:
            pipeline: Video generation pipeline for first frame generation
            text_prompt: Text prompt for scoring
            image_scorer: LLaVA-OneVision scorer for first frame images
            batch_size: Batch size (typically 1)
            num_frames: Total number of frames
            num_channels: Latent channels (16)
            height: Latent height (60)
            width: Latent width (104)
            device: CUDA device
            dtype: Data type (bfloat16)
            num_particles: Number of particles to maintain
            num_iterations: Number of particle filter iterations
            comparison_batch_size: Batch size for LLaVA comparison (4-8 images at once)
            resample_ratio: Fraction of particles to resample each iteration
            perturbation_scale: Scale of perturbation for exploration

        Returns:
            List of final particle noise candidates
        """
        print(f"\n🎯 Starting Particle Filter Importance Sampling:")
        print(f"   Particles: {num_particles}")
        print(f"   Iterations: {num_iterations}")
        print(f"   Comparison batch size: {comparison_batch_size}")
        print(f"   Resample ratio: {resample_ratio}")

        # Initialize particles randomly
        particles = []
        for _ in range(num_particles):
            noise = torch.randn(
                batch_size, num_frames, num_channels, height, width,
                device=device, dtype=dtype
            )
            particles.append(noise)

        particle_weights = torch.ones(num_particles, device=device) / num_particles
        print(f"  ✓ Initialized {num_particles} particles")

        # Particle filter iterations
        for iteration in range(num_iterations):
            print(f"\n  📍 Iteration {iteration + 1}/{num_iterations}")

            # Step 1: Evaluate all particles
            print(f"     Evaluating {num_particles} particles...")
            particle_rankings = []

            # Evaluate in batches using LLaVA comparison
            for start_idx in range(0, num_particles, comparison_batch_size):
                end_idx = min(start_idx + comparison_batch_size, num_particles)
                batch_particles = particles[start_idx:end_idx]

                # Generate first frames for this batch
                first_frame_images = []
                for particle in batch_particles:
                    with torch.no_grad():
                        first_frame_noise = particle[:, :1, :, :, :]
                        # Denoise first frame
                        latent = pipeline.generate_first_frame_latent(
                            first_frame_noise,
                            text_prompts=[text_prompt]
                        )
                        # Decode to image
                        image = pipeline.vae.decode_to_pixel(latent.unsqueeze(1))
                        image = image.squeeze(1)
                        first_frame_images.append(image)

                # Compare batch with LLaVA-OneVision
                best_local_idx = image_scorer.compare_images(first_frame_images, text_prompt)

                # Assign rankings (best gets highest rank)
                for local_idx in range(len(batch_particles)):
                    if local_idx == best_local_idx:
                        rank = len(batch_particles)  # Highest rank for best
                    else:
                        rank = len(batch_particles) - abs(local_idx - best_local_idx)
                    particle_rankings.append((start_idx + local_idx, rank))

            # Step 2: Update weights based on rankings
            for particle_idx, rank in particle_rankings:
                particle_weights[particle_idx] = rank

            # Normalize weights
            particle_weights = particle_weights / particle_weights.sum()

            best_particle_idx = torch.argmax(particle_weights).item()
            print(f"     Best particle: #{best_particle_idx} (weight: {particle_weights[best_particle_idx]:.4f})")

            # Step 3: Resample particles
            if iteration < num_iterations - 1:  # Don't resample on last iteration
                num_to_resample = int(num_particles * resample_ratio)

                # Keep top particles, resample bottom ones
                sorted_indices = torch.argsort(particle_weights, descending=True)
                keep_indices = sorted_indices[:num_particles - num_to_resample]

                # Resample from top particles
                resample_from = torch.multinomial(
                    particle_weights[keep_indices],
                    num_samples=num_to_resample,
                    replacement=True
                )

                # Create new particle set
                new_particles = []
                for idx in keep_indices:
                    new_particles.append(particles[idx])

                for resample_idx in resample_from:
                    # Clone from good particle and add perturbation
                    source_particle = particles[keep_indices[resample_idx]]
                    perturbation = torch.randn_like(source_particle) * perturbation_scale
                    new_particle = source_particle + perturbation
                    new_particles.append(new_particle)

                particles = new_particles
                particle_weights = torch.ones(num_particles, device=device) / num_particles

                print(f"     Resampled {num_to_resample} particles")

        print(f"\n  ✓ Particle filter completed")
        print(f"  Final particles sorted by weight:")
        sorted_indices = torch.argsort(particle_weights, descending=True)
        for i, idx in enumerate(sorted_indices[:min(5, num_particles)]):
            print(f"     {i+1}. Particle #{idx.item()} (weight: {particle_weights[idx]:.4f})")

        return particles

    def generate_fixed_first_frame_candidates(
        self,
        first_frame_noise,
        batch_size,
        num_frames,
        num_channels,
        height,
        width,
        device,
        dtype,
        num_candidates=10,
        noise_generation_type="mixed"
    ):
        """
        Generate noise candidates with a fixed first frame and varying subsequent frames.

        This is useful when you've already optimized the first frame (e.g., via particle filter)
        and want to explore different temporal continuations.

        Args:
            first_frame_noise: Fixed first frame noise tensor [B, 1, C, H, W]
            batch_size: Batch size
            num_frames: Total number of frames (including the fixed first frame)
            num_channels: Number of channels (16 for latent space)
            height: Height dimension
            width: Width dimension
            device: Device to generate noise on
            dtype: Data type for noise
            num_candidates: Number of candidates to generate
            noise_generation_type: Type of noise generation for subsequent frames
                - "pure_random": Random noise for each candidate
                - "mixed": Mix of different noise strategies
                - "temporal_smooth": Smoothly varying from first frame

        Returns:
            List of noise candidates with fixed first frame
        """
        candidates = []

        print(f"🎬 Generating {num_candidates} candidates with fixed first frame...")
        print(f"   First frame shape: {first_frame_noise.shape}")
        print(f"   Remaining frames to generate: {num_frames - 1}")

        if noise_generation_type == "pure_random":
            # Generate random noise for remaining frames
            for i in range(num_candidates):
                # Generate noise for frames 2 to num_frames
                remaining_noise = torch.randn(
                    batch_size, num_frames - 1, num_channels, height, width,
                    device=device, dtype=dtype
                )
                # Concatenate fixed first frame with varying remaining frames
                full_noise = torch.cat([first_frame_noise, remaining_noise], dim=1)
                candidates.append(full_noise)

        elif noise_generation_type == "mixed":
            # Mix of different noise strategies for temporal diversity
            strategies = ["gaussian", "smooth_transition", "varying_scale", "structured_temporal"]

            for i in range(num_candidates):
                strategy = strategies[i % len(strategies)]

                if strategy == "gaussian":
                    # Standard Gaussian noise for remaining frames
                    remaining_noise = torch.randn(
                        batch_size, num_frames - 1, num_channels, height, width,
                        device=device, dtype=dtype
                    )

                elif strategy == "smooth_transition":
                    # Smoothly transition from first frame
                    remaining_noise = torch.randn(
                        batch_size, num_frames - 1, num_channels, height, width,
                        device=device, dtype=dtype
                    )
                    # Add gradual drift from first frame
                    for t in range(num_frames - 1):
                        alpha = (t + 1) / num_frames  # Gradually reduce influence
                        remaining_noise[:, t] = (1 - alpha) * first_frame_noise[:, 0] + alpha * remaining_noise[:, t]

                elif strategy == "varying_scale":
                    # Gaussian with different scales over time
                    remaining_noise = torch.randn(
                        batch_size, num_frames - 1, num_channels, height, width,
                        device=device, dtype=dtype
                    )
                    for t in range(num_frames - 1):
                        scale = 0.5 + (t / (num_frames - 1))  # Scale increases over time
                        remaining_noise[:, t] = remaining_noise[:, t] * scale

                elif strategy == "structured_temporal":
                    # Structured noise with temporal coherence
                    remaining_noise = torch.randn(
                        batch_size, num_frames - 1, num_channels, height, width,
                        device=device, dtype=dtype
                    )
                    # Apply temporal smoothing
                    if num_frames > 2:
                        kernel_size = min(3, num_frames - 1)
                        if kernel_size >= 2:
                            # Simple temporal smoothing
                            smoothed = remaining_noise.clone()
                            for t in range(1, num_frames - 2):
                                smoothed[:, t] = (remaining_noise[:, t-1] + remaining_noise[:, t] + remaining_noise[:, t+1]) / 3.0
                            remaining_noise = smoothed

                # Concatenate fixed first frame with varying remaining frames
                full_noise = torch.cat([first_frame_noise, remaining_noise], dim=1)
                candidates.append(full_noise)

        elif noise_generation_type == "temporal_smooth":
            # Generate candidates that smoothly vary temporally from first frame
            for i in range(num_candidates):
                remaining_noise = []

                for t in range(num_frames - 1):
                    # Generate noise that gradually drifts from first frame
                    base_noise = torch.randn(
                        batch_size, 1, num_channels, height, width,
                        device=device, dtype=dtype
                    )

                    # Blend with first frame based on temporal distance
                    blend_factor = (t + 1) / num_frames
                    noise_strength = 0.3 + 0.7 * blend_factor  # Start at 30%, end at 100%

                    frame_noise = (1 - noise_strength) * first_frame_noise + noise_strength * base_noise
                    remaining_noise.append(frame_noise)

                remaining_noise = torch.cat(remaining_noise, dim=1)
                full_noise = torch.cat([first_frame_noise, remaining_noise], dim=1)
                candidates.append(full_noise)

        else:
            # Fallback to pure random
            for i in range(num_candidates):
                remaining_noise = torch.randn(
                    batch_size, num_frames - 1, num_channels, height, width,
                    device=device, dtype=dtype
                )
                full_noise = torch.cat([first_frame_noise, remaining_noise], dim=1)
                candidates.append(full_noise)

        print(f"✓ Generated {len(candidates)} candidates with fixed first frame using '{noise_generation_type}' strategy")
        return candidates