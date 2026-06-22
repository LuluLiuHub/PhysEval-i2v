from typing import List, Optional
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import clip
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from pipeline.vlm_scorer import MultiObjectiveScorer

from demo_utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Initialize multi-objective scorer
        self.scorer = MultiObjectiveScorer(device, vae=self.vae)

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        # Iterative sampling configuration
        self.iterative_sampling = getattr(args, 'iterative_sampling', False)
        self.num_candidates_per_iteration = getattr(args, 'num_candidates_per_iteration', 5)
        self.num_iterations = getattr(args, 'num_iterations', 3)
        self.refinement_noise_scale = getattr(args, 'refinement_noise_scale', 0.5)  # Scale for noise around best candidate
        self.refinement_distance = getattr(args, 'refinement_distance', 1.0)  # L2 distance for refined candidates

        print(f"KV inference with {self.num_frame_per_block} frames per block")
        if self.iterative_sampling:
            print(f"Iterative sampling enabled: {self.num_iterations} iterations, {self.num_candidates_per_iteration} candidates per iteration, distance={self.refinement_distance}")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        save_intermediates: bool = False,
        output_folder: str = None,
        fix_first_frame_noise: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts for generation (can be extended prompts).
            original_prompts (List[str]): The original short prompts for scoring (optional, defaults to text_prompts).
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: BLOCK-BY-BLOCK WORKFLOW
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        # Outer loop: iterate through frame blocks that need to be generated
        for block_idx, current_num_frames in enumerate(all_num_frames):
            if profile:
                block_start.record()

            # Standard generation
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Generate frames with standard denoising
            if save_intermediates:
                denoised_pred, intermediate_results = self._denoise_frames(
                    noisy_input, conditional_dict, current_num_frames,
                    batch_size, noise.device, current_start_frame,
                    save_intermediates=True,
                    fix_first_frame_noise=fix_first_frame_noise
                )
                # Update output with current block
                output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
                # Save intermediate videos with full accumulated frames
                self._save_intermediate_videos(intermediate_results, output_folder, output, current_start_frame + current_num_frames)
            else:
                denoised_pred = self._denoise_frames(
                    noisy_input, conditional_dict, current_num_frames,
                    batch_size, noise.device, current_start_frame,
                    fix_first_frame_noise=fix_first_frame_noise
                )

            # Step 3.2: record the model's output (if not already done for save_intermediates)
            if not save_intermediates:
                output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def score_frame_with_clip(self, frame_latent, text_prompt, clip_model, clip_preprocess):
        """Score a single frame latent using CLIP model"""
        # Decode the latent to pixel space
        with torch.no_grad():
            frame_pixel = self.vae.decode_to_pixel(frame_latent.unsqueeze(1), use_cache=False)  # Add time dimension
            frame_pixel = (frame_pixel * 0.5 + 0.5).clamp(0, 1)  # Normalize to [0,1]
            frame_pixel = frame_pixel.squeeze(1)  # Remove time dimension [batch, C, H, W]

            scores = []
            for i in range(frame_pixel.shape[0]):
                # Convert to PIL format for CLIP preprocessing
                frame = frame_pixel[i]  # [C, H, W]
                frame_pil = transforms.ToPILImage()(frame.cpu())
                frame_processed = clip_preprocess(frame_pil).unsqueeze(0).to(frame_latent.device)

                # Encode text and image
                import clip
                text_tokens = clip.tokenize([text_prompt]).to(frame_latent.device)

                image_features = clip_model.encode_image(frame_processed)
                text_features = clip_model.encode_text(text_tokens)

                # Normalize and compute similarity
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

                similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
                scores.append(similarity.item())

        return scores



    def collect_training_data(self, text_prompts, num_frames_per_prompt=10):
        """
        Collect training data for noise network by comprehensive search.
        Returns: (prev_noise, text_embeds, best_noise, scores) tuples
        """
        training_data = []

        for text_prompt in text_prompts:
            print(f"Collecting data for: {text_prompt}")

            # Get text embeddings
            with torch.no_grad():
                text_embeds = self.text_encoder([text_prompt])["prompt_embeds"]

            # Start with random previous noise
            prev_noise = torch.randn(1, 16, 60, 104, device=text_embeds.device)

            for frame_idx in range(num_frames_per_prompt):
                # Comprehensive noise search
                best_noise, all_candidates, all_scores = self.comprehensive_noise_search(
                    prev_noise, text_prompt, text_embeds, num_candidates=50
                )

                # Collect training data with combined score (includes aesthetic)
                training_data.append({
                    'prev_noise': prev_noise.clone(),
                    'text_embeds': text_embeds.clone(),
                    'target_noise': best_noise['noise'].clone(),
                    'scores': best_noise['scores'],
                    'text_prompt': text_prompt
                })

                # Update prev_noise for next frame
                prev_noise = best_noise['noise']

        return training_data

    def _denoise_single_step(self, noisy_input, conditional_dict, current_num_frames, batch_size, device, current_start_frame, step_index=0):
        """Perform only a single denoising step for candidate evaluation"""
        current_timestep = self.denoising_step_list[step_index]

        # Set current timestep
        timestep = torch.ones(
            [batch_size, current_num_frames],
            device=device,
            dtype=torch.int64) * current_timestep

        # Run single denoising step
        _, denoised_pred = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=current_start_frame * self.frame_seq_length
        )

        return denoised_pred

    def _denoise_frames(self, noisy_input, conditional_dict, current_num_frames, batch_size, device, current_start_frame, save_intermediates=False, update_cache=True, fix_first_frame_noise=False):
        """Perform the denoising steps for a given noisy input

        Args:
            fix_first_frame_noise: If True and current_num_frames==1 (first frame),
                                   use fixed seed for intermediate noise across refinement steps
        """
        intermediate_results = [] if save_intermediates else None

        for index, current_timestep in enumerate(self.denoising_step_list):
            # set current timestep
            timestep = torch.ones(
                [batch_size, current_num_frames],
                device=device,
                dtype=torch.int64) * current_timestep

            if index < len(self.denoising_step_list) - 1:
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    update_cache=update_cache
                )

                # Save intermediate result if requested
                if save_intermediates:
                    intermediate_results.append({
                        'timestep': current_timestep.item(),
                        'denoised_pred': denoised_pred.clone().cpu(),
                        'step_index': index
                    })
                next_timestep = self.denoising_step_list[index + 1]
                # Generate intermediate noise for next refinement step
                # If fix_first_frame_noise is True and this is the first frame (current_num_frames==1),
                # use a fixed seed to maintain consistency across candidates
                if fix_first_frame_noise and current_num_frames == 1:
                    # Use fixed seed for first frame intermediate noise
                    torch.manual_seed(42 + index)
                    intermediate_noise = torch.randn_like(denoised_pred.flatten(0, 1))
                else:
                    # Random intermediate noise for other frames
                    intermediate_noise = torch.randn_like(denoised_pred.flatten(0, 1))

                noisy_input = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    intermediate_noise,
                    next_timestep * torch.ones(
                        [batch_size * current_num_frames], device=device, dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                # for getting final output
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    update_cache=update_cache
                )

                # Save final result if requested
                if save_intermediates:
                    intermediate_results.append({
                        'timestep': current_timestep.item(),
                        'denoised_pred': denoised_pred.clone().cpu(),
                        'step_index': index
                    })

        if save_intermediates:
            return denoised_pred, intermediate_results
        else:
            return denoised_pred

    def _save_intermediate_videos(self, intermediate_results, output_folder, full_output=None, total_frames=None):
        """Save intermediate denoising steps as MP4 videos"""
        import os
        from torchvision.io import write_video
        from einops import rearrange

        if not intermediate_results or not output_folder:
            return

        for result in intermediate_results:
            # Use full accumulated output if available, otherwise just the current block
            if full_output is not None and total_frames is not None:
                # Save the full accumulated video up to current point
                accumulated_latents = full_output[:, :total_frames]
                device = next(self.vae.parameters()).device
                video_data = self.vae.decode_to_pixel(accumulated_latents.to(device), use_cache=False)
                video_data = (video_data * 0.5 + 0.5).clamp(0, 1)
                video_data = 255.0 * rearrange(video_data, 'b t c h w -> b t h w c').cpu()
                filename = f"intermediate_step_{result['step_index']}_timestep_{result['timestep']}_full.mp4"
            elif 'video' in result:
                # Use the stored video data
                video_data = result['video']
                filename = result['filename']
            else:
                # Convert latent to video (just current block)
                denoised_pred = result['denoised_pred']
                device = next(self.vae.parameters()).device
                video_data = self.vae.decode_to_pixel(denoised_pred.to(device), use_cache=False)
                video_data = (video_data * 0.5 + 0.5).clamp(0, 1)
                video_data = 255.0 * rearrange(video_data, 'b t c h w -> b t h w c').cpu()
                filename = f"intermediate_step_{result['step_index']}_timestep_{result['timestep']}.mp4"

            # Save the video file
            output_path = os.path.join(output_folder, filename)
            write_video(output_path, video_data[0], fps=16)  # Use first batch item
