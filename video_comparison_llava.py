#!/usr/bin/env python3
"""
Video Comparison Script using LLaVA-OneVision
Multi-turn conversation: analyze each video, then compare.
"""

import argparse
import torch
import os
from pathlib import Path
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(description="Compare videos using LLaVA-OneVision multi-turn conversation")
    parser.add_argument("--video_paths", type=str, nargs='+', required=True,
                        help="Paths to video files to compare")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for comparison")
    parser.add_argument("--model_name", type=str,
                        default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
                        help="LLaVA-OneVision model name")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Number of frames to sample from each video")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional output file to save results")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="Maximum number of tokens to generate")

    return parser.parse_args()


def compare_videos_multiturn(video_paths, prompt, model_name, num_frames=8, max_new_tokens=200):
    """Compare videos using multi-turn conversation

    Args:
        video_paths: List of video file paths
        prompt: Text prompt for comparison
        model_name: LLaVA-OneVision model name
        num_frames: Number of frames per video
        max_new_tokens: Maximum tokens for generation

    Returns:
        Best video index (0-based), conversation history
    """
    print(f"\n🎬 Loading LLaVA-OneVision model: {model_name}")

    # Load model and processor
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()

    print(f"✓ Model loaded")

    # Verify all videos exist
    for path in video_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video not found: {path}")

    print(f"\n📹 Creating multi-turn conversation with {len(video_paths)} videos...")

    # Build multi-turn conversation
    conversation = []

    # Create a single comprehensive question with all videos
    print(f"\n  Creating comprehensive analysis and comparison question")

    content = []

    # Add all videos
    for i, video_path in enumerate(video_paths):
        print(f"  Adding video {i+1}: {os.path.basename(video_path)}")
        content.append({"type": "video", "video": {"path": video_path}})

    # Add comprehensive question
    analysis_text = f"""I'm showing you {len(video_paths)} videos based on the prompt: "{prompt}"

For EACH video, analyze:
- Physical realism and mechanics (gravity, motion, forces)
- Object properties (materials, thermodynamics, interactions)
- Spatial relationships and composition
- Alignment with the prompt
- Motion smoothness and temporal consistency
- Visual quality

Then select which video best matches the prompt.

Please respond in this format:

Video 1 Analysis: [your analysis]

Video 2 Analysis: [your analysis]
"""

    if len(video_paths) > 2:
        for i in range(3, len(video_paths) + 1):
            analysis_text += f"\nVideo {i} Analysis: [your analysis]\n"

    analysis_text += f"""
Best video: [number 1-{len(video_paths)}]
Reason: [brief explanation comparing the videos]"""

    content.append({"type": "text", "text": analysis_text})

    conversation.append({
        "role": "user",
        "content": content,
    })

    print(f"\n🔍 Processing multi-turn conversation...")

    # Apply chat template
    inputs = processor.apply_chat_template(
        [conversation],
        num_frames=num_frames,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )

    # Move inputs to device
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Generate response
    print(f"  Generating response (max {max_new_tokens} tokens)...")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode
    full_response = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    print(f"\n📊 Full Conversation Response:")
    print("=" * 80)
    print(full_response)
    print("=" * 80)

    # Extract the final comparison part
    # The response contains all turns, we want the last assistant response
    parts = full_response.split("assistant")
    if len(parts) > 1:
        final_response = parts[-1].strip()
    else:
        final_response = full_response

    print(f"\n🏆 Final Comparison:")
    print("-" * 80)
    print(final_response)
    print("-" * 80)

    # Extract best video number
    import re
    best_match = re.search(r'Best video:\s*(\d+)', final_response, re.IGNORECASE)
    if best_match:
        best_idx = int(best_match.group(1)) - 1
    else:
        # Fallback patterns
        patterns = [
            r'video\s+(\d+)',
            r'\b([1-9])\b'
        ]
        best_idx = 0
        for pattern in patterns:
            match = re.search(pattern, final_response, re.IGNORECASE)
            if match:
                num = int(match.group(1))
                if 1 <= num <= len(video_paths):
                    best_idx = num - 1
                    break

    # Validate
    best_idx = max(0, min(best_idx, len(video_paths) - 1))

    return best_idx, full_response


def main():
    args = parse_args()

    print("=" * 80)
    print("Video Comparison with LLaVA-OneVision (Multi-turn Conversation)")
    print("=" * 80)
    print(f"Videos to compare: {len(args.video_paths)}")
    for i, path in enumerate(args.video_paths):
        print(f"  {i+1}. {os.path.basename(path)}")
    print(f"Prompt: {args.prompt}")
    print(f"Frames per video: {args.num_frames}")
    print("=" * 80)

    # Run comparison
    best_idx, conversation = compare_videos_multiturn(
        args.video_paths,
        args.prompt,
        args.model_name,
        args.num_frames,
        args.max_new_tokens
    )

    print(f"\n✅ FINAL RESULT:")
    print(f"Best Video: #{best_idx + 1}")
    print(f"File: {os.path.basename(args.video_paths[best_idx])}")
    print(f"Path: {args.video_paths[best_idx]}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(f"Video Comparison Results - LLaVA-OneVision Multi-turn\n")
            f.write(f"=" * 80 + "\n")
            f.write(f"Prompt: {args.prompt}\n")
            f.write(f"Videos: {len(args.video_paths)}\n\n")

            for i, path in enumerate(args.video_paths):
                f.write(f"  {i+1}. {path}\n")
            f.write("\n")

            f.write(f"Full Conversation:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{conversation}\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"Best Video: #{best_idx + 1}\n")
            f.write(f"Path: {args.video_paths[best_idx]}\n")

        print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
