#!/usr/bin/env python3
"""
Qwen Prompt Optimizer for Video Generation
Optimizes prompts before feeding them to the video generation pipeline
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time


class QwenPromptOptimizer:
    """Qwen-based prompt optimizer for video generation"""

    def __init__(self, model_name="Qwen/Qwen2.5-Coder-32B-Instruct", device="cuda"):
        self.device = device
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def _initialize_qwen(self):
        """Initialize Qwen model for prompt optimization"""
        if self.model is None:
            print(f"Initializing Qwen model: {self.model_name}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load model with 4-bit quantization for efficient memory usage
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True,
                trust_remote_code=True
            )
            self.model.eval()
            print(f"Qwen model initialized successfully")

    def optimize_prompt(self, original_prompt, max_total_length=1024):
        """
        Optimize a prompt for video generation using Qwen

        Args:
            original_prompt (str): Original prompt to optimize
            max_total_length (int): Maximum total sequence length (input + output)

        Returns:
            str: Optimized prompt
        """
        self._initialize_qwen()

        start_time = time.time()

        # Create optimization instruction
        system_prompt = """You are an expert prompt optimizer for video generation models. Your task is to enhance prompts to make them more detailed, specific, and suitable for high-quality video generation.

Guidelines:
1. Add visual details (lighting, colors, textures, composition)
2. Specify camera movements and angles when appropriate
3. Include temporal elements (motion, progression, changes)
4. Maintain the core concept while making it more vivid
5. Keep the prompt concise but descriptive (maximum 2-3 sentences)
6. Focus on elements that will improve video quality and coherence
7. Write a complete, finished response - do not cut off mid-sentence

Original prompt: {original_prompt}

Enhanced prompt:"""

        prompt = system_prompt.format(original_prompt=original_prompt)

        try:
            # Prepare conversation format for Qwen
            messages = [
                {"role": "system", "content": "You are an expert prompt optimizer for video generation."},
                {"role": "user", "content": f"Enhance this prompt for video generation: '{original_prompt}'. Make it more detailed and specific while keeping the core concept. Focus on visual details, camera work, and temporal elements."}
            ]

            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_length=max_total_length,  # Maximum total sequence length (including input)
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id  # Ensure proper stopping
                )

            # Decode response
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            optimized_prompt = response.strip()

            end_time = time.time()
            optimization_time = end_time - start_time

            print(f"[QWEN] Original: '{original_prompt}'")
            print(f"[QWEN] Optimized: '{optimized_prompt}'")
            print(f"[QWEN] Optimization took {optimization_time:.2f} seconds")

            return optimized_prompt

        except Exception as e:
            print(f"[QWEN ERROR] Prompt optimization failed: {e}")
            print(f"[QWEN] Using original prompt: {original_prompt}")
            return original_prompt

    def optimize_batch(self, prompts, max_new_tokens=150):
        """
        Optimize a batch of prompts

        Args:
            prompts (list): List of prompts to optimize
            max_new_tokens (int): Maximum tokens for optimization

        Returns:
            list: List of optimized prompts
        """
        optimized_prompts = []

        for i, prompt in enumerate(prompts):
            print(f"[QWEN] Optimizing prompt {i+1}/{len(prompts)}")
            optimized = self.optimize_prompt(prompt, max_new_tokens)
            optimized_prompts.append(optimized)

        return optimized_prompts


def test_qwen_optimizer():
    """Test function for Qwen optimizer"""
    optimizer = QwenPromptOptimizer()

    test_prompts = [
        "A cat walking",
        "Ocean waves",
        "City street at night"
    ]

    for prompt in test_prompts:
        optimized = optimizer.optimize_prompt(prompt)
        print(f"Original: {prompt}")
        print(f"Optimized: {optimized}")
        print("-" * 50)


if __name__ == "__main__":
    test_qwen_optimizer()