#!/usr/bin/env python3
"""
VLM Scoring Module for Video Generation Quality Assessment
Supports LLaVA models for multi-frame video evaluation
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
import PIL.Image
import os
import json
from typing import List, Optional
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-genai not installed. Gemini-based comparison will not be available.")



class MultiObjectiveScorer:
    """
    Multi-objective scorer combining VLM, temporal consistency, and motion smoothness.

    Supports both paid Gemini API and FREE open-source models (Qwen) for all tasks.
    """

    def __init__(self, device, vae=None, use_video_scoring=False, use_videollama=True, use_gemini=False,
                 use_qwen_vl=False, qwen_vl_model='Qwen/Qwen3-VL-30B-A3B-Instruct',
                 qwen_vl_fps=1.0, qwen_vl_max_pixels=480*720,
                 use_stage1_model=False, stage1_model='Qwen/Qwen3-VL-8B-Instruct',
                 use_free_llm_for_checklist=False, free_llm_model='Qwen/Qwen2.5-Coder-32B-Instruct'):
        """
        Initialize the Multi-Objective Scorer.

        Args:
            device: PyTorch device for model inference
            vae: Optional VAE model for latent-space operations
            use_video_scoring: Enable video quality scoring
            use_videollama: Enable VideoLLaMA model support
            use_gemini: Enable Gemini API for video comparison (requires GEMINI_API_KEY)
            use_qwen_vl: Use FREE Qwen3-VL-30B for Stage 1 & 2 video understanding (replaces Gemini)
                        Completely open-source! Best for video analysis tasks.
            qwen_vl_model: Hugging Face model for Qwen-VL. Default:
                          - 'Qwen/Qwen3-VL-30B-A3B-Instruct' (30B params, best quality for video)
                          - Alternative: 'Qwen/Qwen2-VL-7B-Instruct' (~14GB VRAM, lighter)
            use_free_llm_for_checklist: Use FREE Qwen2.5 for prompt checklist generation
                                       and Stage 3 tournament (text-only tasks)
            free_llm_model: Hugging Face model for free text LLM:
                           - 'Qwen/Qwen2.5-Coder-32B-Instruct' (default, excellent quality, ~20GB VRAM with 4-bit)
                           - 'Qwen/Qwen2.5-7B-Instruct' (lighter, faster)
                           - 'Qwen/Qwen2.5-3B-Instruct' (fastest, minimal VRAM)

        Example Usage:
            # Use Gemini for everything (costs API credits):
            scorer = MultiObjectiveScorer(device, use_gemini=True)

            # FULLY FREE - Qwen3-VL-30B for video, Qwen2.5-Coder-32B for text (NO API costs!):
            scorer = MultiObjectiveScorer(device,
                                         use_qwen_vl=True,
                                         qwen_vl_model='Qwen/Qwen3-VL-30B-A3B-Instruct',
                                         use_free_llm_for_checklist=True,
                                         free_llm_model='Qwen/Qwen2.5-Coder-32B-Instruct')

            # Hybrid - Qwen3-VL for video, Gemini for text:
            scorer = MultiObjectiveScorer(device,
                                         use_qwen_vl=True,
                                         use_gemini=True)
        """
        self.device = device
        self.use_video_scoring = use_video_scoring
        self.use_gemini = use_gemini and GEMINI_AVAILABLE
        self.gemini_client = None

        # Qwen-VL settings for video understanding (Stage 2 forensics)
        self.use_qwen_vl = use_qwen_vl
        self.qwen_vl_model = qwen_vl_model
        self.qwen_vl_fps = qwen_vl_fps
        self.qwen_vl_max_pixels = qwen_vl_max_pixels
        self.qwen_vl = None
        self.qwen_vl_processor = None

        # Stage 1 model settings (lightweight 7B for frame/consistency checking)
        self.use_stage1_model = use_stage1_model
        self.stage1_model_name = stage1_model
        self.stage1_models = {}  # Dict of {gpu_id: (model, processor, device)}

        # Free LLM settings for text generation (checklist + Stage 3 tournament)
        self.use_free_llm_for_checklist = use_free_llm_for_checklist
        self.free_llm_model = free_llm_model
        self.free_llm = None
        self.free_llm_tokenizer = None

        # Initialize Gemini client if requested
        if self.use_gemini:
            self._initialize_gemini()

        # Initialize Qwen-VL for video tasks if requested
        if self.use_qwen_vl:
            self._initialize_qwen_vl()

        # Initialize Stage 1 models on multiple GPUs if requested
        if self.use_stage1_model:
            self._initialize_stage1_models()

        # Initialize free LLM for text tasks if requested
        if self.use_free_llm_for_checklist:
            self._initialize_free_llm()

    def _initialize_gemini(self, model_name: str = "gemini-2.5-flash"):
        """Initialize Gemini client for video comparison using new SDK (uses standard flash, not lite)"""
        if not GEMINI_AVAILABLE:
            raise RuntimeError("google-genai package not installed. Install with: pip install google-genai")

        try:
            # Initialize the new Gemini Client API
            self.gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            self.gemini_model_name = model_name
            print(f"Gemini Client initialized successfully with model: {model_name}")
        except Exception as e:
            print(f"❌ Error initializing Gemini Client: {e}")
            print("Please ensure your GEMINI_API_KEY environment variable is set correctly.")
            self.gemini_client = None
            self.use_gemini = False

    def _initialize_qwen_vl(self):
        """Initialize Qwen3-VL-30B for video understanding (FREE alternative to Gemini)"""
        try:
            from transformers import AutoProcessor

            print(f"\n🎥 Loading FREE Qwen3-VL: {self.qwen_vl_model}")
            print(f"   Device: {self.device}")
            print(f"   This may take a few minutes on first run...")

            # Load processor
            self.qwen_vl_processor = AutoProcessor.from_pretrained(
                self.qwen_vl_model,
                trust_remote_code=True
            )

            # Determine which model class to use based on model name
            model_class = None
            if "30B" in self.qwen_vl_model or "A3B" in self.qwen_vl_model:
                # Qwen3-VL-30B uses MoE architecture
                try:
                    from transformers import Qwen3VLMoeForConditionalGeneration
                    model_class = Qwen3VLMoeForConditionalGeneration
                    print(f"   Using Qwen3VLMoeForConditionalGeneration (MoE architecture)")
                except ImportError:
                    print(f"   ❌ Qwen3VLMoeForConditionalGeneration not found in transformers!")
                    print(f"   Your transformers version is too old for Qwen3-VL-30B.")
                    print(f"   Please upgrade transformers:")
                    print(f"      pip install --upgrade transformers>=4.50.0")
                    print(f"   Or install from source:")
                    print(f"      pip install git+https://github.com/huggingface/transformers")
                    raise ImportError("Qwen3VLMoeForConditionalGeneration requires transformers>=4.50.0")
            else:
                # Standard Qwen2-VL models (2B, 7B, etc.)
                from transformers import Qwen2VLForConditionalGeneration
                model_class = Qwen2VLForConditionalGeneration
                print(f"   Using Qwen2VLForConditionalGeneration")

            # Load model with flash attention 2 if available
            try:
                self.qwen_vl = model_class.from_pretrained(
                    self.qwen_vl_model,
                    dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                    trust_remote_code=True
                )
                print(f"   ✓ Using Flash Attention 2 for faster inference")
            except Exception as flash_error:
                print(f"   ⚠️  Flash Attention 2 not available, falling back to standard attention")
                print(f"      Error: {flash_error}")
                self.qwen_vl = model_class.from_pretrained(
                    self.qwen_vl_model,
                    dtype="auto",
                    device_map="auto",
                    trust_remote_code=True
                )

            print(f"✓ Qwen3-VL initialized successfully: {self.qwen_vl_model}")
            print(f"   Will be used for Stage 1 (filtering) & Stage 2 (forensics)")

        except Exception as e:
            print(f"❌ Error loading Qwen3-VL: {e}")
            print(f"   Make sure the model is available: {self.qwen_vl_model}")
            print(f"   For Qwen3-VL-30B, you need transformers>=4.50.0:")
            print(f"     pip install --upgrade 'transformers>=4.50.0' qwen-vl-utils accelerate")
            print(f"   Optional for better performance:")
            print(f"     pip install flash-attn")
            self.qwen_vl = None
            self.qwen_vl_processor = None
            self.use_qwen_vl = False

    def _initialize_free_llm(self):
        """Initialize free open-source LLM for text generation (Qwen)"""
        try:
            print(f"\n🤖 Loading FREE LLM: {self.free_llm_model}")
            print(f"   Device: {self.device}")
            print(f"   VRAM: ~20GB with 4-bit quantization (fits on single RTX 4090)")

            # Load tokenizer
            self.free_llm_tokenizer = AutoTokenizer.from_pretrained(
                self.free_llm_model,
                trust_remote_code=True
            )

            # Configure 4-bit quantization for efficient memory usage
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            # Load model with 4-bit quantization
            self.free_llm = AutoModelForCausalLM.from_pretrained(
                self.free_llm_model,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )

            print(f"✓ Free LLM initialized successfully: {self.free_llm_model}")

        except Exception as e:
            print(f"❌ Error loading free LLM: {e}")
            print(f"   Make sure the model is available: {self.free_llm_model}")
            print(f"   Install required packages:")
            print(f"     pip install transformers>=4.37.0 bitsandbytes accelerate")
            self.free_llm = None
            self.free_llm_tokenizer = None
            self.use_free_llm_for_checklist = False

    def _initialize_stage1_models(self):
        """
        Initialize Stage 1 models (Qwen2.5-VL-7B) on multiple GPUs.
        Loads models ONCE during scorer initialization for reuse across all prompts.
        """
        from pipeline.stage1_filter import load_stage1_model_on_gpu

        num_gpus = torch.cuda.device_count()
        print(f"\n📦 Stage 1 Multi-GPU Setup:")
        print(f"   Available GPUs: {num_gpus}")
        print(f"   Model: {self.stage1_model_name}")
        print(f"   Loading models (one-time setup)...")

        # Load model on each GPU
        for gpu_id in range(num_gpus):
            model, processor, device = load_stage1_model_on_gpu(gpu_id, self.stage1_model_name)
            self.stage1_models[gpu_id] = (model, processor, device)

        print(f"   ✓ Stage 1 models loaded on {num_gpus} GPUs and ready for reuse")

    def run_stage1_filter(self, video_paths: List[str], text_prompt: str, checklist: str,
                         num_gpus: int = None, workers_per_gpu: int = 5,
                         max_iterations: int = 5, run_stage0: bool = False,
                         num_frames: int = 30) -> List[str]:
        """
        Run Stage 1 hallucination filter (Phase 1a + 1b) with multi-GPU support.

        Phase 1a: Static Element Check
        - Sample 10 random frames per video
        - Check each frame against checklist (batch inference)
        - STRICT: Any frame fails → reject video

        Phase 1b: Temporal Consistency Check (Prompt-Aware)
        - Sample 12 frames: 4 from beginning + 4 from middle + 4 from end
        - Captures full temporal progression: initial state → transition → final state
        - Check if temporal changes MATCH what the prompt describes
        - Only flag changes that contradict or aren't mentioned in prompt
        - Reject if actual hallucinations detected

        Convergence: Repeat until no passed videos change to failed.

        Args:
            video_paths: List of candidate video paths
            text_prompt: Original generation prompt (e.g., "An egg breaks on the floor")
            checklist: Prompt alignment checklist (from generate_prompt_alignment_checklist)
            num_gpus: Number of GPUs (None = auto-detect)
            workers_per_gpu: Worker threads per GPU (default: 5)
            max_iterations: Maximum convergence iterations (default: 5)
            run_stage0: Run Stage 0 first frame validation (default: False, skip it)
            num_frames: Number of uniformly sampled frames for Stage 1 (default: 30)

        Returns:
            List of video paths that passed Stage 1 filtering

        Example:
            # Generate checklist first
            checklist = scorer.generate_prompt_alignment_checklist("a red ball bouncing")

            # Run Stage 1 filter
            survivors = scorer.run_stage1_filter(
                video_paths=["video1.mp4", "video2.mp4", ...],
                text_prompt="a red ball bouncing",
                checklist=checklist,
                num_gpus=2,
                workers_per_gpu=5
            )
        """
        from pipeline.stage1_filter import run_stage1_until_convergence

        if not self.use_stage1_model:
            raise RuntimeError("Stage 1 model not enabled. Set use_stage1_model=True in constructor.")

        if num_gpus is None:
            num_gpus = torch.cuda.device_count()

        # Pass pre-loaded models (loaded once in __init__)
        return run_stage1_until_convergence(
            video_paths=video_paths,
            text_prompt=text_prompt,  # 🆕 Pass text_prompt for prompt-aware Stage 1b
            checklist=checklist,
            num_gpus=num_gpus,
            workers_per_gpu=workers_per_gpu,
            max_iterations=max_iterations,
            model_name=self.stage1_model_name,
            gpu_models=self.stage1_models,  # Pre-loaded models from __init__
            run_stage0=run_stage0,  # Pass run_stage0 parameter
            num_frames=num_frames  # Pass num_frames parameter
        )

    def _call_qwen_vl(self, video_path: str, prompt_text: str, max_new_tokens: int = 1500,
                      fps: float = 1.0, max_pixels: int = 480 * 720) -> str:
        """
        Helper method to call Qwen3-VL with a video and text prompt.
        Uses official Qwen3-VL approach with process_vision_info for memory efficiency.

        Args:
            video_path: Path to video file
            prompt_text: Text prompt/question
            max_new_tokens: Max tokens to generate
            fps: Frames per second to sample (default: 1.0 for memory efficiency)
            max_pixels: Maximum pixels per frame (default: 360*420 = 151,200 pixels)
                       Lower values = less memory. Examples:
                       - 360*420 = 151K pixels (low res, ~40GB VRAM)
                       - 720*480 = 345K pixels (medium, ~70GB VRAM)
                       - 1280*720 = 922K pixels (high, 100GB+ VRAM)

        Returns:
            Generated text response
        """
        if self.qwen_vl is None or self.qwen_vl_processor is None:
            raise RuntimeError("Qwen3-VL not initialized. Set use_qwen_vl=True in constructor.")

        # Build messages in Qwen3-VL format with fps and max_pixels control
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": max_pixels,  # Control resolution
                        "fps": fps                  # Control frame sampling rate
                    },
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        # Use official process_vision_info helper
        from qwen_vl_utils import process_vision_info

        # Get text template
        text = self.qwen_vl_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Extract video inputs with kwargs
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True
        )

        # Prepare inputs using official approach
        inputs = self.qwen_vl_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        ).to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.qwen_vl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=True,
                top_p=0.95
            )

        # Decode (skip input tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.qwen_vl_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()

    def _evaluate_video_with_qwen_vl_stage1(self, video_path: str, text_prompt: str,
                                            prompt_alignment_checklist: str) -> dict:
        """
        Stage 1: Evaluate video with Qwen3-VL (FREE alternative to Gemini).

        Args:
            video_path: Path to video file
            text_prompt: Generation prompt
            prompt_alignment_checklist: Pre-generated checklist

        Returns:
            Dict with 'passed': bool, 'path': str, 'candidate_num': int, 'response': str
        """
        import re

        # Extract candidate number from filename
        filename = os.path.basename(video_path)
        if 'candidate_' in filename:
            candidate_match = filename.split('candidate_')[1].split('_')[0]
        elif 'particle' in filename:
            candidate_match = filename.split('particle')[1].split('_')[0].split('.')[0]
        else:
            candidate_match = '0'
        candidate_num = int(candidate_match)

        try:
            # Build Stage 1 prompt
            stage1_prompt = f"""
# ROLE: VISUAL HALLUCINATION HUNTER
**GOAL:** You are a "Glitch Hunter." Your job is to find visual errors that break the laws of physics or identity.
**METHOD:** Continuous Frame-by-Frame Scan.
**STRICTNESS:** Zero Tolerance. If you see a glitch for 0.1 seconds, it is a FAIL.

---
## STEP 1: LOAD THE BLUEPRINT
{prompt_alignment_checklist}

*Verify the video strictly against the [VISUAL_SPECS] above.*

---
## STEP 2: THE "GLITCH SCAN" (Specific Visual Anchors)
Watch the entire video (0:00 to End). Look for these **3 Specific Failure Modes**.

### 1. THE "GHOST" SCAN (Cardinality & Negative Space)
**Task:** Scan the **EMPTY SPACE** around the main subject.
* **The Glitch (Multiplicity):** Does a second object flicker into existence?
* **The Glitch (Vanishing):** Does the subject disappear without leaving the frame?
* **FAIL TRIGGER:** If the visible count of [Subject_Name] is ever NOT [Expected_Count].

### 2. THE "MUTANT" SCAN (Topology & Integrity)
**Task:** Watch the **SILHOUETTE** of the main subject.
* **The Glitch (Stacking/Fusion):** Does the subject look like two objects stuck together?
* **The Glitch (Morphing):** Does the subject change geometry?
* **FAIL TRIGGER:** If the subject loses its defined [Subject_Shape] or splits/merges.

### 3. THE "MATRIX" SCAN (Background Stability)
**Task:** Watch the **FIXED FEATURES** (Floor tiles, Wall corners).
* **The Glitch:** Do textures slide or breathe?
* **FAIL TRIGGER:** If the floor moves independently of the camera.

---
## OUTPUT FORMAT (STRICT):

**GLITCH_LOG:**
* **Ghost_Log:** [e.g., "0:02 (2nd object flickered in background)", or "NONE"]
* **Mutant_Log:** [e.g., "0:04 (Subject developed 'snowman' stack)", or "NONE"]
* **Stability_Log:** [e.g., "0:03 (Floor texture sliding)", or "NONE"]

**VERDICT:**
[PASS / FAIL] - If ANY glitch is found (Ghost, Mutant, Matrix) → FAIL. Otherwise → PASS.

**REASONING:**
[1-2 sentences explaining the verdict]
"""

            # Call Qwen3-VL
            response = self._call_qwen_vl(video_path, stage1_prompt, max_new_tokens=800)

            # Parse PASS/FAIL from response
            passed = False
            if re.search(r'\bPASS\b', response, re.IGNORECASE):
                passed = True
            elif re.search(r'\bFAIL\b', response, re.IGNORECASE):
                passed = False
            else:
                # If no clear verdict, look for "NONE" in all glitch logs
                if "Ghost_Log:" in response and "Mutant_Log:" in response and "Stability_Log:" in response:
                    ghost_none = "NONE" in response.split("Ghost_Log:")[1].split("\n")[0]
                    mutant_none = "NONE" in response.split("Mutant_Log:")[1].split("\n")[0]
                    stability_none = "NONE" in response.split("Stability_Log:")[1].split("\n")[0]
                    passed = ghost_none and mutant_none and stability_none

            return {
                'passed': passed,
                'path': video_path,
                'candidate_num': candidate_num,
                'response': response,
                'filename': filename
            }

        except Exception as e:
            print(f"      ❌ Error evaluating candidate_{candidate_num} with Qwen3-VL: {e}")
            return {
                'passed': False,
                'path': video_path,
                'candidate_num': candidate_num,
                'response': f"Error: {e}",
                'filename': filename
            }

    def _evaluate_video_with_qwen_vl_stage2(self, video_path: str, text_prompt: str) -> dict:
        """
        Stage 2: Generate forensic report with Qwen3-VL (FREE alternative to Gemini).

        Args:
            video_path: Path to video file
            text_prompt: Generation prompt

        Returns:
            Dict with 'candidate_num', 'forensic_log', 'path'
        """
        import re

        # Extract candidate number
        filename = os.path.basename(video_path)
        if 'candidate_' in filename:
            candidate_match = filename.split('candidate_')[1].split('_')[0]
        elif 'particle' in filename:
            candidate_match = filename.split('particle')[1].split('_')[0].split('.')[0]
        else:
            candidate_match = '0'
        candidate_num = int(candidate_match)

        try:
            # Build Stage 2 forensic prompt (SCRIBE format)
            stage2_prompt = f"""
# ROLE: PHYSICS FORENSIC ANALYST (SCRIBE)
**MISSION:** Generate a detailed Frame-by-Frame Physics Observation Report.

**THE PROMPT (Ground Truth):**
"{text_prompt}"

---

## YOUR TASK: 3-PART FORENSIC REPORT

### PART 1: VISUAL NARRATION (What You See)
Describe the video's visual content chronologically. Focus on:
- **Objects:** What objects are present? Describe their appearance.
- **Motion:** How do objects move? (e.g., "ball rolls", "ball bounces", "ball comes to rest")
- **Interactions:** What physical events occur? (e.g., "ball collides with wall")
- **Environment:** Describe the scene/background

Be factual and precise. This is pure visual observation, no physics analysis yet.

---

### PART 2: PHYSICS OBSERVATION LOG
Now analyze the physical behavior you observed. For each major physical event:

**Event: [Brief description]**
- **Physics Category:** [Gravity / Collision / Friction / Elasticity / Energy / Other]
- **Observation:** [Describe what physically happens in 1-2 sentences]
- **Realism Assessment:** [Is this behavior physically plausible? Why?]

List 3-5 major physical events chronologically.

---

### PART 3: VISUAL ANOMALY SCAN
Scan for these specific visual glitches:

1. **Object Conservation:** Do objects appear/disappear without reason?
   Status: [None observed / Observed: description]

2. **Jitter/Teleportation:** Does the object jump positions between frames?
   Status: [None observed / Observed: description]

3. **Boundary Violations:** Does the object clip through walls/floor?
   Status: [None observed / Observed: description]

4. **Friction Anomalies:** Does the object slide without stopping on a flat surface?
   Status: [None observed / Observed: description]

5. **Background Stability:** Do fixed features (walls, floor) morph or move?
   Status: [None observed / Observed: description]

---

## FINAL SUMMARY
**Overall Physics Quality:** [EXCELLENT / GOOD / FAIR / POOR]
**Key Strengths:** [1-2 sentences]
**Key Issues:** [1-2 sentences, or "None observed"]
"""

            # Call Qwen3-VL
            forensic_log = self._call_qwen_vl(video_path, stage2_prompt, max_new_tokens=2000)

            return {
                'candidate_num': candidate_num,
                'forensic_log': forensic_log,
                'path': video_path
            }

        except Exception as e:
            print(f"      ❌ Error generating forensics for candidate_{candidate_num} with Qwen3-VL: {e}")
            return {
                'candidate_num': candidate_num,
                'forensic_log': f"Error generating forensic report: {e}",
                'path': video_path
            }

    def free_stage1_models(self):
        """
        Free the pre-loaded Qwen3-VL-8B stage1 models from VRAM on all GPUs.
        Call this before the tournament model loads to reclaim ~18GB per GPU.
        """
        if self.stage1_models:
            print(f"   🧹 Freeing {len(self.stage1_models)} stage1 VLM instance(s) from VRAM...")
            import gc
            for gpu_id, (m, p, d) in self.stage1_models.items():
                del m
                del p
            self.stage1_models.clear()
            gc.collect()
            torch.cuda.empty_cache()
            print(f"   ✓ Stage1 VLM models freed")

    def free_text_llm(self):
        """
        Free the 32B checklist LLM from VRAM.
        Call this after generating all checklists for a prompt, before running Stage 1,
        to reclaim ~20GB of VRAM for the tournament 14B model and Qwen2.5-VL instances.
        The model will be lazy-reloaded next time _generate_checklist_with_free_llm is called.
        """
        if self.free_llm is not None:
            print(f"   🧹 Freeing text LLM ({self.free_llm_model}) from VRAM...")
            del self.free_llm
            self.free_llm = None
            del self.free_llm_tokenizer
            self.free_llm_tokenizer = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print(f"   ✓ Text LLM freed")

    def _generate_checklist_with_free_llm(self, alignment_prompt: str) -> str:
        """
        Generate prompt alignment checklist using free open-source LLM (Qwen).
        Works on remote compute clusters without requiring Ollama server.
        Lazy-reloads the model if it was previously freed via free_text_llm().

        Args:
            alignment_prompt: The prompt for checklist generation

        Returns:
            String containing the prompt alignment checklist
        """
        # Lazy-reload if freed between prompts
        if self.free_llm is None or self.free_llm_tokenizer is None:
            print(f"   🔄 Reloading text LLM ({self.free_llm_model}) for checklist generation...")
            self._initialize_free_llm()
        if self.free_llm is None or self.free_llm_tokenizer is None:
            raise RuntimeError("Free LLM not initialized. Check initialization logs for errors.")

        print(f"\n🤖 Using Free LLM ({self.free_llm_model}) for checklist generation")

        try:
            gen_start = time.time()

            # Prepare messages in chat format
            messages = [
                {"role": "system", "content": "You are a precise visual requirement extractor for video generation evaluation."},
                {"role": "user", "content": alignment_prompt}
            ]

            # Format with chat template
            text = self.free_llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            model_inputs = self.free_llm_tokenizer([text], return_tensors="pt").to(self.device)

            # Generate (increased tokens for large JSON output)
            with torch.no_grad():
                generated_ids = self.free_llm.generate(
                    **model_inputs,
                    max_new_tokens=4500,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.95
                )

            # Decode (skip prompt tokens)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            raw_response = self.free_llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            gen_time = time.time() - gen_start

            # Extract JSON from response (handle markdown code blocks and extra text)
            import json
            import re

            spec_json_str = raw_response

            # Try to extract JSON from markdown code blocks first
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', raw_response, re.DOTALL)
            if json_match:
                spec_json_str = json_match.group(1)
                print(f"  📝 Extracted JSON from markdown code block")
            else:
                # Try to find JSON object boundaries (first { to last })
                first_brace = raw_response.find('{')
                last_brace = raw_response.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    spec_json_str = raw_response[first_brace:last_brace+1]
                    if spec_json_str != raw_response:
                        print(f"  📝 Extracted JSON object from response (removed extra text)")

            # Try to parse and validate JSON
            try:
                spec_dict = json.loads(spec_json_str)

                # Check for expected top-level keys
                has_meta = 'meta' in spec_dict
                has_entities = 'entities' in spec_dict
                has_action_phases = 'action_phases' in spec_dict
                has_state_changes = 'state_changes' in spec_dict
                has_validation_constraints = 'validation_constraints' in spec_dict

                # Pretty-print the JSON
                formatted_json = json.dumps(spec_dict, indent=2)

                print(f"✓ Structured Video Specification Generated (Free LLM):")
                print(f"  ⏱️  Generation time: {gen_time:.1f}s")
                print(f"  🤖 Model: {self.free_llm_model}")
                print(f"  ✓ meta: {'Present' if has_meta else 'Missing'}")
                print(f"  ✓ entities: {'Present' if has_entities else 'Missing'}")
                print(f"  ✓ action_phases: {'Present' if has_action_phases else 'Missing'}")
                print(f"  ✓ state_changes: {'Present' if has_state_changes else 'Missing'}")
                print(f"  ✓ validation_constraints: {'Present' if has_validation_constraints else 'Missing'}")

                # Print summary
                if has_entities:
                    entity_count = len([k for k in spec_dict['entities'].keys()
                                       if k not in ['secondary_objects']])
                    print(f"  📦 Entities detected: {entity_count}")

                if has_state_changes:
                    expected_changes = len(spec_dict['state_changes'].get('expected_changes', []))
                    invariant_props = len(spec_dict['state_changes'].get('invariant_properties', []))
                    print(f"  🔄 Expected changes: {expected_changes}")
                    print(f"  🔒 Invariant properties: {invariant_props}")

                print(f"\n{'═' * 80}")
                print(f"📋 FULL JSON SPECIFICATION:")
                print(f"{'═' * 80}")
                print(formatted_json)
                print(f"{'═' * 80}")
                print(f"  Total length: {len(spec_json_str)} characters")

                return spec_json_str  # Return original (not formatted) for consistency

            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Response is not valid JSON: {e}")
                print(f"  JSON parsing failed at position {e.pos}: {e.msg}")
                print(f"  Returning raw response (may need manual inspection)")
                print(f"\n{'─' * 80}")
                print("RAW RESPONSE (after extraction):")
                print(f"{'─' * 80}")
                spec_lines = spec_json_str.split('\n')
                for i, line in enumerate(spec_lines[:50], 1):  # Show first 50 lines
                    print(f"  {i:3d} | {line}")
                if len(spec_lines) > 50:
                    remaining = len(spec_lines) - 50
                    print(f"  ... ({remaining} more lines)")
                print(f"{'─' * 80}")
                print(f"  First 100 chars: {spec_json_str[:100]}")
                print(f"  Last 100 chars: {spec_json_str[-100:]}")
                print(f"{'─' * 80}")

                # Show the original response if different
                if spec_json_str != raw_response:
                    print(f"\n{'─' * 80}")
                    print("ORIGINAL RAW RESPONSE (before extraction):")
                    print(f"{'─' * 80}")
                    raw_lines = raw_response.split('\n')
                    for i, line in enumerate(raw_lines[:30], 1):
                        print(f"  {i:3d} | {line}")
                    if len(raw_lines) > 30:
                        remaining = len(raw_lines) - 30
                        print(f"  ... ({remaining} more lines)")
                    print(f"{'─' * 80}")

                return spec_json_str

        except Exception as e:
            print(f"❌ Error generating checklist with free LLM: {e}")
            raise

    def generate_prompt_alignment_checklist(self, text_prompt: str,
                                           model_name: str = "gemini-2.5-flash",
                                           max_output_tokens: int = 4500,
                                           temperature: float = 0.1) -> str:
        """
        Generate a structured JSON specification for video validation (Bootstrap Phase 0).

        This creates a comprehensive specification including:
        - Entities (agent/tool/target/background) with geometry, material, appearance
        - Action phases (initial → transition → final)
        - Expected changes vs invariant properties (critical for hallucination detection)
        - Spatial relationships and validation constraints

        Can use either:
        - Free open-source LLM (Qwen) - RECOMMENDED for remote clusters (100% free)
        - Gemini API (paid) - Default option

        Args:
            text_prompt: The text prompt describing what should be in the video
            model_name: Gemini model to use (ignored if using free LLM)
            max_output_tokens: Maximum tokens in response (default: 4500 for large JSON)
            temperature: Sampling temperature (ignored if using free LLM)

        Returns:
            String containing the JSON specification
        """
        print(f"\n📋 Generating Structured Video Specification for: '{text_prompt}'")

        alignment_prompt = f"""
You are a video scene specification parser. Your task is to analyze a text description of a video and extract a structured JSON specification that will be used for automated video validation.

=== CRITICAL RULES ===

1. **DO NOT HALLUCINATE ENTITIES** - ONLY include objects/entities that are EXPLICITLY MENTIONED in the text
   - ❌ BAD: "ball is dropped" → adding "hand" entity (NOT mentioned in text)
   - ✅ GOOD: "ball is dropped" → only "ball" as subject (mechanism of drop not specified, so no hand)
   - ✅ GOOD: "hand drops ball" → "hand" as interactive_entity (explicitly mentioned)
   - ❌ BAD: Inferring or assuming entities not in the text

2. **Extract ONLY what is explicitly stated in the text** - Do not infer mechanism or add details
3. **Mark ALL visual details as "UNKNOWN"** - colors, exact sizes, exact positions, textures
4. **Use conceptual terms for spatial relationships** - "above", "near", "on" (not pixel coordinates)
5. **Track ALL property changes including shape deformations** - elastic changes must be captured
6. **Output valid JSON only** - no additional commentary

=== INPUT ===

Video description: "{text_prompt}"

=== YOUR TASK ===

Generate a JSON specification with the following structure:

**1. META**
- Store the original prompt text
- Add timestamp and version

**2. ENTITIES**

ENTITY ROLE DEFINITIONS:
- **subject**: The PRIMARY object the video is about (the main actor/focus)
  Examples: "ball" in "ball bounces", "rocket" in "rocket launches", "egg" in "egg breaks", "apple" in "apple falls"

- **interactive_entity**: ALL objects/entities the subject physically interacts with (can be multiple, list all)
  - Include EVERYTHING the subject touches, contacts, or is affected by
  - Can be passive (floor, wall, branch) or active (hand applying force, fire/exhaust propelling)
  - Can be surfaces, targets, actors, mechanisms
  Examples:
    * "ball bounces on floor" → floor
    * "rocket launches from ground" → ground, fire/exhaust (propulsion mechanism)
    * "needle presses into ball" → ball (target being pressed), hand (actor applying force)
    * "apple falls from tree branch" → tree branch (separation point)
    * "egg breaks on floor" → floor (impact point)

- **background**: Static environment/setting that does NOT interact with subject
  - Non-interactive parts of the scene
  - Walls, sky, room, distant objects
  Examples: For "apple falls from tree branch" → REST of tree (trunk, other branches), sky, landscape

ASSIGNMENT RULES:
1. Identify the ONE main moving/changing object → **subject**
2. List ALL entities the subject interacts with (touches, contacts, is held by, is propelled by, collides with) → **interactive_entity**
   - This can be multiple entities - list them all!
   - Include: starting points ("from X"), landing points ("on Y"), contact points ("into Z"), actors ("held by hand"), mechanisms ("propelled by fire")
3. Everything else that's static and non-interactive → **background**

EDGE CASES:
- "Apple falls from tree branch"
  - subject: apple
  - interactive_entity: tree branch (separation point)
  - background: rest of tree, ground, sky

- "Hand presses needle into ball"
  - subject: needle (the primary action focus)
  - interactive_entity: ball (target being pressed), hand (actor applying force)
  - background: table, room, walls

- "Rocket launches from ground"
  - subject: rocket
  - interactive_entity: ground (departure point), fire/exhaust (propulsion mechanism)
  - background: sky, clouds, landscape

For each entity mentioned:
- name: What is it called?
- role: subject, interactive_entity, or background
- count: How many? (exact number if stated)
- geometry.type: Shape type (sphere, cylinder, articulated, etc.)
- geometry.characteristics: List descriptors (thin, sharp, round, etc.)
- geometry.typical_dimensions: Estimated size ranges if inferable (e.g., needle 30-50mm)
- geometry.exact_dimensions: "UNKNOWN"
- geometry.exact_bbox: "UNKNOWN"
- material.type: What it's made of (if stated or strongly implied)
- material.properties: Physical properties (rigid, elastic, soft, etc.)
- appearance.color: "UNKNOWN"
- appearance.color_name: "UNKNOWN"
- appearance.color_hex: "UNKNOWN"
- appearance.typical_colors: List possible colors (e.g., ["red", "blue", "yellow"])
- appearance.texture: "UNKNOWN"
- appearance.visual_signature: "UNKNOWN"
- initial_state.description: Conceptual starting state
- initial_state.exact_position: "UNKNOWN"
- properties: List relevant properties

**3. ACTION_PHASES**
Structure phases based on PHYSICAL INTERACTIONS following this mandatory 3-part structure:

🔑 **MANDATORY PHASE STRUCTURE:**

**Phase 0 - INITIAL STATE** (always required):
- Subject's state BEFORE any interaction with interactive entities
- Phase name: "initial"
- Description: Subject's configuration/position before interactions begin
- Examples: "ball held in hand", "rocket on launch pad", "needle positioned above balloon"

**Phase 1, 2, 3... - CONTACT POINTS** (one phase per interactive entity):
- Each phase represents subject's interaction with ONE interactive entity
- Phase name format: "contact_[entity_name]" or specific interaction type
- Description: What happens when subject interacts with this entity (including reaction forces)
- Examples:
  * "impact_floor" - ball contacts floor (includes bouncing reaction)
  * "penetration_balloon" - needle penetrates balloon surface
  * "separation_ground" - rocket leaves launch pad
  * "collision_wall" - object hits wall

**Final Phase - FINAL STATE** (always required):
- Subject's state AFTER all interactions complete
- Phase name: "final"
- Description: Subject's end configuration/position
- Examples: "ball at rest on floor", "rocket in sky", "balloon deflated"

**CRITICAL RULES:**

1. **Always have 3+ phases**: initial + (1 or more contact phases) + final
2. **One contact phase per interactive entity**: If ball interacts with floor and wall, create separate phases
3. **Contact phase can repeat**: Ball hitting floor multiple times is ONE phase (VLM will detect repetition)
4. **Include reaction forces**: "impact_floor" includes both contact AND bounce reaction

**EXAMPLE 1: Ball Bouncing**
```
Interactive entities: [floor]

Phase 0: "initial"
  - Description: "ball dropped from height, falling toward floor"

Phase 1: "impact_floor"
  - Description: "ball contacts floor and bounces (includes compression and rebound reaction)"

Phase 2: "final"
  - Description: "ball comes to rest on floor after bouncing stops"
```

**EXAMPLE 2: Rocket Launch**
```
Interactive entities: [launch_pad, exhaust_fire]

Phase 0: "initial"
  - Description: "rocket on launch pad, engines igniting"

Phase 1: "separation_pad"
  - Description: "rocket separates from launch pad and begins ascending"

Phase 2: "propulsion_exhaust"
  - Description: "rocket propelled upward by exhaust fire (continuous interaction)"

Phase 3: "final"
  - Description: "rocket ascending into sky at high altitude"
```

**EXAMPLE 3: Needle Popping Balloon**
```
Interactive entities: [balloon, hand]

Phase 0: "initial"
  - Description: "needle held in hand, approaching inflated balloon"

Phase 1: "contact_balloon"
  - Description: "needle penetrates balloon surface, balloon begins deflating"

Phase 2: "held_hand"
  - Description: "needle remains held in hand throughout (continuous contact)"

Phase 3: "final"
  - Description: "needle held in hand, balloon fully deflated"
```

**JSON FORMAT:**
```json
{{
  "total_phases": 3,
  "phases": [
    {{"phase_id": 0, "phase_name": "initial", "description": "..."}},
    {{"phase_id": 1, "phase_name": "contact_[entity]", "description": "..."}},
    {{"phase_id": 2, "phase_name": "final", "description": "..."}}
  ]
}}
```

**4. STATE_CHANGES**
CRITICAL: You MUST analyze ALL intrinsic properties and classify each as either expected_change or invariant_property.

🚨 **SPECIAL ATTENTION FOR CONTACT/COLLISION SCENARIOS:**
If the action involves CONTACT between subject and interactive_entity (bouncing, colliding, pressing, impacting):
- **MUST track geometry.shape changes** for elastic/deformable objects
- **MUST track state changes** if objects break/deform permanently
- At the moment of contact, check: Does the subject or interactive_entity change shape? Add to expected_changes!

INTRINSIC PROPERTIES TO ANALYZE (for each entity):
1. **count**: Number of instances (usually invariant - 1 ball stays 1 ball)
2. **geometry.type**: Base shape (sphere, cylinder, etc.) - usually invariant
3. **geometry.shape**: Actual shape - may deform temporarily (sphere → compressed → sphere)
   ⚠️ **CRITICAL FOR BOUNCING/COLLISION**: Track elastic deformation at contact point!
4. **geometry.size**: Overall size/dimensions - usually invariant (small tolerance for compression)
5. **material.type**: Material composition - ALWAYS invariant (rubber stays rubber)
6. **material.properties**: Physical properties (elastic, rigid) - usually invariant
7. **appearance.color**: Color - ALWAYS invariant (red stays red)
8. **appearance.texture**: Surface texture - usually invariant
9. **position**: Location in space - often changes (ball moves)
10. **velocity**: Motion state - often changes (static → moving)
11. **orientation**: Rotation angle - may change
12. **state**: Physical state (whole → broken, inflated → deflated) - depends on action

expected_changes: For properties that WILL change during the action:
- entity: Which entity
- property: What property (position, shape, velocity, state, etc.)
- initial_value: Starting value (can be conceptual)
- final_value: Ending value (can be "UNKNOWN" if not stated)
- can_change: true
- transition_type: gradual/sudden/continuous/elastic (elastic = temporary deformation that recovers)
- estimated_range: If applicable (e.g., "2-10" mm)
- note: Optional explanation (e.g., "ball compresses at contact then returns to sphere")

invariant_properties: For properties that MUST NOT change during the action:
- entity: Which entity
- property: What property (count, color, material, base_geometry_type, etc.)
- value: Expected value (can be "UNKNOWN" if visual)
- must_not_change: true
- tolerance: Numeric tolerance if applicable (e.g., 0.05 for 5% color variance, 0.1 for 10% size variance during compression)

CRITICAL EXAMPLES:
**Breaking egg scenario:**
- state: changes (whole → broken)
- geometry.shape: changes permanently
- count/color/material: invariant

**Rocket launching:**
- position/velocity: change
- fire/exhaust: appears and intensifies
- rocket geometry.shape/color/material: invariant

**5. SPATIAL_CONFIGURATION**

conceptual_layout.relationships: Spatial relationships you can infer
- entity1, entity2: The two related entities
- relation: above_or_approaching, holding, in_contact_with, penetrating, resting_on, etc.
- valid_phases: Which phases this relationship holds
- confidence: high/medium/low
- exact_distance: "UNKNOWN"
- exact_distance_px: "UNKNOWN"

conceptual_layout.depth_ordering: Which objects are in front/back
- front, back: Entity names
- confidence: high/medium/low
- exact_z_values: "UNKNOWN"

exact_layout: ALL fields "UNKNOWN"

**6. VALIDATION_CONSTRAINTS**

count_constraints: Expected count for each entity
- min, max, exact: Numbers

geometry_constraints: Shape requirements
- must_be: Description
- aspect_ratio_min: If applicable
- must_have: Required features
- allowed_shapes: List of acceptable shapes

material_constraints: Material requirements
- must_be: rigid/deformable/etc.
- allowed_materials: List

forbidden_changes: What transformations are NOT allowed
- morphing: Entity cannot become other objects
- material_changes: Properties that cannot change
- appearance_changes: Visual properties that cannot change

required_persistence: What must stay constant
- entity: Entity name
- properties: List of properties
- must_persist: true
- exact_values: "UNKNOWN"
- tolerance: Numeric tolerance (e.g., 0.05 for 5%)

**7. FIELDS_TO_COMPLETE_FROM_FIRST_FRAME**
List all the "UNKNOWN" fields that will be filled in later

**8. GROUNDING_INSTRUCTIONS**
- first_frame_validation_checks: List checks to perform
- information_to_extract: List information to extract
- constraints_to_generate: List constraints to generate

=== EXAMPLES ===

**Example 1:** "A rocket launches into the sky"

Expected Output (abbreviated):
{{
  "meta": {{
    "prompt_text": "A rocket launches into the sky",
    "parse_timestamp": "2026-03-22T10:30:00Z",
    "parser_version": "1.0"
  }},

  "entities": {{
    "subject": {{
      "name": "rocket",
      "role": "subject",
      "count": 1,
      "geometry": {{"type": "cylinder", "characteristics": ["elongated", "pointed_top"]}},
      "material": {{"type": "metal", "properties": ["rigid", "solid"]}},
      "appearance": {{"color": "UNKNOWN", "typical_colors": ["white", "silver", "multi-color"]}},
      "initial_state": {{"description": "on ground/launch_pad, vertical orientation", "exact_position": "UNKNOWN"}}
    }},
    "interactive_entities": [
      {{
        "name": "ground/launch_pad",
        "role": "interactive_entity",
        "count": 1,
        "geometry": {{"type": "plane_surface"}},
        "material": {{"type": "concrete_or_metal", "properties": ["rigid", "solid"]}},
        "appearance": {{"color": "UNKNOWN"}},
        "initial_state": {{"description": "stationary platform", "exact_position": "UNKNOWN"}},
        "interaction_type": "departure_point"
      }},
      {{
        "name": "fire/exhaust",
        "role": "interactive_entity",
        "count": 1,
        "geometry": {{"type": "flame_plume"}},
        "material": {{"type": "gas/fire", "properties": ["dynamic", "high_energy"]}},
        "appearance": {{"color": "UNKNOWN", "typical_colors": ["orange", "red", "yellow", "bright"]}},
        "initial_state": {{"description": "emitting from rocket base", "exact_position": "UNKNOWN"}},
        "interaction_type": "propulsion_mechanism"
      }}
    ],
    "background": {{
      "name": "UNKNOWN",
      "role": "background",
      "typical_types": ["sky", "clouds", "landscape", "outdoor_environment"]
    }}
  }},

  "action_phases": {{
    "total_phases": 4,
    "phases": [
      {{"phase_id": 0, "phase_name": "initial", "description": "rocket on launch pad, engines igniting"}},
      {{"phase_id": 1, "phase_name": "separation_pad", "description": "rocket separates from launch pad and begins ascending"}},
      {{"phase_id": 2, "phase_name": "propulsion_exhaust", "description": "rocket propelled upward by exhaust fire"}},
      {{"phase_id": 3, "phase_name": "final", "description": "rocket ascending into sky at altitude"}}
    ]
  }},

  "state_changes": {{
    "expected_changes": [
      {{"entity": "rocket", "property": "position", "initial_value": "on_ground", "final_value": "in_sky", "can_change": true, "transition_type": "continuous"}},
      {{"entity": "rocket", "property": "velocity", "initial_value": "0", "final_value": "upward_motion", "can_change": true, "transition_type": "gradual"}},
      {{"entity": "fire/exhaust", "property": "intensity", "initial_value": "igniting", "final_value": "full_thrust", "can_change": true, "transition_type": "gradual"}},
      {{"entity": "fire/exhaust", "property": "visibility", "initial_value": "small_flame", "final_value": "large_plume", "can_change": true}}
    ],
    "invariant_properties": [
      {{"entity": "rocket", "property": "count", "value": 1, "must_not_change": true, "tolerance": 0}},
      {{"entity": "rocket", "property": "color", "value": "UNKNOWN", "must_not_change": true, "tolerance": 0.05}},
      {{"entity": "rocket", "property": "geometry.type", "value": "cylindrical", "must_not_change": true}},
      {{"entity": "rocket", "property": "geometry.size", "value": "UNKNOWN", "must_not_change": true, "tolerance": 0.02}},
      {{"entity": "rocket", "property": "material.type", "value": "metal", "must_not_change": true}},
      {{"entity": "rocket", "property": "material.properties", "value": ["rigid", "solid"], "must_not_change": true}},
      {{"entity": "ground/launch_pad", "property": "position", "value": "stationary", "must_not_change": true}},
      {{"entity": "ground/launch_pad", "property": "count", "value": 1, "must_not_change": true}}
    ]
  }},

  "validation_constraints": {{
    "count_constraints": {{
      "rocket": {{"min": 1, "max": 1, "exact": 1}},
      "ground/launch_pad": {{"min": 1, "max": 1, "exact": 1}},
      "fire/exhaust": {{"min": 1, "max": 1, "exact": 1}}
    }},
    "forbidden_changes": {{
      "morphing": [
        {{"entity": "rocket", "cannot_become": ["airplane", "bird", "other_object"]}}
      ],
      "appearance_changes": [
        {{"entity": "rocket", "property": "color", "cannot_change": true}}
      ]
    }}
  }}
}}

=== OUTPUT FORMAT ===

Output ONLY valid JSON following this exact structure. Do not include any text before or after the JSON.
"""

        # Route to free open-source LLM (Qwen) if configured - BEST for remote clusters
        if self.use_free_llm_for_checklist:
            return self._generate_checklist_with_free_llm(alignment_prompt)

        # Otherwise use Gemini API (requires payment)
        if not self.use_gemini or self.gemini_client is None:
            raise RuntimeError("Gemini is not initialized. Set use_gemini=True in constructor, or "
                             "use_free_llm_for_checklist=True for FREE Qwen (no API costs).")

        from google.genai import types

        max_retries = 3
        retry_delay = 60

        for attempt in range(max_retries):
            try:
                api_start = time.time()
                response = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents=alignment_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                        thinking_config=types.ThinkingConfig(include_thoughts=False)
                    )
                )
                raw_response = response.text
                api_time = time.time() - api_start

                # Extract JSON from response (handle markdown code blocks and extra text)
                import json
                import re

                spec_json_str = raw_response

                # Try to extract JSON from markdown code blocks first
                json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', raw_response, re.DOTALL)
                if json_match:
                    spec_json_str = json_match.group(1)
                    print(f"  📝 Extracted JSON from markdown code block")
                else:
                    # Try to find JSON object boundaries (first { to last })
                    first_brace = raw_response.find('{')
                    last_brace = raw_response.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        spec_json_str = raw_response[first_brace:last_brace+1]
                        if spec_json_str != raw_response:
                            print(f"  📝 Extracted JSON object from response (removed extra text)")

                # Try to parse and validate JSON
                try:
                    spec_dict = json.loads(spec_json_str)

                    # Check for expected top-level keys
                    has_meta = 'meta' in spec_dict
                    has_entities = 'entities' in spec_dict
                    has_action_phases = 'action_phases' in spec_dict
                    has_state_changes = 'state_changes' in spec_dict
                    has_validation_constraints = 'validation_constraints' in spec_dict

                    # Pretty-print the JSON
                    formatted_json = json.dumps(spec_dict, indent=2)

                    print(f"✓ Structured Video Specification Generated:")
                    print(f"  ⏱️  API call time: {api_time:.1f}s")
                    print(f"  ✓ meta: {'Present' if has_meta else 'Missing'}")
                    print(f"  ✓ entities: {'Present' if has_entities else 'Missing'}")
                    print(f"  ✓ action_phases: {'Present' if has_action_phases else 'Missing'}")
                    print(f"  ✓ state_changes: {'Present' if has_state_changes else 'Missing'}")
                    print(f"  ✓ validation_constraints: {'Present' if has_validation_constraints else 'Missing'}")

                    # Print summary
                    if has_entities:
                        entity_count = len([k for k in spec_dict['entities'].keys()
                                           if k not in ['secondary_objects']])
                        print(f"  📦 Entities detected: {entity_count}")

                    if has_state_changes:
                        expected_changes = len(spec_dict['state_changes'].get('expected_changes', []))
                        invariant_props = len(spec_dict['state_changes'].get('invariant_properties', []))
                        print(f"  🔄 Expected changes: {expected_changes}")
                        print(f"  🔒 Invariant properties: {invariant_props}")

                    print(f"\n{'═' * 80}")
                    print(f"📋 FULL JSON SPECIFICATION:")
                    print(f"{'═' * 80}")
                    print(formatted_json)
                    print(f"{'═' * 80}")
                    print(f"  Total length: {len(spec_json_str)} characters")

                    return spec_json_str  # Return original (not formatted) for consistency

                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Response is not valid JSON: {e}")
                    print(f"  JSON parsing failed at position {e.pos}: {e.msg}")
                    print(f"  Returning raw response (may need manual inspection)")
                    print(f"\n{'─' * 80}")
                    print("RAW RESPONSE (after extraction):")
                    print(f"{'─' * 80}")
                    spec_lines = spec_json_str.split('\n')
                    for i, line in enumerate(spec_lines[:50], 1):  # Show first 50 lines
                        print(f"  {i:3d} | {line}")
                    if len(spec_lines) > 50:
                        remaining = len(spec_lines) - 50
                        print(f"  ... ({remaining} more lines)")
                    print(f"{'─' * 80}")
                    print(f"  First 100 chars: {spec_json_str[:100]}")
                    print(f"  Last 100 chars: {spec_json_str[-100:]}")
                    print(f"{'─' * 80}")

                    # Show the original response if different
                    if spec_json_str != raw_response:
                        print(f"\n{'─' * 80}")
                        print("ORIGINAL RAW RESPONSE (before extraction):")
                        print(f"{'─' * 80}")
                        raw_lines = raw_response.split('\n')
                        for i, line in enumerate(raw_lines[:30], 1):
                            print(f"  {i:3d} | {line}")
                        if len(raw_lines) > 30:
                            remaining = len(raw_lines) - 30
                            print(f"  ... ({remaining} more lines)")
                        print(f"{'─' * 80}")

                    return spec_json_str

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "503" in error_msg or "quota" in error_msg.lower():
                    if attempt < max_retries - 1:
                        print(f"⚠️  API error (attempt {attempt + 1}/{max_retries}): {error_msg[:100]}... Waiting {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        print(f"❌ Failed after {max_retries} retries: {error_msg}")
                        raise
                else:
                    print(f"❌ Error generating prompt alignment checklist: {e}")
                    raise

        raise RuntimeError("Failed to generate prompt alignment checklist after all retries")

    def generate_physics_ground_truth(self, text_prompt: str,
                                      model_name: str = "gemini-2.5-flash",
                                      max_output_tokens: int = 2500,
                                      temperature: float = 0.1) -> str:
        """
        Generate physics ground truth rules for a given prompt using Gemini.

        This function creates a 3-point physics checklist that defines expected
        behavior for the scenario described in the prompt. These rules are used
        to audit AI-generated videos in Stage 2.

        Args:
            text_prompt: The text prompt describing the physics scenario
            model_name: Gemini model to use (default: gemini-2.5-flash-lite)
            max_output_tokens: Maximum tokens in response (default: 600)
            temperature: Sampling temperature (default: 0.1 for consistency)

        Returns:
            String containing the 3-point physics checklist
        """
        if not self.use_gemini or self.gemini_client is None:
            raise RuntimeError("Gemini is not initialized. Set use_gemini=True in constructor.")

        print(f"\n🔬 Generating Physics Ground Truth for prompt: '{text_prompt}'")

        # Create the prompt for generating physics rules
        physics_prompt = f"""
ROLE: Senior Physics Auditor

INPUT PROMPT: "{text_prompt}"

TASK: Create a 3-Point Physics Checklist that defines "Ground Truth" behavior for this specific scenario. These rules will be used to audit AI-generated videos.

REQUIRED CATEGORIES:

1. Kinetic Expectation: Define the specific acceleration or velocity behavior (e.g., "Gravity should cause exponential acceleration until impact").

2. Interaction Fidelity: Define exactly what must happen at the moment of contact/climax (e.g., "The water must displace outward in a radial splash proportional to the object size").

3. Energy/Mass Conservation: Define how the motion should settle or decay (e.g., "Each subsequent bounce must lose at least 30% height; no energy gain").

STRICT FORMAT:

Rule 1: [Specific behavior]
Rule 2: [Specific behavior]
Rule 3: [Specific behavior]

Avoid generic advice like "look realistic." Be scientifically descriptive.
"""

        # Retry logic for rate limiting
        max_retries = 3
        retry_delay = 60

        for attempt in range(max_retries):
            try:
                # Use stateless generate_content for one-shot request
                api_start = time.time()
                response = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents=physics_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                        thinking_config=types.ThinkingConfig(include_thoughts=False)  # Disable to save tokens for output
                    )
                )
                physics_rules = response.text
                api_time = time.time() - api_start

                # Count how many rules were generated
                rule_count = physics_rules.count('Rule ')

                print(f"✓ Physics Ground Truth Generated ({rule_count} rules):")
                print(f"  ⏱️  API call time: {api_time:.1f}s")
                print(f"{'─' * 80}")
                for line in physics_rules.split('\n'):
                    print(f"  {line}")
                print(f"{'─' * 80}")
                print(f"  Total length: {len(physics_rules)} characters")

                if rule_count < 3:
                    print(f"  ⚠️  WARNING: Only {rule_count} rules generated (expected 3). May need more tokens.")

                return physics_rules

            except Exception as e:
                error_msg = str(e)
                # Retry on 429 (rate limit), 503 (overloaded), or quota errors
                if "429" in error_msg or "503" in error_msg or "quota" in error_msg.lower() or "overloaded" in error_msg.lower():
                    if attempt < max_retries - 1:
                        print(f"⚠️  API error (attempt {attempt + 1}/{max_retries}): {error_msg[:100]}... Waiting {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        print(f"❌ Failed after {max_retries} retries: {error_msg}")
                        raise
                else:
                    print(f"❌ Error generating physics ground truth: {e}")
                    raise

        raise RuntimeError("Failed to generate physics ground truth after all retries")

    def compare_videos_gemini_tournament(self, video_paths: List[str], text_prompt: str,
                                         model_name: str = "gemini-2.5-flash-lite",
                                         batch_size: int = 5,
                                         max_output_tokens: int = 800,
                                         temperature: float = 0.2,
                                         prompt_alignment_checklist: str = None,
                                         max_stage3_workers: int = 5) -> str:
        """
        Compare videos using Gemini 2.5 Flash in a tournament-style elimination.

        NEW PIPELINE: Stage 1 (Visual Hallucination Filter) → Stage 2 (Forensic Observation) → Stage 3 (Tournament)
        Uses raw forensic logs without rigid rulebook filtering for more flexible physics evaluation.
        NOW WITH PARALLEL STAGE 3: Process multiple tournament batches concurrently.

        Args:
            video_paths: List of video file paths to compare
            text_prompt: The text prompt that was used to generate these videos
            model_name: Gemini model to use (default: gemini-2.5-flash-lite for cost efficiency)
            batch_size: Maximum videos per comparison batch (max 10 for Gemini Flash)
            max_output_tokens: Maximum tokens in response (default: 500, limits API cost)
            temperature: Sampling temperature (default: 0.2, lower = more deterministic)
            prompt_alignment_checklist: Pre-generated prompt alignment checklist (optional, will generate if not provided)
            max_stage3_workers: Maximum parallel workers for Stage 3 tournament (default: 5)

        Returns:
            Path to the winning video
        """
        if not self.use_gemini or self.gemini_client is None:
            raise RuntimeError("Gemini is not initialized. Set use_gemini=True in constructor.")

        # If a different model_name is requested, update the stored model name
        if hasattr(self, 'gemini_model_name') and self.gemini_model_name != model_name:
            print(f"Switching Gemini model from {self.gemini_model_name} to {model_name}")
            self.gemini_model_name = model_name

        if not video_paths:
            raise ValueError("No video paths provided")

        if len(video_paths) == 1:
            print(f"Only one video provided, returning it as the winner: {video_paths[0]}")
            return video_paths[0]

        current_candidates = video_paths.copy()
        round_num = 1

        print(f"\nStarting Gemini Tournament Comparison")
        print(f"   Total candidates: {len(current_candidates)}")
        print(f"   Text prompt: {text_prompt}")
        print(f"   Batch size: {batch_size}")

        # ========== GENERATE PROMPT ALIGNMENT CHECKLIST (ONCE AT START) ==========
        # Only generate if not provided (allows reuse across multiple video batches for same prompt)
        if prompt_alignment_checklist is None:
            print(f"\n{'='*80}")
            print(f"STEP 0: Generating Prompt Alignment Checklist")
            print(f"{'='*80}")
            prompt_alignment_checklist = self.generate_prompt_alignment_checklist(
                text_prompt=text_prompt,
                model_name=model_name,
                max_output_tokens=1500,
                temperature=0.1
            )
        else:
            print(f"\n{'='*80}")
            print(f"STEP 0: Using Pre-generated Prompt Alignment Checklist")
            print(f"{'='*80}")

        # New 3-stage pipeline: Process ALL videos through each stage sequentially
        # Stage 1: Visual Hallucination Filter → Stage 2: Forensic Observation → Stage 3: Compare forensic logs

        print(f"\n{'='*80}")
        print(f"STAGE 1: PROMPT ALIGNMENT & BACKGROUND CHECK - Processing ALL {len(current_candidates)} videos")
        print(f"{'='*80}")

        all_retained_videos, rejected_videos = self.run_parallel_stage1(
            video_paths=current_candidates,
            text_prompt=text_prompt,
            prompt_alignment_checklist=prompt_alignment_checklist,
            max_workers=5  # Parallel processing with 5 workers
        )

        if len(all_retained_videos) == 0:
            print(f"\n❌ Stage 1 eliminated ALL videos")
            print(f"   Defaulting to first video from original list: {os.path.basename(video_paths[0])}")
            return video_paths[0]

        print(f"\n{'='*80}")
        print(f"STAGE 2: FORENSIC OBSERVATION - Generating logs for {len(all_retained_videos)} retained videos")
        print(f"{'='*80}")

        all_forensic_reports = self.run_parallel_stage2_scribes(
            retained_videos=all_retained_videos,
            text_prompt=text_prompt,
            max_workers=3
        )

        print(f"\n{'='*80}")
        print(f"STAGE 3: TOURNAMENT SELECTION - Comparing {len(all_forensic_reports)} forensic logs")
        print(f"{'='*80}")

        final_winner = self.new_run_stage3_forensic_tournament(
            forensic_entries=all_forensic_reports,
            text_prompt=text_prompt,
            selection_batch_size=10,
            max_workers=max_stage3_workers
        )

        # Determine winner path
        if final_winner and 'path' in final_winner:
            winner_path = final_winner['path']
            print(f"\nTournament Complete! Winner: {os.path.basename(winner_path)}")
        else:
            winner_path = video_paths[0]
            print(f"\n❌ Tournament Failed: Could not select winner")
            print(f"   Defaulting to first video from original list: {os.path.basename(winner_path)}")

        return winner_path

    def _compare_forensic_batch_with_free_llm(self, forensic_batch: List[dict], text_prompt: str,
                                               stage3_round: int, batch_num: int,
                                               all_have_anomalies: bool = False) -> dict:
        """
        Compare a batch of forensic logs using FREE Qwen2.5-Coder-32B-Instruct (text-only, no API cost).
        This is used for Stage 3 tournament when use_free_llm_for_checklist=True.

        Args:
            forensic_batch: List of forensic entry dicts with 'candidate_num', 'forensic_log', 'path'
            text_prompt: Original generation prompt
            stage3_round: Current tournament round number
            batch_num: Current batch number within the round
            all_have_anomalies: Whether all candidates have anomalies (affects prompt wording)

        Returns:
            Winner entry dict with 'candidate_num', 'forensic_log', 'path', 'response'
        """
        import re
        import time

        if self.free_llm is None or self.free_llm_tokenizer is None:
            raise RuntimeError("Free LLM not initialized. Set use_free_llm_for_checklist=True in constructor.")

        # If only 1 video in batch, it automatically advances to next round
        if len(forensic_batch) == 1:
            print(f"      Round {stage3_round} Batch {batch_num}: Only 1 video - auto-advances")
            return forensic_batch[0]

        # Build forensic log table
        forensic_table = [f"candidate_{entry['candidate_num']}:\n{entry['forensic_log']}" for entry in forensic_batch]
        separator = "\n\n" + "="*80 + "\n\n"
        forensic_data_block = separator.join(forensic_table)

        # Build prompt based on whether all candidates have anomalies
        if all_have_anomalies:
            decision_process = """
## 5. DECISION PROCESS - ALL CANDIDATES HAVE ANOMALIES
⚠️ **IMPORTANT:** All candidates have visual anomalies. You must choose the LEAST SEVERE option.

**Anomaly Severity Ranking (Least → Most Severe):**
1. Background Stability (morphing walls/floor) - LEAST SEVERE
2. Friction (sliding without stopping)
3. Jitter (teleportation, frame skipping)
4. Boundaries (clipping through walls/floor)
5. Conservation (objects appearing/disappearing) - MOST SEVERE

**Process:**
- Step 1: Identify which anomaly each candidate has (from PART 3)
- Step 2: Rank candidates by anomaly severity
- Step 3: Among candidates with the same severity, apply the two-stage evaluation above
"""
        else:
            decision_process = """
## 5. DECISION PROCESS - CLEAN CANDIDATES ONLY
✅ **PRE-FILTERED:** All candidates below have NO visual anomalies (already filtered out).

**Your Task:**
Apply the two-stage evaluation process above to find the most physically plausible video that matches the prompt.
"""

        stage3_instruction = f"""
# ROLE: VIDEO QUALITY TOURNAMENT JUDGE
You are selecting the SINGLE BEST video that both matches the prompt AND demonstrates the most plausible physics.

## 1. THE PROMPT (GROUND TRUTH)
"{text_prompt}"

## 2. FORENSIC OBSERVATION LOGS (Round {stage3_round}, Batch {batch_num})
{forensic_data_block}

---

## 3. TWO-STAGE EVALUATION PROCESS

### STAGE 1: PROMPT ALIGNMENT FILTER (Must Pass)
**Question: Does the video show what the prompt asked for?**

**Required Elements Check:**
- All objects/subjects mentioned in prompt are present
- Actions/behaviors match prompt description
- Scene context matches (if specified: indoor/outdoor, surface type, lighting, etc.)

**Disqualifiers:**
- Missing core elements (prompt says "two balls" but video shows one)
- Wrong behavior (prompt says "rolling" but video shows bouncing)
- Wrong objects (prompt says "red ball" but video shows blue)

**Regarding Extra Details:**
- Minor additions are acceptable IF they don't contradict the prompt (e.g., if prompt says "ball bouncing," a wet ball with splash is acceptable IF the wetness doesn't contradict any prompt constraints)
- Penalize if additions contradict prompt specifications (e.g., prompt says "dry surface" but video shows water)
- When in doubt: Prefer videos that match the prompt exactly without unnecessary additions

**ACTION:** First, categorize all candidates:
- ALIGNED: Videos that show all required elements with correct behavior
- MISALIGNED: Videos missing elements or showing wrong behavior

**DECISION RULES:**
1. **If ANY videos are prompt-aligned:** ONLY consider those for Stage 2. Discard misaligned videos entirely.
2. **If ALL videos are misaligned:** You MUST still select a winner - choose the video with the LEAST SEVERE misalignment issues and proceed to Stage 2 with all candidates.

---

### STAGE 2: PHYSICS QUALITY RANKING
**Question: Which video demonstrates the MOST PLAUSIBLE PHYSICS OVERALL?**
(If some videos were aligned: compare only aligned ones. If all were misaligned: compare all and choose the least-bad option.)

**Evaluation Criteria:**

**A. Overall Physical Plausibility (70% weight):**
Focus on the BIG PICTURE of physics quality, not minor details:
- Does the overall motion look realistic and physically plausible?
- Are forces (gravity, friction, collisions) behaving correctly in general?
- Is energy dissipation happening in a believable way overall?
- Do material properties (deformation, elasticity, rigidity) make sense for the object type?
- Does the video show natural, continuous physical behavior from start to finish?

**B. Temporal Smoothness (30% weight):**
- Is the motion smooth and continuous without jumps or glitches?
- Does the physical behavior progress logically from beginning to end?

**CRITICAL - Comparing Forensic Logs:**
⚠️ **DO NOT penalize videos for having shorter or less detailed forensic logs!**

**CORRECT Comparison Approach:**
- Focus on the PHYSICS BEHAVIORS described, NOT the verbosity of the description
- A shorter log that says "ball bounces and comes to rest" is EQUALLY VALID as a longer log that says "ball exhibits progressive energy dissipation through elastic collisions, transitioning to rolling motion before achieving complete rest"
- Both describe the SAME physics - one is just more verbose
- Judge based on: Which VIDEO has better physics? NOT which LOG has more words

**WRONG Comparison Approach (DO NOT DO THIS):**
- ❌ "Candidate A is better because the log is more comprehensive and detailed"
- ❌ "Candidate B lost because the description is less complete"
- ❌ "Candidate A explicitly mentions X while Candidate B only briefly mentions it"
- ❌ Comparing how "thorough" or "detailed" the descriptions are

**RIGHT Comparison Approach (DO THIS):**
- ✅ "Candidate A demonstrates more realistic energy dissipation (continues bouncing indefinitely vs. comes to rest)"
- ✅ "Candidate B shows better collision physics (maintains momentum vs. loses energy too quickly)"
- ✅ Compare the PHYSICAL BEHAVIORS mentioned, not the number of words used to describe them

**Key Principle:**
If both logs describe the same physics behavior (e.g., both say "ball comes to rest"), they are EQUAL in physics quality regardless of verbosity differences. Only declare a winner if there's a REAL physics difference (e.g., one shows energy conservation violation, one shows unrealistic friction).

---

{decision_process}

---

## 4. MANDATORY RESPONSE FORMAT
YOU MUST INCLUDE ALL SECTIONS BELOW IN YOUR RESPONSE:

**Step 1 - Prompt Alignment Check (REQUIRED):**
Candidate X: [ALIGNED / MISALIGNED] - [Brief reason]
Candidate Y: [ALIGNED / MISALIGNED] - [Brief reason]
...
(List ALL candidates)

**Step 2 - Overall Physics Quality (REQUIRED):**
(Only evaluate aligned videos if any exist. If all misaligned, evaluate all.)
Candidate X: [Brief assessment of overall physical plausibility and naturalness]
Candidate Y: [Brief assessment of overall physical plausibility and naturalness]
...
(Assess ALL candidates from Step 1 that you're evaluating)

**Winner: [Candidate with best OVERALL physics quality]**

---

**FINAL OUTPUT (REQUIRED - Must appear at end):**
BEST_VIDEO: Video_candidate_X
WINNING_REASON: [One sentence: (1) state alignment status, (2) describe why this video has the best overall physics. If all were misaligned, note it has "least severe misalignment issues."]
COMPARATIVE_NOTE: [One sentence explaining why the second-best candidate lost. MUST be based on PHYSICS DIFFERENCES, NOT description detail/verbosity differences.]

⚠️ **WARNING:** Your COMPARATIVE_NOTE must explain a REAL PHYSICS DIFFERENCE, not a description quality difference.
- ❌ WRONG: "Candidate X lost because its description was less detailed/comprehensive/complete"
- ❌ WRONG: "Candidate X's log didn't explicitly mention Y while the winner did"
- ✅ RIGHT: "Candidate X lost because it showed unrealistic friction/gravity/energy dissipation"
- ✅ RIGHT: "Candidate X lost because of [specific physics behavior issue described in the log]"

IMPORTANT: You must include ALL sections above (Step 1, Step 2, Winner, and Final Output). Do not skip any section.
"""

        print(f"      Round {stage3_round} Batch {batch_num}: Comparing {len(forensic_batch)} videos with FREE LLM...")

        # Retry up to 3 times if no winner is found
        max_retries = 3
        winner_found = False

        for retry_attempt in range(max_retries):
            if retry_attempt > 0:
                print(f"      ⚠️  Retry attempt {retry_attempt}/{max_retries - 1} - no winner found in previous response")

            # Call free LLM with the tournament instruction
            llm_start = time.time()

            # Build messages in chat format
            messages = [
                {"role": "user", "content": stage3_instruction}
            ]

            # Apply chat template
            text = self.free_llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize and generate
            model_inputs = self.free_llm_tokenizer([text], return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.free_llm.generate(
                    **model_inputs,
                    max_new_tokens=2000,
                    temperature=0.05,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.free_llm_tokenizer.eos_token_id
                )

            # Decode response (skip input tokens)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            stage3_text = self.free_llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            llm_time = time.time() - llm_start

            print(f"      ⏱️  Free LLM inference time: {llm_time:.1f}s")
            print(f"\n      Round {stage3_round} Batch {batch_num} Winner:")
            print(f"      {'-' * 60}")
            for line in stage3_text.split('\n'):
                print(f"      {line}")
            print(f"      {'-' * 60}")

            # Parse winner - accept multiple formats: "BEST_VIDEO: Video_candidate_X", "**Winner: Video_candidate_X**", "BEST_VIDEO: candidate_X"
            winner_match = re.search(r'(?:BEST_VIDEO:|\*\*Winner:)\s*(?:Video_)?candidate[_\s]?(\d+)', stage3_text, re.IGNORECASE)
            if winner_match:
                winner_num = int(winner_match.group(1))

                # Find the entry with this candidate number
                for entry in forensic_batch:
                    if entry['candidate_num'] == winner_num:
                        winner_entry = {
                            'candidate_num': entry['candidate_num'],
                            'forensic_log': entry['forensic_log'],
                            'path': entry['path'],
                            'response': stage3_text
                        }
                        winner_found = True
                        break

                if winner_found:
                    break  # Exit retry loop

        if not winner_found:
            # Fallback: use first candidate in batch
            print(f"      ⚠️  Warning: No winner found after {max_retries} attempts. Using first candidate as fallback.")
            winner_entry = forensic_batch[0]

        return winner_entry


    def new_run_stage3_forensic_tournament(self, forensic_entries: List[dict], text_prompt: str,
                                           selection_batch_size: int = 5,
                                           max_workers: int = 5) -> dict:
        """
        NEW Stage 3: Compare forensic logs with automatic anomaly pre-filtering.
        NOW WITH PARALLEL PROCESSING: Process multiple batches concurrently for faster tournaments.

        Process:
        1. Pre-filter: Check PART 3 of each forensic log for "Observed" anomalies
        2. Tournament: Compare only clean candidates (or least severe if all have anomalies)
        3. Flash decides winner based on physics plausibility and temporal consistency
        4. Parallelize: Process up to max_workers batches concurrently per round

        Args:
            forensic_entries: List of dicts with 'candidate_num', 'forensic_log', 'path'
            text_prompt: Original generation prompt
            selection_batch_size: How many reports to compare at once (default 5)
            max_workers: Maximum number of parallel API calls per round (default 5)

        Returns:
            Dict with 'winner', 'clean_count', 'disqualified_count' or None if failed
        """
        import re

        if len(forensic_entries) == 0:
            return {'winner': None, 'clean_count': 0, 'disqualified_count': 0}

        # If only 1 video, it wins by default
        if len(forensic_entries) == 1:
            winner = forensic_entries[0]
            winner['response'] = f"BEST_VIDEO: Video_candidate_{winner['candidate_num']}\nWINNING_REASON: Only candidate that passed Stage 1.\nCOMPARATIVE_NOTE: No other candidates to compare."
            print(f"\n   Stage 3: Only 1 candidate - Winner by default: candidate_{winner['candidate_num']}")
            return {'winner': winner, 'clean_count': 1, 'disqualified_count': 0}

        print(f"\n   Stage 3: Processing {len(forensic_entries)} forensic logs in batches of {selection_batch_size}...")

        # =========================================================================
        # PRE-FILTER: Remove candidates with observed anomalies
        # =========================================================================
        print(f"\n   🔍 Pre-filtering: Checking for visual anomalies...")
        clean_candidates = []
        disqualified_candidates = []

        for entry in forensic_entries:
            forensic_log = entry['forensic_log']

            # Check if ANY anomaly is "Observed" in PART 3
            part3_match = re.search(r'## PART 3: VISUAL ANOMALY SCAN\s+(.*?)(?=##|\Z)',
                                   forensic_log, re.DOTALL | re.IGNORECASE)
            if part3_match:
                part3_text = part3_match.group(1)
                # Check for any "Observed" status (but NOT "None observed")
                # Look for patterns like "Observed" that are NOT preceded by "None"
                has_anomaly = False
                for line in part3_text.split('\n'):
                    # Skip empty lines
                    if not line.strip():
                        continue
                    # Check if line contains "Observed" but NOT "None observed"
                    if re.search(r'\bObserved\b', line, re.IGNORECASE) and not re.search(r'\bNone\s+observed\b', line, re.IGNORECASE):
                        has_anomaly = True
                        break

                if has_anomaly:
                    disqualified_candidates.append(entry)
                    print(f"      ❌ candidate_{entry['candidate_num']} DISQUALIFIED: Visual anomaly detected")
                else:
                    clean_candidates.append(entry)
                    print(f"      ✅ candidate_{entry['candidate_num']} CLEAN: No anomalies")
            else:
                # No Part 3 found, allow it (shouldn't happen, but safe fallback)
                clean_candidates.append(entry)
                print(f"      ⚠️  candidate_{entry['candidate_num']} CLEAN (no Part 3 found)")

        # If no clean candidates, use all but note this
        if len(clean_candidates) == 0:
            print(f"\n   ⚠️  All {len(forensic_entries)} candidates have anomalies - choosing least severe")
            current_candidates = forensic_entries[:]
            all_have_anomalies = True
        else:
            print(f"\n   ✓ {len(clean_candidates)} clean candidates after anomaly filtering")
            print(f"   ✗ {len(disqualified_candidates)} candidates disqualified")
            current_candidates = clean_candidates
            all_have_anomalies = False

        # If only 1 clean candidate remains, it wins
        if len(current_candidates) == 1:
            winner = current_candidates[0]
            winner['response'] = f"BEST_VIDEO: Video_candidate_{winner['candidate_num']}\nWINNING_REASON: Only candidate without visual anomalies.\nCOMPARATIVE_NOTE: All other candidates disqualified due to observed anomalies."
            print(f"\n   Stage 3: Only 1 clean candidate - Winner by default: candidate_{winner['candidate_num']}")
            return winner

        stage3_round = 1

        while len(current_candidates) > 1:
            print(f"\n   Stage 3 Round {stage3_round}: Comparing {len(current_candidates)} candidates...")
            next_round_winners = []

            # Create batches for this round
            batches = []
            for s3_batch_idx in range(0, len(current_candidates), selection_batch_size):
                s3_batch = current_candidates[s3_batch_idx:s3_batch_idx + selection_batch_size]
                batch_num = s3_batch_idx // selection_batch_size + 1
                batches.append((s3_batch, batch_num))

            print(f"   Processing {len(batches)} batches in parallel (max {max_workers} workers)...")

            # Process batches in parallel
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def process_single_batch(batch_data):
                """Helper function to process a single tournament batch

                Routes to either FREE Qwen2.5-Coder-32B (if use_free_llm_for_checklist=True)
                or Gemini API (default).
                """
                s3_batch, batch_num = batch_data

                # Route to FREE LLM if enabled (100% free, no API cost)
                if self.use_free_llm_for_checklist:
                    return self._compare_forensic_batch_with_free_llm(
                        forensic_batch=s3_batch,
                        text_prompt=text_prompt,
                        stage3_round=stage3_round,
                        batch_num=batch_num,
                        all_have_anomalies=all_have_anomalies
                    )

                # Otherwise use Gemini (original implementation with API cost)
                # If only 1 video in batch, it automatically advances to next round
                if len(s3_batch) == 1:
                    print(f"      Round {stage3_round} Batch {batch_num}: Only 1 video - auto-advances")
                    return s3_batch[0]

                # Build forensic log table
                forensic_table = [f"candidate_{entry['candidate_num']}:\n{entry['forensic_log']}" for entry in s3_batch]
                separator = "\n\n" + "="*80 + "\n\n"
                forensic_data_block = separator.join(forensic_table)

                # Build prompt based on whether all candidates have anomalies
                if all_have_anomalies:
                    decision_process = """
## 5. DECISION PROCESS - ALL CANDIDATES HAVE ANOMALIES
⚠️ **IMPORTANT:** All candidates have visual anomalies. You must choose the LEAST SEVERE option.

**Anomaly Severity Ranking (Least → Most Severe):**
1. Background Stability (morphing walls/floor) - LEAST SEVERE
2. Friction (sliding without stopping)
3. Jitter (teleportation, frame skipping)
4. Boundaries (clipping through walls/floor)
5. Conservation (objects appearing/disappearing) - MOST SEVERE

**Process:**
- Step 1: Identify which anomaly each candidate has (from PART 3)
- Step 2: Rank candidates by anomaly severity
- Step 3: Among candidates with the same severity, apply the two-stage evaluation above
"""
                else:
                    decision_process = """
## 5. DECISION PROCESS - CLEAN CANDIDATES ONLY
✅ **PRE-FILTERED:** All candidates below have NO visual anomalies (already filtered out).

**Your Task:**
Apply the two-stage evaluation process above to find the most physically plausible video that matches the prompt.
"""

                stage3_instruction = f"""
# ROLE: VIDEO QUALITY TOURNAMENT JUDGE
You are selecting the SINGLE BEST video that both matches the prompt AND demonstrates the most plausible physics.

## 1. THE PROMPT (GROUND TRUTH)
"{text_prompt}"

## 2. FORENSIC OBSERVATION LOGS (Round {stage3_round}, Batch {batch_num})
{forensic_data_block}

---

## 3. TWO-STAGE EVALUATION PROCESS

### STAGE 1: PROMPT ALIGNMENT FILTER (Must Pass)
**Question: Does the video show what the prompt asked for?**

**Required Elements Check:**
- All objects/subjects mentioned in prompt are present
- Actions/behaviors match prompt description
- Scene context matches (if specified: indoor/outdoor, surface type, lighting, etc.)

**Disqualifiers:**
- Missing core elements (prompt says "two balls" but video shows one)
- Wrong behavior (prompt says "rolling" but video shows bouncing)
- Wrong objects (prompt says "red ball" but video shows blue)

**Regarding Extra Details:**
- Minor additions are acceptable IF they don't contradict the prompt (e.g., if prompt says "ball bouncing," a wet ball with splash is acceptable IF the wetness doesn't contradict any prompt constraints)
- Penalize if additions contradict prompt specifications (e.g., prompt says "dry surface" but video shows water)
- When in doubt: Prefer videos that match the prompt exactly without unnecessary additions

**ACTION:** First, categorize all candidates:
- ALIGNED: Videos that show all required elements with correct behavior
- MISALIGNED: Videos missing elements or showing wrong behavior

**DECISION RULES:**
1. **If ANY videos are prompt-aligned:** ONLY consider those for Stage 2. Discard misaligned videos entirely.
2. **If ALL videos are misaligned:** You MUST still select a winner - choose the video with the LEAST SEVERE misalignment issues and proceed to Stage 2 with all candidates.

---

### STAGE 2: PHYSICS QUALITY RANKING
**Question: Which video demonstrates the MOST PLAUSIBLE PHYSICS OVERALL?**
(If some videos were aligned: compare only aligned ones. If all were misaligned: compare all and choose the least-bad option.)

**Evaluation Criteria:**

**A. Overall Physical Plausibility (70% weight):**
Focus on the BIG PICTURE of physics quality, not minor details:
- Does the overall motion look realistic and physically plausible?
- Are forces (gravity, friction, collisions) behaving correctly in general?
- Is energy dissipation happening in a believable way overall?
- Do material properties (deformation, elasticity, rigidity) make sense for the object type?
- Does the video show natural, continuous physical behavior from start to finish?

**B. Temporal Smoothness (30% weight):**
- Is the motion smooth and continuous without jumps or glitches?
- Does the physical behavior progress logically from beginning to end?

**CRITICAL - Comparing Forensic Logs:**
⚠️ **DO NOT penalize videos for having shorter or less detailed forensic logs!**

**CORRECT Comparison Approach:**
- Focus on the PHYSICS BEHAVIORS described, NOT the verbosity of the description
- A shorter log that says "ball bounces and comes to rest" is EQUALLY VALID as a longer log that says "ball exhibits progressive energy dissipation through elastic collisions, transitioning to rolling motion before achieving complete rest"
- Both describe the SAME physics - one is just more verbose
- Judge based on: Which VIDEO has better physics? NOT which LOG has more words

**WRONG Comparison Approach (DO NOT DO THIS):**
- ❌ "Candidate A is better because the log is more comprehensive and detailed"
- ❌ "Candidate B lost because the description is less complete"
- ❌ "Candidate A explicitly mentions X while Candidate B only briefly mentions it"
- ❌ Comparing how "thorough" or "detailed" the descriptions are

**RIGHT Comparison Approach (DO THIS):**
- ✅ "Candidate A demonstrates more realistic energy dissipation (continues bouncing indefinitely vs. comes to rest)"
- ✅ "Candidate B shows better collision physics (maintains momentum vs. loses energy too quickly)"
- ✅ Compare the PHYSICAL BEHAVIORS mentioned, not the number of words used to describe them

**Key Principle:**
If both logs describe the same physics behavior (e.g., both say "ball comes to rest"), they are EQUAL in physics quality regardless of verbosity differences. Only declare a winner if there's a REAL physics difference (e.g., one shows energy conservation violation, one shows unrealistic friction).

---

{decision_process}

---

## 4. MANDATORY RESPONSE FORMAT
YOU MUST INCLUDE ALL SECTIONS BELOW IN YOUR RESPONSE:

**Step 1 - Prompt Alignment Check (REQUIRED):**
Candidate X: [ALIGNED / MISALIGNED] - [Brief reason]
Candidate Y: [ALIGNED / MISALIGNED] - [Brief reason]
...
(List ALL candidates)

**Step 2 - Overall Physics Quality (REQUIRED):**
(Only evaluate aligned videos if any exist. If all misaligned, evaluate all.)
Candidate X: [Brief assessment of overall physical plausibility and naturalness]
Candidate Y: [Brief assessment of overall physical plausibility and naturalness]
...
(Assess ALL candidates from Step 1 that you're evaluating)

**Winner: [Candidate with best OVERALL physics quality]**

---

**FINAL OUTPUT (REQUIRED - Must appear at end):**
BEST_VIDEO: Video_candidate_X
WINNING_REASON: [One sentence: (1) state alignment status, (2) describe why this video has the best overall physics. If all were misaligned, note it has "least severe misalignment issues."]
COMPARATIVE_NOTE: [One sentence explaining why the second-best candidate lost. MUST be based on PHYSICS DIFFERENCES, NOT description detail/verbosity differences.]

⚠️ **WARNING:** Your COMPARATIVE_NOTE must explain a REAL PHYSICS DIFFERENCE, not a description quality difference.
- ❌ WRONG: "Candidate X lost because its description was less detailed/comprehensive/complete"
- ❌ WRONG: "Candidate X's log didn't explicitly mention Y while the winner did"
- ✅ RIGHT: "Candidate X lost because it showed unrealistic friction/gravity/energy dissipation"
- ✅ RIGHT: "Candidate X lost because of [specific physics behavior issue described in the log]"

IMPORTANT: You must include ALL sections above (Step 1, Step 2, Winner, and Final Output). Do not skip any section.
"""

                print(f"      Round {stage3_round} Batch {batch_num}: Comparing {len(s3_batch)} videos...")

                # Retry up to 3 times if no winner is found
                max_retries = 3
                winner_found = False

                for retry_attempt in range(max_retries):
                    if retry_attempt > 0:
                        print(f"      ⚠️  Retry attempt {retry_attempt}/{max_retries - 1} - no winner found in previous response")

                    # Use generate_content for stateless comparison
                    api_start = time.time()
                    stage3_response = self.gemini_client.models.generate_content(
                        model=self.gemini_model_name,
                        contents=stage3_instruction,
                        config=types.GenerateContentConfig(
                            max_output_tokens=8192,  # Maximum for Gemini 2.5 Flash
                            temperature=0.05,
                            thinking_config=types.ThinkingConfig(include_thoughts=True)
                        )
                    )
                    stage3_text = stage3_response.text
                    api_time = time.time() - api_start

                    print(f"      ⏱️  API call time: {api_time:.1f}s")
                    print(f"\n      Round {stage3_round} Batch {batch_num} Winner:")
                    print(f"      {'-' * 60}")
                    for line in stage3_text.split('\n'):
                        print(f"      {line}")
                    print(f"      {'-' * 60}")

                    # Parse winner - accept multiple formats: "BEST_VIDEO: Video_candidate_X", "**Winner: Video_candidate_X**", "BEST_VIDEO: candidate_X"
                    winner_match = re.search(r'(?:BEST_VIDEO:|\*\*Winner:)\s*(?:Video_)?candidate[_\s]?(\d+)', stage3_text, re.IGNORECASE)
                    if winner_match:
                        winner_num = int(winner_match.group(1))

                        # Find the entry with this candidate number
                        for entry in s3_batch:
                            if entry['candidate_num'] == winner_num:
                                winner_entry = {
                                    'candidate_num': entry['candidate_num'],
                                    'forensic_log': entry['forensic_log'],
                                    'path': entry['path'],
                                    'response': stage3_text
                                }
                                winner_found = True
                                break

                        if winner_found:
                            break  # Exit retry loop

                if not winner_found:
                    # Fallback: use first candidate in batch
                    print(f"      ⚠️  Warning: No winner found after {max_retries} attempts. Using first candidate as fallback.")
                    winner_entry = s3_batch[0]

                return winner_entry

            # Execute batches in parallel with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all batch processing tasks
                future_to_batch = {executor.submit(process_single_batch, batch_data): batch_data
                                  for batch_data in batches}

                # Collect results as they complete
                for future in as_completed(future_to_batch):
                    batch_data = future_to_batch[future]
                    try:
                        winner = future.result()
                        next_round_winners.append(winner)
                    except Exception as e:
                        batch_num = batch_data[1]
                        print(f"      ❌ Error processing batch {batch_num}: {e}")
                        # Fallback: use first candidate from failed batch
                        s3_batch = batch_data[0]
                        if len(s3_batch) > 0:
                            next_round_winners.append(s3_batch[0])

            # Prepare for next round
            current_candidates = next_round_winners
            stage3_round += 1

        # Return final winner with statistics
        clean_count = len(clean_candidates)
        disqualified_count = len(disqualified_candidates)

        if len(current_candidates) == 1:
            final_winner = current_candidates[0]
            print(f"\n   Stage 3 Complete (after {stage3_round - 1} rounds) - Winner: candidate_{final_winner['candidate_num']}")
            return {'winner': final_winner, 'clean_count': clean_count, 'disqualified_count': disqualified_count}
        else:
            print(f"   ⚠️  Stage 3: No winner found")
            return {'winner': None, 'clean_count': clean_count, 'disqualified_count': disqualified_count}

    def _wait_for_files_active(self, uploaded_files: List) -> None:
        """
        Wait for all uploaded files to finish processing and become ACTIVE.

        Args:
            uploaded_files: List of uploaded file objects from Gemini API

        Raises:
            RuntimeError: If any file fails to become active after timeout
        """
        max_wait_time = 900  # 15 minutes max wait (video processing can be slow)
        check_interval = 5   # Check every 5 seconds

        for file_obj in uploaded_files:
            file_start = time.time()
            consecutive_503_errors = 0
            while True:
                try:
                    # Get current file status
                    file_status = self.gemini_client.files.get(name=file_obj.name)
                    consecutive_503_errors = 0  # Reset on success

                    if file_status.state.name == "ACTIVE":
                        break  # File is ready
                    elif file_status.state.name == "FAILED":
                        raise RuntimeError(f"File processing failed: {file_obj.name}")

                    # Check if we've exceeded max wait time
                    elapsed = time.time() - file_start
                    if elapsed > max_wait_time:
                        raise RuntimeError(f"Timeout waiting for file to process: {file_obj.name} (waited {elapsed:.0f}s)")

                    # Progress update every 30 seconds
                    if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                        print(f"      ⏳ Still waiting for file processing... ({elapsed:.0f}s elapsed, state: {file_status.state.name})")

                    # Wait before checking again
                    time.sleep(check_interval)

                except Exception as e:
                    # Handle 503 Service Unavailable errors with exponential backoff
                    if '503' in str(e) or 'UNAVAILABLE' in str(e):
                        consecutive_503_errors += 1
                        if consecutive_503_errors > 10:
                            raise RuntimeError(f"Gemini API persistently unavailable after 10 retries: {e}")

                        backoff_time = min(60, 5 * (2 ** (consecutive_503_errors - 1)))  # Exponential backoff, max 60s
                        print(f"      ⚠️  Gemini API 503 error (attempt {consecutive_503_errors}/10), retrying in {backoff_time}s...")
                        time.sleep(backoff_time)
                    else:
                        # Re-raise non-503 errors
                        raise

    # =========================================================================
    # PARALLEL PROCESSING METHODS (Threading-based for Gemini API)
    # =========================================================================

    def evaluate_single_video_stage1(self, video_path: str, text_prompt: str,
                                      prompt_alignment_checklist: str) -> dict:
        """
        Evaluate a single video through Stage 1 (Visual Hallucination Filter).
        Thread-safe method for parallel processing.

        Supports both Gemini API and FREE Qwen3-VL-30B.

        Args:
            video_path: Path to video file
            text_prompt: Generation prompt
            prompt_alignment_checklist: Pre-generated checklist

        Returns:
            Dict with 'passed': bool, 'path': str, 'candidate_num': int, 'response': str
        """
        # Route to Qwen3-VL if enabled (FREE alternative to Gemini)
        if self.use_qwen_vl:
            return self._evaluate_video_with_qwen_vl_stage1(
                video_path, text_prompt, prompt_alignment_checklist
            )

        # Otherwise use Gemini (original implementation)
        import re
        import time
        from google.genai import types

        # Extract candidate/particle number from filename
        filename = os.path.basename(video_path)
        # Support both 'candidate_' (from inference.py) and 'particle' (from smc_inference.py)
        if 'candidate_' in filename:
            candidate_match = filename.split('candidate_')[1].split('_')[0]
        elif 'particle' in filename:
            candidate_match = filename.split('particle')[1].split('_')[0].split('.')[0]
        else:
            candidate_match = '0'
        candidate_num = int(candidate_match)

        try:
            # Upload video with retry on failure
            max_upload_retries = 3
            uploaded_file = None
            for attempt in range(max_upload_retries):
                try:
                    uploaded_file = self.gemini_client.files.upload(file=video_path)
                    break  # Success
                except Exception as upload_error:
                    if attempt < max_upload_retries - 1:
                        print(f"      ⚠️  Upload attempt {attempt+1}/{max_upload_retries} failed for candidate_{candidate_num}: {upload_error}")
                        print(f"      Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        # Final attempt failed, mark as failed with error
                        print(f"      ❌ All upload attempts failed for candidate_{candidate_num}: {upload_error}")
                        return {
                            'passed': False,
                            'path': video_path,
                            'candidate_num': candidate_num,
                            'response': f"Error: Failed to upload video after {max_upload_retries} attempts: {upload_error}",
                            'filename': filename
                        }

            # Wait for processing
            self._wait_for_files_active([uploaded_file])

            # Build Stage 1 prompt (single video)
            stage1_prompt_parts = [
                f"""
# ROLE: VISUAL HALLUCINATION HUNTER
**GOAL:** You are a "Glitch Hunter." Your job is to find visual errors that break the laws of physics or identity.
**METHOD:** Continuous Frame-by-Frame Scan.
**STRICTNESS:** Zero Tolerance. If you see a glitch for 0.1 seconds, it is a FAIL.

---
## STEP 1: LOAD THE BLUEPRINT
{prompt_alignment_checklist}

*Verify the video strictly against the [VISUAL_SPECS] above.*

---
## STEP 2: THE "GLITCH SCAN" (Specific Visual Anchors)
Watch the entire video (0:00 to End). Look for these **3 Specific Failure Modes**.

### 1. THE "GHOST" SCAN (Cardinality & Negative Space)
**Task:** Scan the **EMPTY SPACE** around the main subject.
* **The Glitch (Multiplicity):** Does a second object flicker into existence?
* **The Glitch (Vanishing):** Does the subject disappear without leaving the frame?
* **FAIL TRIGGER:** If the visible count of [Subject_Name] is ever NOT [Expected_Count].

### 2. THE "MUTANT" SCAN (Topology & Integrity)
**Task:** Watch the **SILHOUETTE** of the main subject.
* **The Glitch (Stacking/Fusion):** Does the subject look like two objects stuck together?
* **The Glitch (Morphing):** Does the subject change geometry?
* **FAIL TRIGGER:** If the subject loses its defined [Subject_Shape] or splits/merges.

### 3. THE "MATRIX" SCAN (Background Stability)
**Task:** Watch the **FIXED FEATURES** (Floor tiles, Wall corners).
* **The Glitch:** Do textures slide or breathe?
* **FAIL TRIGGER:** If the floor moves independently of the camera.

---
## OUTPUT FORMAT (STRICT):

**GLITCH_LOG:**
* **Ghost_Log:** [e.g., "0:02 (2nd object flickered in background)", or "NONE"]
* **Mutant_Log:** [e.g., "0:04 (Subject developed 'snowman' stack)", or "NONE"]
* **Stability_Log:** [e.g., "0:03 (Floor texture sliding)", or "NONE"]

**VERDICT:**
* [PASS / FAIL]
""",
                types.Part(
                    file_data=types.FileData(
                        file_uri=uploaded_file.uri,
                        mime_type=uploaded_file.mime_type
                    ),
                    video_metadata=types.VideoMetadata(fps=10)
                )
            ]

            # Call Gemini
            stage1_response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=stage1_prompt_parts,
                config=types.GenerateContentConfig(
                    max_output_tokens=5000,
                    temperature=0.05,
                    thinking_config=types.ThinkingConfig(include_thoughts=False)
                )
            )
            stage1_text = stage1_response.text

            # Parse verdict
            verdict_pattern = r'\*\*VERDICT:\*\*\s*[\*\-]*\s*(PASS|FAIL)'
            verdict_match = re.search(verdict_pattern, stage1_text, re.IGNORECASE)

            video_passed = False
            if verdict_match:
                verdict = verdict_match.group(1).upper()
                video_passed = (verdict == 'PASS')

            # Cleanup
            try:
                self.gemini_client.files.delete(name=uploaded_file.name)
            except Exception:
                pass

            return {
                'passed': video_passed,
                'path': video_path,
                'candidate_num': candidate_num,
                'response': stage1_text,
                'filename': filename
            }

        except Exception as e:
            print(f"      ❌ Error evaluating candidate_{candidate_num}: {e}")
            return {
                'passed': False,
                'path': video_path,
                'candidate_num': candidate_num,
                'response': f"Error: {e}",
                'filename': filename
            }

    def evaluate_single_video_stage2_scribe(self, video_path: str, text_prompt: str,
                                             candidate_num: int) -> dict:
        """
        Generate SCRIBE forensic log for a single video (Stage 2).
        Supports both Gemini API and FREE Qwen3-VL-30B.
        Thread-safe method for parallel processing.

        Args:
            video_path: Path to video file
            text_prompt: Generation prompt
            candidate_num: Candidate number

        Returns:
            Dict with 'candidate_num', 'forensic_log', 'path'
        """
        # Route to Qwen3-VL if enabled (FREE alternative to Gemini)
        if self.use_qwen_vl:
            return self._evaluate_video_with_qwen_vl_stage2(video_path, text_prompt, candidate_num)

        # Otherwise use Gemini (original implementation)
        from google.genai import types
        import time

        try:
            # Upload video with retry on failure
            max_upload_retries = 3
            uploaded_file = None
            for attempt in range(max_upload_retries):
                try:
                    uploaded_file = self.gemini_client.files.upload(file=video_path)
                    break  # Success
                except Exception as upload_error:
                    if attempt < max_upload_retries - 1:
                        print(f"      ⚠️  Upload attempt {attempt+1}/{max_upload_retries} failed for candidate_{candidate_num}: {upload_error}")
                        print(f"      Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        # Final attempt failed, return error forensic log
                        print(f"      ❌ All upload attempts failed for candidate_{candidate_num}: {upload_error}")
                        return {
                            'candidate_num': candidate_num,
                            'forensic_log': f"Error: Failed to upload video after {max_upload_retries} attempts: {upload_error}",
                            'path': video_path
                        }

            # Wait for processing
            self._wait_for_files_active([uploaded_file])

            # Build SCRIBE prompt
            scribe_prompt = [
                types.Part(
                    file_data=types.FileData(
                        file_uri=uploaded_file.uri,
                        mime_type=uploaded_file.mime_type
                    ),
                    video_metadata=types.VideoMetadata(fps=10)
                ),
                f"""
# ROLE: FORENSIC PHYSICS OBSERVER
**TASK:** Document this video like a crash test analysis - capture EVERY physical state change.
**DO NOT JUDGE.** Record observable facts only.
**CRITICAL:** Track the ENTIRE FRAME + the PHYSICS of motion, not just object positions.

**Context Prompt:** "{text_prompt}"

---
## PART 1: FULL SCENE NARRATION
Describe the ENTIRE visual scene with physics focus.

**A. Main Subject(s) - Physical State Analysis:**
- What is the primary object/action?
- **Motion states:** Describe transitions between states (airborne → contact → compression → rebound)
- **Shape changes:** Note deformation during impacts, compression depth, recovery
- **Energy dissipation:** Track height loss between bounces, rotation speed changes
- **Contact dynamics:** Duration of floor contact, bounce intensity changes

**B. Secondary Objects & Negative Space:**
- Scan ENTIRE FRAME: Any unexpected objects appearing/disappearing/flickering?
- Background objects: Stable or morphing/shifting?
- Void space: Anything materializing where nothing should exist?

**C. Environment & Background:**
- Floor, walls, lighting, shadows - describe and track consistency
- Environmental changes, morphing, or unexpected transitions?
- Surface wetness, debris, or other state changes?

## PART 2: PHYSICS EVENT LOG (16-20 timestamps)
Record key PHYSICAL EVENTS and STATE TRANSITIONS, not just positions.

**CRITICAL SAMPLING RULES:**
✓ Temporal coverage: 4+ samples from each video quarter (beginning/early/middle/late)
✓ **EVENT-BASED sampling:** Prioritize capturing:
  - Impact moments (contact initiation)
  - Maximum compression states
  - Launch/separation from surface
  - Apex points (peak height)
  - State transitions (wet→dry, fast→slow, rotating→still)

**Format:**
`T=[time] | State: [airborne/contact/compressed] | Main Action: [...] | Physics: [height, deformation, velocity] | Background: [...] | Anomalies: [...]`

**Capture these details:**
- **Contact events:** "Ball strikes floor", "Maximum compression at X%", "Launches from surface"
- **Deformation:** "Flattened to 80% of original height", "Rebounds to sphere shape"
- **Height tracking:** "Peak at ~50cm", "Second bounce reaches ~30cm" (relative measurements)
- **Velocity changes:** "Decelerating", "Accelerating downward", "Slowing rotation"
- **Full frame:** Background stability, object count, unexpected elements
- **Surface effects:** Water splash, debris displacement, floor reflections

**EXAMPLE ENTRY:**
`T=00:01 | State: contact/compressed | Main Action: Ball strikes floor with maximum deformation, compressed to ~70% height, water splash radiates outward | Physics: Kinetic energy converting to elastic potential, rotation continuing | Background: Floor tiles stable, reflection visible, grout lines clear | Anomalies: None`

## PART 3: VISUAL ANOMALY SCAN
Full-frame scan for visual/physical glitches. For each category, respond with **"Observed"** if the anomaly is present, or **"None"** if not observed.

**REQUIRED FORMAT:** Each line must contain either "Observed" or "None". Do NOT use "Anomalies: None" or prose descriptions.

**EXAMPLE FORMAT (if no anomalies):**
1. **Conservation:** None
2. **Boundaries:** None
3. **Jitter:** None
4. **Friction/Gravity:** None
5. **Contact Physics:** None
6. **Background Stability:** None
7. **Object Multiplicity:** None
8. **Material Properties:** None

**EXAMPLE FORMAT (if anomalies found):**
1. **Conservation:** Observed - Ball's surface colors morph dynamically without rotation, indicating visual information appearing/disappearing
2. **Boundaries:** None
3. **Jitter:** None
...

**CATEGORIES TO CHECK:**
1. **Conservation:** Mass/energy violations? Examples: Objects appearing/disappearing, surface patterns morphing or changing without rotation, energy increasing (ball bouncing higher)
2. **Boundaries:** Clipping through surfaces? Penetration without collision?
3. **Jitter:** Teleportation, frame skipping, position discontinuities?
4. **Friction/Gravity:** Unrealistic sliding, floating, or hovering? Energy NOT decreasing?
5. **Contact Physics:** Objects passing through each other? Missing collisions?
6. **Background Stability:** Morphing walls/floor, scene transitions, background elements changing?
7. **Object Multiplicity:** Duplicate instances appearing/flickering?
8. **Material Properties:** Abrupt changes in material behavior? Examples: Dry→wet transitions without water source, rigid object suddenly becoming elastic, color/texture changes that don't match physical interactions

**OUTPUT FORMAT:** Text report with all sections above. PART 3 must use "Observed" or "None" keywords as shown.

**IMPORTANT:** Each anomaly type should be DISTINCT. For surface pattern changes, use Conservation. For material property changes (wet/dry, rigid/elastic), use Material Properties.
"""
            ]

            # Call Gemini (no thinking_config to avoid hidden thinking tokens eating the budget)
            scribe_response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=scribe_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=15000, 
                    temperature=0.0
                    # No thinking_config - SCRIBE is purely observational, no reasoning needed
                )
            )
            forensic_log = scribe_response.text

            # Debug: Check if response was truncated
            if hasattr(scribe_response, 'candidates') and scribe_response.candidates:
                candidate = scribe_response.candidates[0]
                finish_reason = candidate.finish_reason

                if finish_reason != 'STOP':
                    print(f"      ⚠️  Warning: candidate_{candidate_num} SCRIBE response finish_reason={finish_reason} (not STOP)")
                    print(f"      Response length: {len(forensic_log)} chars (~{len(forensic_log)//4} tokens)")

                    # Check token usage if available
                    if hasattr(scribe_response, 'usage_metadata'):
                        usage = scribe_response.usage_metadata
                        print(f"      Token usage: input={usage.prompt_token_count}, output={usage.candidates_token_count}, total={usage.total_token_count}")
                        print(f"      Context remaining: ~{1000000 - usage.total_token_count:,} tokens")

            if len(forensic_log) < 2000:
                print(f"      ⚠️  Warning: candidate_{candidate_num} SCRIBE log suspiciously short ({len(forensic_log)} chars)")

            # Cleanup
            try:
                self.gemini_client.files.delete(name=uploaded_file.name)
            except Exception:
                pass

            return {
                'candidate_num': candidate_num,
                'forensic_log': forensic_log,
                'path': video_path
            }

        except Exception as e:
            print(f"      ❌ Error generating SCRIBE log for candidate_{candidate_num}: {e}")
            return {
                'candidate_num': candidate_num,
                'forensic_log': f"Error: {e}",
                'path': video_path
            }

    def run_parallel_stage1(self, video_paths: List[str], text_prompt: str,
                            prompt_alignment_checklist: str,
                            max_workers: int = 5) -> List[dict]:
        """
        Run Stage 1 evaluation in parallel using threading.

        Args:
            video_paths: List of video paths
            text_prompt: Generation prompt
            prompt_alignment_checklist: Pre-generated checklist
            max_workers: Maximum concurrent Gemini API calls (default 5)

        Returns:
            List of dicts for videos that passed Stage 1
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"\n{'='*80}")
        print(f"STAGE 1: VISUAL HALLUCINATION FILTER - Processing {len(video_paths)} videos")
        print(f"   🔄 Parallel processing with {max_workers} workers")
        print(f"{'='*80}")

        retained_videos = []
        rejected_videos = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all videos
            future_to_video = {
                executor.submit(self.evaluate_single_video_stage1, vp, text_prompt, prompt_alignment_checklist): vp
                for vp in video_paths
            }

            # Collect results as they complete
            for future in as_completed(future_to_video):
                result = future.result()

                if result['passed']:
                    print(f"   ✅ candidate_{result['candidate_num']}: PASSED")
                    retained_videos.append(result)
                else:
                    print(f"   ❌ candidate_{result['candidate_num']}: FAILED")
                    rejected_videos.append(result)

        print(f"\n   Stage 1 Complete: {len(retained_videos)} passed, {len(rejected_videos)} failed")
        return retained_videos, rejected_videos

    def run_parallel_stage2_scribes(self, retained_videos: List[dict], text_prompt: str,
                                     max_workers: int = 3) -> List[dict]:
        """
        Run Stage 2 SCRIBE generation in parallel using threading.

        Args:
            retained_videos: List of dicts from Stage 1 with 'path', 'candidate_num'
            text_prompt: Generation prompt
            max_workers: Maximum concurrent Gemini API calls (default 3)

        Returns:
            List of dicts with 'candidate_num', 'forensic_log', 'path'
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"\n{'='*80}")
        print(f"STAGE 2: FORENSIC OBSERVATION - Generating logs for {len(retained_videos)} videos")
        print(f"   🔄 Parallel processing with {max_workers} workers")
        print(f"{'='*80}")

        forensic_reports = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all videos
            future_to_video = {
                executor.submit(self.evaluate_single_video_stage2_scribe, v['path'], text_prompt, v['candidate_num']): v
                for v in retained_videos
            }

            # Collect results as they complete
            for future in as_completed(future_to_video):
                result = future.result()
                print(f"   ✅ candidate_{result['candidate_num']}: Forensic log generated ({len(result['forensic_log'])} chars)")
                forensic_reports.append(result)

        print(f"\n   Stage 2 Complete: Generated {len(forensic_reports)} forensic logs")
        return forensic_reports

    def run_direct_video_comparison(
        self,
        video_paths: List[str],
        text_prompt: str,
        top_k: int = 10,
        max_workers: int = 5,
        comparison_batch_size: int = 5,
    ) -> List[str]:
        """
        Tournament-style video ranking for SMC intermediate stages.

        Each round: split candidates into groups of comparison_batch_size (default 5),
        run ALL groups in parallel, each call picks exactly 1 winner.
        Repeat rounds until len(candidates) <= top_k.

        This avoids cross-batch inconsistency: every call has a single clear
        winner, and parallel execution minimises latency.

        Example (50 videos, top_k=10, batch_size=5):
          Round 1: 10 parallel calls, each picks 1 winner -> 10 winners = top_k, done.

        Args:
            video_paths:           List of video file paths to rank
            text_prompt:           The generation prompt
            top_k:                 Number of top paths to return
            max_workers:           Max concurrent Gemini API calls per round
            comparison_batch_size: Videos per single Gemini call (default 5)

        Returns:
            List of winner paths, length <= top_k (order reflects tournament wins)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import re
        from google.genai import types

        if not video_paths:
            return []

        if len(video_paths) <= top_k:
            return list(video_paths[:top_k])

        print(f"\n  [Direct Comparison] Tournament: {len(video_paths)} videos -> top {top_k} "
              f"(batch_size={comparison_batch_size}, workers={max_workers})...")

        # ── Upload all videos up-front in parallel ────────────────────────────
        def _upload_one(video_path):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    uploaded = self.gemini_client.files.upload(file=video_path)
                    return video_path, uploaded
                except Exception as upload_err:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        print(f"    Warning: Upload failed for {os.path.basename(video_path)}: {upload_err}")
                        return video_path, None

        upload_start = time.time()
        path_to_uploaded = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_upload_one, p): p for p in video_paths}
            for future in as_completed(futures):
                path, uploaded = future.result()
                if uploaded is not None:
                    path_to_uploaded[path] = uploaded

        active_files = list(path_to_uploaded.values())
        if active_files:
            self._wait_for_files_active(active_files)

        print(f"    Uploaded {len(path_to_uploaded)}/{len(video_paths)} videos in {time.time()-upload_start:.1f}s")

        available_paths = [p for p in video_paths if p in path_to_uploaded]

        if len(available_paths) <= top_k:
            for uf in path_to_uploaded.values():
                try:
                    self.gemini_client.files.delete(name=uf.name)
                except Exception:
                    pass
            return available_paths

        # Assign stable numeric IDs for Gemini to reference
        path_to_id = {p: i for i, p in enumerate(available_paths)}
        id_to_path = {i: p for p, i in path_to_id.items()}

        # ── Round function: run one batch call, return single winner path ─────
        def _compare_batch(batch: List[str], round_num: int, batch_num: int) -> Optional[str]:
            """Compare up to comparison_batch_size videos; return the best path."""
            if len(batch) == 1:
                return batch[0]

            contents = []
            for p in batch:
                cid = path_to_id[p]
                uploaded = path_to_uploaded.get(p)
                if uploaded is None:
                    continue
                contents.append(f"[Candidate {cid}]")
                contents.append(types.Part(
                    file_data=types.FileData(
                        file_uri=uploaded.uri,
                        mime_type=uploaded.mime_type
                    ),
                    video_metadata=types.VideoMetadata(fps=10)
                ))

            candidate_id_list = ", ".join(
                str(path_to_id[p]) for p in batch if p in path_to_uploaded
            )
            comparison_prompt = (
                f"# ROLE: VIDEO QUALITY JUDGE\n"
                f"You are selecting the single BEST partial video clip from a group of {len(batch)} candidates.\n"
                f"These are PARTIAL videos (not complete) - judge what is visible so far.\n\n"
                f"## PROMPT (what the video should show):\n\"{text_prompt}\"\n\n"
                f"## CANDIDATES IN THIS GROUP: [{candidate_id_list}]\n\n"
                f"## SELECTION CRITERIA (in priority order):\n"
                f"1. **Prompt Alignment**: Does visible content match the prompt?\n"
                f"2. **Physics Plausibility**: Is the motion realistic so far?\n"
                f"3. **Visual Stability**: No glitches, artifacts, or hallucinations?\n"
                f"4. **Trajectory Quality**: Does the motion look headed in a good direction?\n\n"
                f"## OUTPUT FORMAT (strict - no other text):\n"
                f"BEST_VIDEO: Candidate_[ID]\n"
                f"REASON: [One sentence explaining why this candidate leads]\n"
            )
            contents.append(comparison_prompt)

            try:
                api_start = time.time()
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        max_output_tokens=1000,
                        temperature=0.05,
                        thinking_config=types.ThinkingConfig(include_thoughts=False)
                    )
                )
                response_text = response.text
                api_time = time.time() - api_start

                print(f"      R{round_num}B{batch_num} ({len(batch)} videos, {api_time:.1f}s): "
                      f"{response_text.strip()[:150]}")

                # Parse: BEST_VIDEO: Candidate_N
                match = re.search(r'BEST_VIDEO:\s*Candidate[_\s](\d+)', response_text, re.IGNORECASE)
                if match:
                    winner_id = int(match.group(1))
                    if winner_id in id_to_path and id_to_path[winner_id] in set(batch):
                        return id_to_path[winner_id]

                print(f"      Warning: Could not parse BEST_VIDEO from R{round_num}B{batch_num}, "
                      f"falling back to first candidate")
                return batch[0]

            except Exception as api_err:
                print(f"      Error: R{round_num}B{batch_num} API call failed: {api_err}")
                return batch[0]  # Fallback: keep first candidate

        # ── Tournament rounds ─────────────────────────────────────────────────
        current_paths = available_paths[:]
        round_num = 1

        while len(current_paths) > top_k:
            batches = [
                current_paths[i:i + comparison_batch_size]
                for i in range(0, len(current_paths), comparison_batch_size)
            ]
            n_batches = len(batches)
            print(f"    Round {round_num}: {len(current_paths)} candidates -> "
                  f"{n_batches} parallel calls (batch_size={comparison_batch_size})...")

            winners = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {
                    executor.submit(_compare_batch, batch, round_num, b_idx + 1): batch
                    for b_idx, batch in enumerate(batches)
                }
                for future in as_completed(future_to_batch):
                    winner = future.result()
                    if winner is not None:
                        winners.append(winner)

            print(f"    Round {round_num} done: {len(winners)} winners from {len(current_paths)} candidates")

            if not winners:
                # Safety: keep top_k from previous round if all calls failed
                winners = current_paths[:top_k]
                break

            current_paths = winners
            round_num += 1

        # ── Cleanup all uploaded files ────────────────────────────────────────
        for uf in path_to_uploaded.values():
            try:
                self.gemini_client.files.delete(name=uf.name)
            except Exception:
                pass

        result = current_paths[:top_k]
        print(f"  [Direct Comparison] Done: {len(result)} winners after {round_num - 1} round(s)")
        return result

    def run_3stage_pipeline_parallel(self, video_paths: List[str], text_prompt: str,
                                      prompt_alignment_checklist: str = None,
                                      max_stage1_workers: int = 5,
                                      max_stage2_workers: int = 3,
                                      max_stage3_workers: int = 5,
                                      return_detailed: bool = False):
        """
        Run the complete 3-stage pipeline with parallel processing.
        NOW WITH PARALLEL STAGE 3: Process multiple tournament batches concurrently.

        Args:
            video_paths: List of video paths to evaluate
            text_prompt: Generation prompt
            prompt_alignment_checklist: Pre-generated checklist (optional)
            max_stage1_workers: Max concurrent workers for Stage 1 (default 5)
            max_stage2_workers: Max concurrent workers for Stage 2 (default 3)
            max_stage3_workers: Max concurrent workers for Stage 3 tournament (default 5)
            return_detailed: If True, return dict with timings and reports; if False, return winner path only

        Returns:
            If return_detailed=False: Path to winning video (str)
            If return_detailed=True: Dict with {winner_path, stage1_time, stage2_time, stage3_time,
                                               retained_videos, forensic_reports}
        """
        import time

        if len(video_paths) == 0:
            return None if not return_detailed else {'winner_path': None}

        if len(video_paths) == 1:
            return video_paths[0] if not return_detailed else {'winner_path': video_paths[0]}

        # Generate checklist if not provided
        if prompt_alignment_checklist is None:
            print(f"\n{'='*80}")
            print(f"STEP 0: Generating Prompt Alignment Checklist")
            print(f"{'='*80}")
            prompt_alignment_checklist = self.generate_prompt_alignment_checklist(
                text_prompt=text_prompt,
                model_name="gemini-2.5-flash",
                max_output_tokens=1500,
                temperature=0.1
            )

        # 🆕 NEW Stage 1 Filter: Fast frame-level hallucination pre-filter (OPTIONAL)
        # If enabled, this runs BEFORE the OLD Stage 1 to reject obviously bad videos early
        if self.use_stage1_model:
            print(f"\n{'='*80}")
            print(f"🆕 NEW STAGE 1 FILTER: Frame-Level Hallucination Check")
            print(f"{'='*80}")
            print(f"  Phase 1a: Random 10 frames + checklist verification")
            print(f"  Phase 1b: 12 frames (beginning→middle→end) + prompt-aware temporal consistency")
            print(f"  Model: {self.stage1_model_name}")
            print(f"  Strategy: STRICT rejection (any single frame failure = reject entire video)")

            original_video_paths = video_paths.copy()  # Keep original list for fallback
            stage1_filter_start = time.time()
            surviving_paths = self.run_stage1_filter(
                video_paths=video_paths,
                text_prompt=text_prompt,  # 🆕 Pass text_prompt for prompt-aware Stage 1b
                checklist=prompt_alignment_checklist,
                num_gpus=None,  # Auto-detect available GPUs
                workers_per_gpu=5,
                max_iterations=5,
                run_stage0=False,  # Skip Stage 0 VLM validation; use LLM spec directly
            )
            stage1_filter_time = time.time() - stage1_filter_start

            rejected_count = len(video_paths) - len(surviving_paths)
            print(f"\n✓ NEW Stage 1 Filter completed in {stage1_filter_time:.2f}s")
            print(f"  Survived: {len(surviving_paths)}/{len(video_paths)} candidates")
            print(f"  Rejected: {rejected_count} candidates")

            # Update video_paths to only include survivors
            video_paths = surviving_paths

            # Handle case where all candidates rejected
            if len(video_paths) == 0:
                print(f"\n❌ NEW Stage 1 Filter eliminated ALL videos, returning first original video")
                return original_video_paths[0] if not return_detailed else {
                    'winner_path': original_video_paths[0], 'stage1_filter_time': stage1_filter_time,
                    'stage1_time': 0, 'retained_videos': [], 'forensic_reports': []
                }

            if len(video_paths) == 1:
                print(f"\n   Only 1 video passed NEW Stage 1 Filter: {video_paths[0]}")
                return video_paths[0] if not return_detailed else {
                    'winner_path': video_paths[0], 'stage1_filter_time': stage1_filter_time,
                    'stage1_time': 0, 'retained_videos': [], 'forensic_reports': []
                }

        # OLD Stage 1: Parallel visual hallucination filter (comprehensive check on survivors)
        stage1_start = time.time()
        retained_videos, rejected_videos = self.run_parallel_stage1(
            video_paths, text_prompt, prompt_alignment_checklist, max_stage1_workers
        )
        stage1_time = time.time() - stage1_start

        if len(retained_videos) == 0:
            print(f"\n❌ Stage 1 eliminated ALL videos, returning first video")
            return video_paths[0] if not return_detailed else {
                'winner_path': video_paths[0], 'stage1_time': stage1_time,
                'retained_videos': [], 'rejected_videos': rejected_videos, 'forensic_reports': []
            }

        if len(retained_videos) == 1:
            print(f"\n   Only 1 video passed Stage 1: candidate_{retained_videos[0]['candidate_num']}")
            return retained_videos[0]['path'] if not return_detailed else {
                'winner_path': retained_videos[0]['path'], 'stage1_time': stage1_time,
                'retained_videos': retained_videos, 'rejected_videos': rejected_videos, 'forensic_reports': []
            }

        # Stage 2: Parallel SCRIBE generation
        stage2_start = time.time()
        forensic_reports = self.run_parallel_stage2_scribes(
            retained_videos, text_prompt, max_stage2_workers
        )
        stage2_time = time.time() - stage2_start

        # Stage 3: Tournament (sequential)
        print(f"\n{'='*80}")
        print(f"STAGE 3: TOURNAMENT - Comparing {len(forensic_reports)} forensic logs")
        print(f"{'='*80}")

        stage3_start = time.time()
        stage3_result = self.new_run_stage3_forensic_tournament(
            forensic_entries=forensic_reports,
            text_prompt=text_prompt,
            selection_batch_size=5,
            max_workers=max_stage3_workers
        )
        stage3_time = time.time() - stage3_start

        # Extract winner and statistics from stage3 result
        final_winner = stage3_result.get('winner') if stage3_result else None
        winner_path = final_winner['path'] if final_winner else video_paths[0]

        stage3_clean_count = stage3_result.get('clean_count', 0) if stage3_result else 0
        stage3_disqualified_count = stage3_result.get('disqualified_count', 0) if stage3_result else 0

        if return_detailed:
            return {
                'winner_path': winner_path,
                'stage1_time': stage1_time,
                'stage2_time': stage2_time,
                'stage3_time': stage3_time,
                'retained_videos': retained_videos,
                'rejected_videos': rejected_videos,
                'forensic_reports': forensic_reports,
                'stage3_clean_count': stage3_clean_count,
                'stage3_disqualified_count': stage3_disqualified_count
            }
        else:
            return winner_path

    def generate_pipeline_statistics_report(self, detailed_results: dict,
                                           total_videos: int,
                                           output_dir: str = None,
                                           prompt_idx = None):
        """
        Generate comprehensive statistics report for the 3-stage pipeline.

        Args:
            detailed_results: Dict returned from run_3stage_pipeline_parallel with return_detailed=True
            total_videos: Total number of videos that entered the pipeline
            output_dir: Directory to save statistics files (optional)
            prompt_idx: Prompt index for filename (optional, can be int or str like "0000_batch0")

        Returns:
            Dict with statistics and paths to saved files
        """
        import json
        from datetime import datetime

        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_videos': total_videos,
            'stages': {}
        }

        # ========== STAGE 1: Visual Hallucination Filter ==========
        retained_videos = detailed_results.get('retained_videos', [])
        rejected_videos = detailed_results.get('rejected_videos', [])

        stage1_passed = len(retained_videos)
        stage1_failed = len(rejected_videos)
        stage1_failure_rate = (stage1_failed / total_videos * 100) if total_videos > 0 else 0
        stage1_pass_rate = (stage1_passed / total_videos * 100) if total_videos > 0 else 0

        stats['stages']['stage1'] = {
            'name': 'Visual Hallucination Filter',
            'total_input': total_videos,
            'passed': stage1_passed,
            'failed': stage1_failed,
            'pass_rate': stage1_pass_rate,
            'failure_rate': stage1_failure_rate,
            'time_seconds': detailed_results.get('stage1_time', 0)
        }

        # ========== STAGE 2: Anomaly Detection & SCRIBE Forensic Logs ==========
        forensic_reports = detailed_results.get('forensic_reports', [])

        # Count anomalies
        anomaly_free = 0
        has_anomalies = 0
        truncated_logs = 0
        misaligned_videos = 0

        anomaly_breakdown = {
            'Background Stability': 0,
            'Friction': 0,
            'Jitter': 0,
            'Boundaries': 0,
            'Conservation': 0,
            'Multiple Anomalies': 0
        }

        import re

        for report in forensic_reports:
            forensic_log = report.get('forensic_log', '')

            # Check for anomalies using same logic as Stage 3
            # Look for PART 3 and check for "Observed" (but NOT "None observed")
            part3_match = re.search(r'## PART 3: VISUAL ANOMALY SCAN\s+(.*?)(?=##|\Z)',
                                   forensic_log, re.DOTALL | re.IGNORECASE)

            has_visual_anomaly = False
            if part3_match:
                part3_text = part3_match.group(1)
                for line in part3_text.split('\n'):
                    if not line.strip():
                        continue
                    # Check if line contains "Observed" but NOT "None observed"
                    if re.search(r'\bObserved\b', line, re.IGNORECASE) and not re.search(r'\bNone\s+observed\b', line, re.IGNORECASE):
                        has_visual_anomaly = True

                        # Count specific anomaly types
                        if 'Background Stability' in line:
                            anomaly_breakdown['Background Stability'] += 1
                        if 'Friction' in line:
                            anomaly_breakdown['Friction'] += 1
                        if 'Jitter' in line:
                            anomaly_breakdown['Jitter'] += 1
                        if 'Boundaries' in line:
                            anomaly_breakdown['Boundaries'] += 1
                        if 'Conservation' in line:
                            anomaly_breakdown['Conservation'] += 1

            if has_visual_anomaly:
                has_anomalies += 1

                # Count videos with multiple anomaly types
                if part3_match:
                    part3_text = part3_match.group(1)
                    observed_count = sum([
                        re.search(r'Background Stability.*Observed(?!\s*\()', part3_text, re.IGNORECASE) is not None,
                        re.search(r'Friction.*Observed(?!\s*\()', part3_text, re.IGNORECASE) is not None,
                        re.search(r'Jitter.*Observed(?!\s*\()', part3_text, re.IGNORECASE) is not None,
                        re.search(r'Boundaries.*Observed(?!\s*\()', part3_text, re.IGNORECASE) is not None,
                        re.search(r'Conservation.*Observed(?!\s*\()', part3_text, re.IGNORECASE) is not None
                    ])
                    if observed_count > 1:
                        anomaly_breakdown['Multiple Anomalies'] += 1
            else:
                anomaly_free += 1

            # Check for truncation - better detection based on missing expected sections
            # A complete SCRIBE report should have PART 1, PART 2, and PART 3
            has_part1 = bool(re.search(r'## PART 1:', forensic_log, re.IGNORECASE))
            has_part2 = bool(re.search(r'## PART 2:', forensic_log, re.IGNORECASE))
            has_part3 = bool(re.search(r'## PART 3:', forensic_log, re.IGNORECASE))

            # Check if log is too short (incomplete generation)
            # Complete SCRIBE reports are typically > 5000 characters
            is_too_short = len(forensic_log) <= 5000

            # Truncated if missing any section OR too short (incomplete)
            if not (has_part1 and has_part2 and has_part3) or is_too_short:
                truncated_logs += 1

            # Check for prompt misalignment
            if 'MISALIGNED' in forensic_log or 'misaligned' in forensic_log:
                misaligned_videos += 1

        stage2_anomaly_free_rate = (anomaly_free / stage1_passed * 100) if stage1_passed > 0 else 0
        stage2_anomaly_rate = (has_anomalies / stage1_passed * 100) if stage1_passed > 0 else 0
        stage2_truncation_rate = (truncated_logs / stage1_passed * 100) if stage1_passed > 0 else 0
        stage2_misalignment_rate = (misaligned_videos / stage1_passed * 100) if stage1_passed > 0 else 0

        stats['stages']['stage2'] = {
            'name': 'SCRIBE Forensic Logs + Anomaly Detection',
            'total_input': stage1_passed,
            'anomaly_free': anomaly_free,
            'has_anomalies': has_anomalies,
            'anomaly_free_rate': stage2_anomaly_free_rate,
            'anomaly_detection_rate': stage2_anomaly_rate,
            'anomaly_breakdown': anomaly_breakdown,
            'truncated_logs': truncated_logs,
            'truncation_rate': stage2_truncation_rate,
            'misaligned_videos': misaligned_videos,
            'misalignment_rate': stage2_misalignment_rate,
            'time_seconds': detailed_results.get('stage2_time', 0)
        }

        # ========== STAGE 3: Tournament ==========
        winner_path = detailed_results.get('winner_path', None)
        stage3_clean_count = detailed_results.get('stage3_clean_count', len(forensic_reports))
        stage3_disqualified_count = detailed_results.get('stage3_disqualified_count', 0)
        stage3_total_before_filtering = len(forensic_reports)

        stats['stages']['stage3'] = {
            'name': 'Tournament - Final Selection',
            'total_before_filtering': stage3_total_before_filtering,
            'clean_candidates': stage3_clean_count,
            'disqualified_anomalies': stage3_disqualified_count,
            'disqualification_rate': (stage3_disqualified_count / stage3_total_before_filtering * 100) if stage3_total_before_filtering > 0 else 0,
            'winner_found': winner_path is not None,
            'time_seconds': detailed_results.get('stage3_time', 0)
        }

        # ========== Overall Pipeline Stats ==========
        total_time = (detailed_results.get('stage1_time', 0) +
                     detailed_results.get('stage2_time', 0) +
                     detailed_results.get('stage3_time', 0))

        stats['overall'] = {
            'total_time_seconds': total_time,
            'videos_input': total_videos,
            'videos_after_stage1': stage1_passed,
            'videos_after_stage2': len(forensic_reports),
            'winner_selected': winner_path is not None,
            'overall_retention_rate': (1 / total_videos * 100) if total_videos > 0 else 0,
            'avg_time_per_video_stage1': detailed_results.get('stage1_time', 0) / total_videos if total_videos > 0 else 0,
            'avg_time_per_video_stage2': detailed_results.get('stage2_time', 0) / stage1_passed if stage1_passed > 0 else 0
        }

        # ========== Generate Human-Readable Report ==========
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("3-STAGE PIPELINE STATISTICS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {stats['timestamp']}")
        report_lines.append(f"Total Videos Input: {total_videos}")
        report_lines.append("")

        # Stage 1 Table
        report_lines.append("┌" + "─" * 78 + "┐")
        report_lines.append("│ STAGE 1: Visual Hallucination Filter                                        │")
        report_lines.append("├" + "─" * 78 + "┤")
        report_lines.append(f"│ Total Input Videos:        {stats['stages']['stage1']['total_input']:>6}                                      │")
        report_lines.append(f"│ Passed (No Hallucinations):{stats['stages']['stage1']['passed']:>6}  ({stats['stages']['stage1']['pass_rate']:>5.1f}%)                          │")
        report_lines.append(f"│ Failed (Hallucinations):   {stats['stages']['stage1']['failed']:>6}  ({stats['stages']['stage1']['failure_rate']:>5.1f}%)                          │")
        report_lines.append(f"│ Processing Time:           {stats['stages']['stage1']['time_seconds']:>6.1f}s                                    │")
        report_lines.append("└" + "─" * 78 + "┘")
        report_lines.append("")

        # Stage 2 Table
        report_lines.append("┌" + "─" * 78 + "┐")
        report_lines.append("│ STAGE 2: SCRIBE Forensic Logs + Anomaly Detection                           │")
        report_lines.append("├" + "─" * 78 + "┤")
        report_lines.append(f"│ Total Input Videos:        {stats['stages']['stage2']['total_input']:>6}                                      │")
        report_lines.append(f"│ Anomaly-Free Videos:       {stats['stages']['stage2']['anomaly_free']:>6}  ({stats['stages']['stage2']['anomaly_free_rate']:>5.1f}%)                          │")
        report_lines.append(f"│ Videos with Anomalies:     {stats['stages']['stage2']['has_anomalies']:>6}  ({stats['stages']['stage2']['anomaly_detection_rate']:>5.1f}%)                          │")
        report_lines.append("│                                                                              │")
        report_lines.append("│ Anomaly Breakdown:                                                           │")
        for anomaly_type, count in stats['stages']['stage2']['anomaly_breakdown'].items():
            if count > 0:
                report_lines.append(f"│   - {anomaly_type:<30} {count:>6}                                  │")
        report_lines.append("│                                                                              │")
        report_lines.append(f"│ Truncated Logs (max tokens):{stats['stages']['stage2']['truncated_logs']:>6}  ({stats['stages']['stage2']['truncation_rate']:>5.1f}%)                          │")
        report_lines.append(f"│ Prompt Misaligned Videos:  {stats['stages']['stage2']['misaligned_videos']:>6}  ({stats['stages']['stage2']['misalignment_rate']:>5.1f}%)                          │")
        report_lines.append(f"│ Processing Time:           {stats['stages']['stage2']['time_seconds']:>6.1f}s                                    │")
        report_lines.append("└" + "─" * 78 + "┘")
        report_lines.append("")

        # Stage 3 Table
        report_lines.append("┌" + "─" * 78 + "┐")
        report_lines.append("│ STAGE 3: Tournament - Final Selection                                       │")
        report_lines.append("├" + "─" * 78 + "┤")
        report_lines.append(f"│ Input Videos (from Stage 2):{stats['stages']['stage3']['total_before_filtering']:>6}                                      │")
        report_lines.append(f"│ Clean Candidates (no anomalies):{stats['stages']['stage3']['clean_candidates']:>6}                                      │")
        report_lines.append(f"│ Disqualified (anomalies):  {stats['stages']['stage3']['disqualified_anomalies']:>6}  ({stats['stages']['stage3']['disqualification_rate']:>5.1f}%)                          │")
        report_lines.append(f"│ Winner Selected:           {'YES' if stats['stages']['stage3']['winner_found'] else 'NO ':>6}                                      │")
        report_lines.append(f"│ Processing Time:           {stats['stages']['stage3']['time_seconds']:>6.1f}s                                    │")
        report_lines.append("└" + "─" * 78 + "┘")
        report_lines.append("")

        # Overall Summary
        report_lines.append("┌" + "─" * 78 + "┐")
        report_lines.append("│ OVERALL PIPELINE SUMMARY                                                     │")
        report_lines.append("├" + "─" * 78 + "┤")
        report_lines.append(f"│ Total Processing Time:     {stats['overall']['total_time_seconds']:>6.1f}s                                    │")
        report_lines.append(f"│ Input Videos:              {stats['overall']['videos_input']:>6}                                      │")
        report_lines.append(f"│ After Stage 1:             {stats['overall']['videos_after_stage1']:>6}                                      │")
        report_lines.append(f"│ After Stage 2:             {stats['overall']['videos_after_stage2']:>6}                                      │")
        report_lines.append(f"│ Final Winner:              {'YES' if stats['overall']['winner_selected'] else 'NO ':>6}                                      │")
        report_lines.append(f"│ Avg Time/Video (Stage 1):  {stats['overall']['avg_time_per_video_stage1']:>6.2f}s                                    │")
        report_lines.append(f"│ Avg Time/Video (Stage 2):  {stats['overall']['avg_time_per_video_stage2']:>6.2f}s                                    │")
        report_lines.append("└" + "─" * 78 + "┘")

        report_text = "\n".join(report_lines)

        # Print to console
        print("\n" + report_text)

        # ========== Save to Files ==========
        saved_files = {}

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Save JSON stats
            if prompt_idx is not None:
                # Handle both int (e.g., 0) and string (e.g., "0000_batch0") formats
                if isinstance(prompt_idx, int):
                    prompt_str = f"{prompt_idx:04d}"
                else:
                    prompt_str = str(prompt_idx)
                json_path = os.path.join(output_dir, f"prompt{prompt_str}_pipeline_stats.json")
                txt_path = os.path.join(output_dir, f"prompt{prompt_str}_pipeline_report.txt")
            else:
                json_path = os.path.join(output_dir, "pipeline_stats.json")
                txt_path = os.path.join(output_dir, "pipeline_report.txt")

            with open(json_path, 'w') as f:
                json.dump(stats, f, indent=2)
            saved_files['json'] = json_path

            # Save text report
            with open(txt_path, 'w') as f:
                f.write(report_text)
            saved_files['txt'] = txt_path

            # Save as JPG image
            if prompt_idx is not None:
                # Handle both int and string formats
                if isinstance(prompt_idx, int):
                    prompt_str = f"{prompt_idx:04d}"
                else:
                    prompt_str = str(prompt_idx)
                jpg_path = os.path.join(output_dir, f"prompt{prompt_str}_pipeline_stats.jpg")
            else:
                jpg_path = os.path.join(output_dir, "pipeline_stats.jpg")

            jpg_result = self.save_statistics_as_image(stats, jpg_path)
            if jpg_result:
                saved_files['jpg'] = jpg_path

            print(f"\n📊 Statistics saved:")
            print(f"   JSON: {json_path}")
            print(f"   Report: {txt_path}")
            if jpg_result:
                print(f"   Image: {jpg_path}")

        return {
            'statistics': stats,
            'report_text': report_text,
            'saved_files': saved_files
        }

    def save_statistics_as_image(self, stats: dict, output_path: str):
        """
        Save pipeline statistics as a clean, formatted JPG image.

        Args:
            stats: Statistics dict from generate_pipeline_statistics_report
            output_path: Path to save the JPG file
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            print("Warning: PIL (Pillow) not installed. Cannot generate image. Install with: pip install Pillow")
            return None

        # Image dimensions
        width = 1200
        header_height = 120
        stage_height = 280  # Height for Stage 1 and Stage 2 tables
        stage3_height = 180  # Height for Stage 3 table
        overall_height = 240  # Height for overall summary
        margin = 40
        total_height = header_height + (2 * stage_height) + stage3_height + overall_height + (5 * margin)

        # Create image with white background
        img = Image.new('RGB', (width, total_height), color='white')
        draw = ImageDraw.Draw(img)

        # Try to use a nice font, fallback to default if not available
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
            header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            # Fallback to default font
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # Simple, professional colors - like a clean spreadsheet
        header_bg = (70, 100, 130)  # Dark blue-gray for section headers
        header_text_color = (255, 255, 255)  # White text for headers
        row_bg_light = (245, 245, 245)  # Light gray for odd rows
        row_bg_dark = (210, 210, 210)  # Darker gray for even rows
        text_color = (40, 40, 40)  # Dark gray text
        border_color = (150, 150, 150)  # Medium gray borders

        y_offset = margin

        # ===== TITLE =====
        draw.rectangle([(margin, y_offset), (width - margin, y_offset + 60)],
                      fill=header_bg, outline=border_color, width=2)
        draw.text((width // 2, y_offset + 30), "3-STAGE PIPELINE STATISTICS REPORT",
                  fill=text_color, font=title_font, anchor='mm')

        y_offset += 60 + 10
        timestamp = stats.get('timestamp', 'N/A')
        draw.text((width // 2, y_offset), f"Generated: {timestamp}",
                  fill=text_color, font=small_font, anchor='mm')
        draw.text((width // 2, y_offset + 20), f"Total Videos Input: {stats['total_videos']}",
                  fill=text_color, font=small_font, anchor='mm')

        y_offset += 50 + margin

        # Helper function to draw a stage table
        def draw_stage_table(y_start, stage_name, stage_data, table_height):
            # Header - simple gray background with dark text
            draw.rectangle([(margin, y_start), (width - margin, y_start + 40)],
                          fill=header_bg, outline=border_color, width=2)
            draw.text((width // 2, y_start + 20), stage_name,
                      fill=header_text_color, font=header_font, anchor='mm')

            # Table border
            table_top = y_start + 40
            draw.rectangle([(margin, table_top), (width - margin, y_start + table_height)],
                          outline=border_color, width=2)

            # Content area
            content_y = table_top + 15
            line_height = 22
            row_num = 0  # Track row number for alternating backgrounds

            if 'stage1' in stage_data:
                # Stage 1 format
                data = stage_data['stage1']

                # Row: Total Input
                row_bg = row_bg_light if row_num % 2 == 0 else row_bg_dark
                draw.rectangle([(margin + 1, content_y - 3), (width - margin - 1, content_y + line_height - 3)],
                              fill=row_bg)
                draw.text((margin + 20, content_y), f"Total Input Videos:",
                          fill=text_color, font=body_font)
                draw.text((width - margin - 20, content_y), f"{data['total_input']}",
                          fill=text_color, font=body_font, anchor='rm')
                content_y += line_height
                row_num += 1

                # Stage 1 specific
                row_bg = row_bg_light if row_num % 2 == 0 else row_bg_dark
                draw.rectangle([(margin + 1, content_y - 3), (width - margin - 1, content_y + line_height - 3)],
                              fill=row_bg)
                draw.text((margin + 20, content_y), f"Passed (No Hallucinations):",
                          fill=text_color, font=body_font)
                draw.text((width - margin - 20, content_y),
                          f"{data['passed']}  ({data['pass_rate']:.1f}%)",
                          fill=text_color, font=body_font, anchor='rm')
                content_y += line_height
                row_num += 1

                row_bg = row_bg_light if row_num % 2 == 0 else row_bg_dark
                draw.rectangle([(margin + 1, content_y - 3), (width - margin - 1, content_y + line_height - 3)],
                              fill=row_bg)
                draw.text((margin + 20, content_y), f"Failed (Hallucinations):",
                          fill=text_color, font=body_font)
                draw.text((width - margin - 20, content_y),
                          f"{data['failed']}  ({data['failure_rate']:.1f}%)",
                          fill=text_color, font=body_font, anchor='rm')
                content_y += line_height
                row_num += 1

                # Processing time
                row_bg = row_bg_light if row_num % 2 == 0 else row_bg_dark
                draw.rectangle([(margin + 1, content_y - 3), (width - margin - 1, content_y + line_height - 3)],
                              fill=row_bg)
                draw.text((margin + 20, content_y), f"Processing Time:",
                          fill=text_color, font=body_font)
                draw.text((width - margin - 20, content_y), f"{data['time_seconds']:.1f}s",
                          fill=text_color, font=body_font, anchor='rm')

            elif 'stage3' in stage_data:
                # Stage 3 specific format
                data = stage_data['stage3']

                row_bg = row_bg_light if row_num % 2 == 0 else row_bg_dark
                draw.rectangle([(margin + 1, content_y - 3), (width - margin - 1, content_y + line_height - 3)],
                              fill=row_bg)
                draw.text((margin + 20, content_y), f"Input Videos (from Stage 2):",
                          fill=text_color, font=body_font)
                draw.text((width - margin - 20, content_y), f"{data['total_before_filtering']}",
                          fill=text_color, font=body_font, anchor='rm')
                content_y += line_height
                row_num += 1

                row_bg = row_bg_light if row_num % 2 == 0 else row_bg_dark
                draw.rectangle([(margin + 1, content_y - 3), (width - margin - 1, content_y + line_height - 3)],
                              fill=row_bg)
                draw.text((margin + 20, content_y), f"Clean Candidates:",
                          fill=text_color, font=body_font)
                draw.text((width - margin - 20, content_y), f"{data['clean_candidates']}",
                          fill=text_color, font=body_font, anchor='rm')
                content_y += line_height
                row_num += 1

                row_bg = row_bg_light if row_num % 2 == 0 else row_bg_dark
                draw.rectangle([(margin + 1, content_y - 3), (width - margin - 1, content_y + line_height - 3)],
                              fill=row_bg)
                draw.text((margin + 20, content_y), f"Disqualified (anomalies):",
                          fill=text_color, font=body_font)
                draw.text((width - margin - 20, content_y),
                          f"{data['disqualified_anomalies']}  ({data['disqualification_rate']:.1f}%)",
                          fill=text_color, font=body_font, anchor='rm')
                content_y += line_height
                row_num += 1

                row_bg = row_bg_light if row_num % 2 == 0 else row_bg_dark
                draw.rectangle([(margin + 1, content_y - 3), (width - margin - 1, content_y + line_height - 3)],
                              fill=row_bg)
                winner_text = "YES" if data['winner_found'] else "NO"
                draw.text((margin + 20, content_y), f"Winner Selected:",
                          fill=text_color, font=body_font)
                draw.text((width - margin - 20, content_y), winner_text,
                          fill=text_color, font=body_font, anchor='rm')
                content_y += line_height
                row_num += 1

                # Processing time
                row_bg = row_bg_light if row_num % 2 == 0 else row_bg_dark
                draw.rectangle([(margin + 1, content_y - 3), (width - margin - 1, content_y + line_height - 3)],
                              fill=row_bg)
                draw.text((margin + 20, content_y), f"Processing Time:",
                          fill=text_color, font=body_font)
                draw.text((width - margin - 20, content_y), f"{data['time_seconds']:.1f}s",
                          fill=text_color, font=body_font, anchor='rm')

            elif 'stage2' in stage_data:
                # Stage 2 format (more detailed)
                data = stage_data['stage2']

                row_bg = row_bg_light if row_num % 2 == 0 else row_bg_dark
                draw.rectangle([(margin + 1, content_y - 3), (width - margin - 1, content_y + line_height - 3)],
                              fill=row_bg)
                draw.text((margin + 20, content_y), f"Total Input Videos:",
                          fill=text_color, font=body_font)
                draw.text((width - margin - 20, content_y), f"{data['total_input']}",
                          fill=text_color, font=body_font, anchor='rm')
                content_y += line_height
                row_num += 1

                row_bg = row_bg_light if row_num % 2 == 0 else row_bg_dark
                draw.rectangle([(margin + 1, content_y - 3), (width - margin - 1, content_y + line_height - 3)],
                              fill=row_bg)
                draw.text((margin + 20, content_y), f"Anomaly-Free Videos:",
                          fill=text_color, font=body_font)
                draw.text((width - margin - 20, content_y),
                          f"{data['anomaly_free']}  ({data['anomaly_free_rate']:.1f}%)",
                          fill=text_color, font=body_font, anchor='rm')
                content_y += line_height
                row_num += 1

                row_bg = row_bg_light if row_num % 2 == 0 else row_bg_dark
                draw.rectangle([(margin + 1, content_y - 3), (width - margin - 1, content_y + line_height - 3)],
                              fill=row_bg)
                draw.text((margin + 20, content_y), f"Videos with Anomalies:",
                          fill=text_color, font=body_font)
                draw.text((width - margin - 20, content_y),
                          f"{data['has_anomalies']}  ({data['anomaly_detection_rate']:.1f}%)",
                          fill=text_color, font=body_font, anchor='rm')
                content_y += line_height
                row_num += 1

                # Anomaly breakdown (show top 3)
                anomaly_breakdown = data['anomaly_breakdown']
                sorted_anomalies = sorted(anomaly_breakdown.items(), key=lambda x: x[1], reverse=True)
                top_anomalies = [a for a in sorted_anomalies if a[1] > 0][:3]

                if top_anomalies:
                    # Calculate height needed for anomaly section
                    anomaly_section_height = 20 + (len(top_anomalies) * 18)
                    row_bg = row_bg_light if row_num % 2 == 0 else row_bg_dark
                    draw.rectangle([(margin + 1, content_y - 3), (width - margin - 1, content_y + anomaly_section_height - 3)],
                                  fill=row_bg)
                    draw.text((margin + 20, content_y), f"Top Anomalies:",
                              fill=text_color, font=small_font)
                    content_y += 20
                    for anomaly_name, count in top_anomalies:
                        draw.text((margin + 40, content_y), f"• {anomaly_name}: {count}",
                                  fill=text_color, font=small_font)
                        content_y += 18
                    row_num += 1

                content_y += 4
                row_bg = row_bg_light if row_num % 2 == 0 else row_bg_dark
                draw.rectangle([(margin + 1, content_y - 3), (width - margin - 1, content_y + line_height - 3)],
                              fill=row_bg)
                draw.text((margin + 20, content_y), f"Truncated Logs (max tokens):",
                          fill=text_color, font=body_font)
                draw.text((width - margin - 20, content_y),
                          f"{data['truncated_logs']}  ({data['truncation_rate']:.1f}%)",
                          fill=text_color, font=body_font, anchor='rm')
                content_y += line_height
                row_num += 1

                row_bg = row_bg_light if row_num % 2 == 0 else row_bg_dark
                draw.rectangle([(margin + 1, content_y - 3), (width - margin - 1, content_y + line_height - 3)],
                              fill=row_bg)
                draw.text((margin + 20, content_y), f"Prompt Misaligned Videos:",
                          fill=text_color, font=body_font)
                draw.text((width - margin - 20, content_y),
                          f"{data['misaligned_videos']}  ({data['misalignment_rate']:.1f}%)",
                          fill=text_color, font=body_font, anchor='rm')
                content_y += line_height
                row_num += 1

                row_bg = row_bg_light if row_num % 2 == 0 else row_bg_dark
                draw.rectangle([(margin + 1, content_y - 3), (width - margin - 1, content_y + line_height - 3)],
                              fill=row_bg)
                draw.text((margin + 20, content_y), f"Processing Time:",
                          fill=text_color, font=body_font)
                draw.text((width - margin - 20, content_y), f"{data['time_seconds']:.1f}s",
                          fill=text_color, font=body_font, anchor='rm')

        # ===== STAGE 1 TABLE =====
        draw_stage_table(y_offset, "STAGE 1: Visual Hallucination Filter",
                        {'stage1': stats['stages']['stage1']}, stage_height)
        y_offset += stage_height + margin

        # ===== STAGE 2 TABLE =====
        draw_stage_table(y_offset, "STAGE 2: SCRIBE Forensic Logs + Anomaly Detection",
                        {'stage2': stats['stages']['stage2']}, stage_height)
        y_offset += stage_height + margin

        # ===== STAGE 3 TABLE =====
        draw_stage_table(y_offset, "STAGE 3: Tournament - Final Selection",
                        {'stage3': stats['stages']['stage3']}, 180)
        y_offset += 180 + margin

        # ===== OVERALL SUMMARY =====
        draw.rectangle([(margin, y_offset), (width - margin, y_offset + 40)],
                      fill=header_bg, outline=border_color, width=2)
        draw.text((width // 2, y_offset + 20), "OVERALL PIPELINE SUMMARY",
                  fill=header_text_color, font=header_font, anchor='mm')

        table_top = y_offset + 40
        draw.rectangle([(margin, table_top), (width - margin, y_offset + overall_height)],
                      outline=border_color, width=2)

        content_y = table_top + 15
        line_height = 22
        overall = stats['overall']

        draw.text((margin + 20, content_y), f"Total Processing Time:",
                  fill=text_color, font=body_font)
        draw.text((width - margin - 20, content_y), f"{overall['total_time_seconds']:.1f}s",
                  fill=text_color, font=body_font, anchor='rm')
        content_y += line_height

        draw.text((margin + 20, content_y), f"Videos After Stage 1:",
                  fill=text_color, font=body_font)
        draw.text((width - margin - 20, content_y), f"{overall['videos_after_stage1']}",
                  fill=text_color, font=body_font, anchor='rm')
        content_y += line_height

        draw.text((margin + 20, content_y), f"Videos After Stage 2:",
                  fill=text_color, font=body_font)
        draw.text((width - margin - 20, content_y), f"{overall['videos_after_stage2']}",
                  fill=text_color, font=body_font, anchor='rm')
        content_y += line_height

        winner_text = "YES" if overall['winner_selected'] else "NO"
        draw.text((margin + 20, content_y), f"Final Winner Selected:",
                  fill=text_color, font=body_font)
        draw.text((width - margin - 20, content_y), winner_text,
                  fill=text_color, font=body_font, anchor='rm')
        content_y += line_height

        draw.text((margin + 20, content_y), f"Avg Time/Video (Stage 1):",
                  fill=text_color, font=body_font)
        draw.text((width - margin - 20, content_y), f"{overall['avg_time_per_video_stage1']:.2f}s",
                  fill=text_color, font=body_font, anchor='rm')
        content_y += line_height

        draw.text((margin + 20, content_y), f"Avg Time/Video (Stage 2):",
                  fill=text_color, font=body_font)
        draw.text((width - margin - 20, content_y), f"{overall['avg_time_per_video_stage2']:.2f}s",
                  fill=text_color, font=body_font, anchor='rm')

        # Save as JPG
        img.save(output_path, 'JPEG', quality=95)
        print(f"📊 Statistics image saved: {output_path}")
        return output_path

    def _parse_gemini_winner(self, response_text: str, batch_files: List[str]) -> str:
        """
        Parse Gemini's response to extract the winning video path.

        Looks for pattern: "FINAL CHOICE: Video_candidate_X (File: filename.mp4)"

        Args:
            response_text: Gemini's response text
            batch_files: List of video paths in the batch

        Returns:
            Path to the winning video, or None if parsing failed
        """
        import re

        # Strategy 1: Try to find "BEST_VIDEO: Video_candidate_X (File: filename.mp4)"
        pattern = r'BEST_VIDEO:\s*Video_candidate[_\s]*(\d+)\s*\(File:\s*([^\)]+)\)'
        match = re.search(pattern, response_text, re.IGNORECASE)

        if match:
            candidate_num = int(match.group(1))
            filename = match.group(2).strip()

            # Find the batch_file that contains "candidate_{candidate_num}"
            for batch_file in batch_files:
                if f'candidate_{candidate_num}' in batch_file:
                    print(f"      ✓ Parsed winner from BEST_VIDEO format: Video_candidate_{candidate_num}")
                    return batch_file

            print(f"      Warning: Could not find file with candidate_{candidate_num}")

        # Strategy 2: Try to find "BEST_VIDEO: Candidate_X" (without "Video_" prefix)
        pattern2 = r'BEST_VIDEO:\s*Candidate[_\s]*(\d+)'
        match2 = re.search(pattern2, response_text, re.IGNORECASE)
        if match2:
            candidate_num = int(match2.group(1))
            for batch_file in batch_files:
                if f'candidate_{candidate_num}' in batch_file:
                    print(f"      ✓ Parsed winner from BEST_VIDEO (Candidate_X format): Candidate_{candidate_num}")
                    return batch_file

        # Strategy 3: Try to find just "Video_candidate_X" without File part
        pattern3 = r'BEST_VIDEO:\s*Video_candidate[_\s]*(\d+)'
        match3 = re.search(pattern3, response_text, re.IGNORECASE)
        if match3:
            candidate_num = int(match3.group(1))
            for batch_file in batch_files:
                if f'candidate_{candidate_num}' in batch_file:
                    print(f"      ✓ Parsed winner from simplified BEST_VIDEO: Video_candidate_{candidate_num}")
                    return batch_file

        # Strategy 4: Look for candidate number mentioned in decision section
        last_paragraph = response_text.split('\n\n')[-1] if '\n\n' in response_text else response_text[-500:]
        winner_patterns = [
            r'(?:choose|select|pick|winner is|best is|winner:)\s*Video_candidate[_\s]*(\d+)',
            r'Video_candidate[_\s]*(\d+)\s+(?:is the|wins|is best|has the least)',
        ]
        for pattern in winner_patterns:
            match = re.search(pattern, last_paragraph, re.IGNORECASE)
            if match:
                candidate_num = int(match.group(1))
                for batch_file in batch_files:
                    if f'candidate_{candidate_num}' in batch_file:
                        print(f"      ✓ Inferred winner from context: Video_candidate_{candidate_num}")
                        return batch_file

        # Strategy 5: Fallback - find last mentioned "Video_candidate_X"
        pattern_all = r'Video_candidate[_\s]*(\d+)'
        matches = re.findall(pattern_all, response_text)
        if matches:
            # Take the last mention as the final choice
            candidate_num = int(matches[-1])
            for batch_file in batch_files:
                if f'candidate_{candidate_num}' in batch_file:
                    print(f"      ⚠️  Using last mentioned video as fallback: Video_candidate_{candidate_num}")
                    return batch_file

        print(f"      ❌ Could not parse any valid winner from response")
        return None

    def score_complete_videos(self, video_paths, text_prompt):
        """Score complete videos using VideoLLaMA3 or LLaVA-Video with detailed multi-aspect evaluation

        Args:
            video_paths: List of video file paths (e.g., [path1.mp4, path2.mp4])
            text_prompt: Text prompt for the video

        Returns:
            Tensor of scores [num_candidates] in range [0, 1]
        """
        return self.vlm_scorer.score_complete_videos(video_paths, text_prompt)

    def _calculate_temporal_consistency(self, current_frames, previous_frames):
        """Calculate temporal consistency between current and previous frames in latent space"""
        if previous_frames is None:
            return 1.0  # First frame block has perfect consistency by definition

        with torch.no_grad():
            # Work directly in latent space - no VAE decoding needed!
            # Calculate consistency between last frame of previous block and first frame of current block
            last_prev_frame = previous_frames[:, -1]  # [B, C, H, W] in latent space
            first_curr_frame = current_frames[:, 0]   # [B, C, H, W] in latent space

            # Latent space consistency using cosine similarity + MSE
            # Cosine similarity captures semantic consistency
            cos_sim = F.cosine_similarity(
                last_prev_frame.flatten(1),
                first_curr_frame.flatten(1),
                dim=1
            ).mean().item()

            # MSE captures structural consistency in latent space
            mse = F.mse_loss(last_prev_frame, first_curr_frame).item()

            # Combine both metrics (cosine similarity is more important for semantics)
            consistency = 0.7 * ((cos_sim + 1) / 2) + 0.3 * torch.exp(torch.tensor(-mse * 5)).item()

            return max(0.0, min(1.0, consistency))

    def _calculate_motion_smoothness(self, generated_frames):
        """Calculate motion smoothness in latent space"""
        if generated_frames.shape[1] < 2:
            return 1.0  # Single frame is perfectly smooth

        with torch.no_grad():
            smoothness_scores = []

            for t in range(generated_frames.shape[1] - 1):
                current_frame = generated_frames[:, t]    # [B, C, H, W]
                next_frame = generated_frames[:, t + 1]   # [B, C, H, W]

                # Motion vector estimation in latent space
                # Calculate frame difference (motion proxy)
                frame_diff = next_frame - current_frame  # [B, C, H, W]

                # Variance-based smoothness (lower variance = smoother motion)
                motion_variance = torch.var(frame_diff.flatten(1), dim=1).mean().item()
                variance_smoothness = torch.exp(torch.tensor(-motion_variance * 10)).item()

                # Semantic smoothness using cosine similarity
                cos_sim = F.cosine_similarity(
                    current_frame.flatten(1),
                    next_frame.flatten(1),
                    dim=1
                ).mean().item()
                semantic_smoothness = (cos_sim + 1) / 2  # Normalize to [0,1]

                # Combine both aspects
                smoothness = 0.6 * variance_smoothness + 0.4 * semantic_smoothness
                smoothness_scores.append(max(0.0, min(1.0, smoothness)))

            return sum(smoothness_scores) / len(smoothness_scores) if smoothness_scores else 1.0

    def score(self, generated_frames, text_prompt, previous_frames, frame_position=None, scoring_config=None):
        """Calculate multi-objective score with VLM, smoothness, and temporal consistency
        Returns batch of scores if generated_frames has batch dimension > 1"""
        # Note: frame_position and scoring_config are currently unused but kept for API compatibility

        # 1. VLM quality and relevance scoring - returns tensor of scores for batch
        vlm_scores = self.vlm_scorer.score_frames(generated_frames, previous_frames, text_prompt)

        # Skip temporal consistency and smoothness calculations for now
        # consistency_score = self._calculate_temporal_consistency(generated_frames, previous_frames)
        # smoothness_score = self._calculate_motion_smoothness(generated_frames)

        # If batch scoring, return tensor of scores
        if isinstance(vlm_scores, torch.Tensor) and vlm_scores.numel() > 1:
            # Return batch of scores
            return vlm_scores, {
                'vlm': vlm_scores,
                'total': vlm_scores
            }
        else:
            # Single score (backward compatibility)
            vlm_score = vlm_scores.item() if isinstance(vlm_scores, torch.Tensor) else vlm_scores
            return vlm_score, {
                'vlm': vlm_score,
                'total': vlm_score
            }

    # ========================================================================
    # SIMPLIFIED 2-STAGE PIPELINE (Alternative to 3-Stage)
    # ========================================================================

    def evaluate_video_prompt_and_physics(self, video_path: str, text_prompt: str,
                                         model_name: str = "gemini-2.5-flash",
                                         max_output_tokens: int = 1500,
                                         temperature: float = 0.2) -> dict:
        """
        Simplified Stage 1: Single evaluation asking:
        1. Does the video match the prompt?
        2. Does it follow physics laws?

        Args:
            video_path: Path to video file
            text_prompt: The generation prompt
            model_name: Gemini model to use
            max_output_tokens: Max tokens in response
            temperature: Sampling temperature

        Returns:
            Dict with:
                - candidate_num: Video identifier
                - prompt_match: YES/NO
                - prompt_reasoning: Why it matches/doesn't match
                - physics_correct: YES/NO
                - physics_reasoning: Physics analysis
                - overall_quality: Summary assessment
                - path: Video path
        """
        if not self.use_gemini or self.gemini_client is None:
            raise RuntimeError("Gemini is not initialized")

        # Extract candidate number from filename
        candidate_num = self._extract_candidate_number(video_path)

        # Upload video
        print(f"   📤 Uploading candidate_{candidate_num}...")
        video_file = self.gemini_client.files.upload(path=video_path)

        # Wait for processing
        max_wait = 300
        wait_time = 0
        while video_file.state == "PROCESSING" and wait_time < max_wait:
            time.sleep(2)
            wait_time += 2
            video_file = self.gemini_client.files.get(name=video_file.name)

        if video_file.state != "ACTIVE":
            raise RuntimeError(f"Video processing failed: {video_file.state}")

        # Create evaluation prompt
        evaluation_prompt = f"""You are a physics-aware video quality evaluator. Analyze this video carefully.

**GENERATION PROMPT:**
"{text_prompt}"

**YOUR TASK:**
Evaluate this video on TWO criteria:

## CRITERION 1: Prompt Alignment
Does the video accurately match the generation prompt?
- Check if all elements mentioned in the prompt are present
- Check if the action/scene matches what was requested
- Check if visual details align with the description

**Output Format:**
PROMPT_MATCH: [YES/NO]
PROMPT_REASONING: [1-2 sentences explaining why it does/doesn't match]

## CRITERION 2: Physics Correctness
Does the video follow real-world physics laws?
- Gravity (objects fall downward, no floating without support)
- Conservation (mass/energy don't appear or disappear)
- Continuity (smooth motion, no teleportation or frame skips)
- Material behavior (appropriate deformation, friction, collisions)
- Temporal consistency (logical progression from start to finish)

**Output Format:**
PHYSICS_CORRECT: [YES/NO]
PHYSICS_REASONING: [2-3 sentences describing physics quality - what's correct and what's wrong if applicable]

## OVERALL ASSESSMENT
OVERALL_QUALITY: [EXCELLENT / GOOD / FAIR / POOR]
- EXCELLENT: Perfect prompt match + perfect physics
- GOOD: Good prompt match + good physics (minor issues okay)
- FAIR: Acceptable prompt match OR physics but not both
- POOR: Fails prompt match or has major physics violations

**IMPORTANT:**
- Be objective and critical
- Don't excuse physics violations
- Base assessment on what you SEE in the video, not assumptions
- A video can match the prompt but still have bad physics (or vice versa)

Now analyze the video and provide your evaluation:"""

        try:
            response = self.gemini_client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type),
                    evaluation_prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            )

            response_text = response.text

            # Parse response
            result = {
                'candidate_num': candidate_num,
                'path': video_path,
                'prompt_match': 'UNKNOWN',
                'prompt_reasoning': '',
                'physics_correct': 'UNKNOWN',
                'physics_reasoning': '',
                'overall_quality': 'UNKNOWN',
                'raw_response': response_text
            }

            # Extract fields
            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('PROMPT_MATCH:'):
                    result['prompt_match'] = line.split(':', 1)[1].strip()
                elif line.startswith('PROMPT_REASONING:'):
                    result['prompt_reasoning'] = line.split(':', 1)[1].strip()
                elif line.startswith('PHYSICS_CORRECT:'):
                    result['physics_correct'] = line.split(':', 1)[1].strip()
                elif line.startswith('PHYSICS_REASONING:'):
                    result['physics_reasoning'] = line.split(':', 1)[1].strip()
                elif line.startswith('OVERALL_QUALITY:'):
                    result['overall_quality'] = line.split(':', 1)[1].strip()

            # Cleanup
            self.gemini_client.files.delete(name=video_file.name)

            return result

        except Exception as e:
            print(f"   ❌ Error evaluating candidate_{candidate_num}: {e}")
            # Cleanup on error
            try:
                self.gemini_client.files.delete(name=video_file.name)
            except:
                pass

            return {
                'candidate_num': candidate_num,
                'path': video_path,
                'prompt_match': 'ERROR',
                'prompt_reasoning': f'Evaluation failed: {str(e)}',
                'physics_correct': 'ERROR',
                'physics_reasoning': f'Evaluation failed: {str(e)}',
                'overall_quality': 'ERROR',
                'raw_response': ''
            }

    def run_2stage_simplified_pipeline(self, video_paths: List[str], text_prompt: str,
                                       max_stage1_workers: int = 5,
                                       return_detailed: bool = False):
        """
        Simplified 2-stage pipeline:
        Stage 1: Evaluate all videos for prompt match + physics correctness (parallel)
        Stage 2: Tournament selection based on evaluations

        Args:
            video_paths: List of video paths
            text_prompt: Generation prompt
            max_stage1_workers: Max concurrent evaluations (default 5)
            return_detailed: If True, return detailed dict; if False, return winner path only

        Returns:
            If return_detailed=False: Path to winning video (str)
            If return_detailed=True: Dict with {winner_path, evaluations, stage1_time, stage2_time}
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if len(video_paths) == 0:
            return None if not return_detailed else {'winner_path': None}

        if len(video_paths) == 1:
            return video_paths[0] if not return_detailed else {'winner_path': video_paths[0]}

        # ========== STAGE 1: Evaluate All Videos (Parallel) ==========
        print(f"\n{'='*80}")
        print(f"STAGE 1: PROMPT & PHYSICS EVALUATION - Processing {len(video_paths)} videos")
        print(f"   🔄 Parallel processing with {max_stage1_workers} workers")
        print(f"{'='*80}")

        stage1_start = time.time()
        evaluations = []

        with ThreadPoolExecutor(max_workers=max_stage1_workers) as executor:
            # Submit all videos
            future_to_video = {
                executor.submit(self.evaluate_video_prompt_and_physics, vp, text_prompt): vp
                for vp in video_paths
            }

            # Collect results as they complete
            for future in as_completed(future_to_video):
                result = future.result()
                evaluations.append(result)

                # Print summary
                cand_num = result['candidate_num']
                prompt_match = result['prompt_match']
                physics_ok = result['physics_correct']
                quality = result['overall_quality']

                status_icon = "✅" if quality in ["EXCELLENT", "GOOD"] else "⚠️" if quality == "FAIR" else "❌"
                print(f"   {status_icon} candidate_{cand_num}: Prompt={prompt_match}, Physics={physics_ok}, Quality={quality}")

        stage1_time = time.time() - stage1_start
        print(f"\n   Stage 1 Complete: Evaluated {len(evaluations)} videos in {stage1_time:.1f}s")

        # ========== STAGE 2: Tournament Selection ==========
        print(f"\n{'='*80}")
        print(f"STAGE 2: TOURNAMENT SELECTION - Comparing {len(evaluations)} evaluations")
        print(f"{'='*80}")

        stage2_start = time.time()

        # Prepare evaluation summaries for tournament
        eval_summaries = []
        for eval_data in evaluations:
            summary = f"""candidate_{eval_data['candidate_num']}:
Prompt Match: {eval_data['prompt_match']} - {eval_data['prompt_reasoning']}
Physics Correct: {eval_data['physics_correct']} - {eval_data['physics_reasoning']}
Overall Quality: {eval_data['overall_quality']}"""
            eval_summaries.append(summary)

        # Create tournament prompt
        tournament_prompt = f"""You are selecting the BEST video from multiple candidates.

**GENERATION PROMPT:**
"{text_prompt}"

**CANDIDATE EVALUATIONS:**

{chr(10).join(eval_summaries)}

**YOUR TASK:**
Select the candidate with the BEST overall quality considering:
1. **Prompt alignment** (does it match what was requested?)
2. **Physics correctness** (does it follow real-world physics?)

**SELECTION CRITERIA:**
- Prioritize candidates with PROMPT_MATCH=YES and PHYSICS_CORRECT=YES
- If multiple candidates are good, choose the one with the best physics reasoning
- If no candidates are perfect, choose the least-bad option

**OUTPUT FORMAT (REQUIRED):**
WINNER: candidate_X
REASON: [1-2 sentences explaining why this candidate won]

Provide your selection:"""

        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=tournament_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=500,
                )
            )

            response_text = response.text

            # Parse winner
            winner_num = None
            winner_reason = ""

            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('WINNER:'):
                    winner_str = line.split(':', 1)[1].strip()
                    # Extract number from "candidate_X"
                    import re
                    match = re.search(r'candidate[_\s]*(\d+)', winner_str, re.IGNORECASE)
                    if match:
                        winner_num = int(match.group(1))
                elif line.startswith('REASON:'):
                    winner_reason = line.split(':', 1)[1].strip()

            # Find winner path
            winner_path = None
            if winner_num is not None:
                for eval_data in evaluations:
                    if eval_data['candidate_num'] == winner_num:
                        winner_path = eval_data['path']
                        break

            # Fallback: pick first EXCELLENT, then GOOD, then first video
            if winner_path is None:
                print(f"   ⚠️  Could not parse winner, using quality-based fallback...")
                for quality in ["EXCELLENT", "GOOD", "FAIR", "POOR"]:
                    for eval_data in evaluations:
                        if eval_data['overall_quality'] == quality:
                            winner_path = eval_data['path']
                            winner_num = eval_data['candidate_num']
                            break
                    if winner_path:
                        break

                # Last resort
                if winner_path is None:
                    winner_path = evaluations[0]['path']
                    winner_num = evaluations[0]['candidate_num']

            stage2_time = time.time() - stage2_start

            print(f"\n   🏆 WINNER: candidate_{winner_num}")
            print(f"   Reason: {winner_reason}")
            print(f"   Stage 2 Complete in {stage2_time:.1f}s")

        except Exception as e:
            print(f"   ❌ Tournament failed: {e}")
            print(f"   Defaulting to first candidate")
            winner_path = video_paths[0]
            stage2_time = time.time() - stage2_start

        # Return results
        if return_detailed:
            return {
                'winner_path': winner_path,
                'evaluations': evaluations,
                'stage1_time': stage1_time,
                'stage2_time': stage2_time,
                'total_time': stage1_time + stage2_time
            }
        else:
            return winner_path

    def multi_stage_candidate_selection(self, candidate_paths: List[str], text_prompt: str,
                                       prompt_alignment_checklist: str,
                                       use_stage0: bool = False,
                                       stage0_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
                                       stage0_num_gpus: int = None,
                                       stage0_workers_per_gpu: int = 5,
                                       use_stage1_filter: bool = False,
                                       stage1_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
                                       stage1_num_gpus: int = None,
                                       stage1_workers_per_gpu: int = 5,
                                       stage1_max_iterations: int = 5) -> dict:
        """
        Multi-Stage Candidate Selection Pipeline

        Orchestrates the complete 3-stage evaluation pipeline:
          Stage 0: First Frame Validation & Grounding (optional)
          Stage 1: Visual Hallucination Filter (optional)
          Stage 2: Forensic Frame-by-Frame Observation
          Stage 3: Tournament Selection

        Args:
            candidate_paths: List of video file paths to evaluate
            text_prompt: Original text prompt
            prompt_alignment_checklist: JSON checklist from generate_prompt_alignment_checklist()
            use_stage0: Enable Stage 0 first frame validation
            stage0_model_name: Model for Stage 0
            stage0_num_gpus: Number of GPUs for Stage 0
            stage0_workers_per_gpu: Workers per GPU for Stage 0
            use_stage1_filter: Enable Stage 1 hallucination filter
            stage1_model_name: Model for Stage 1
            stage1_num_gpus: Number of GPUs for Stage 1
            stage1_workers_per_gpu: Workers per GPU for Stage 1
            stage1_max_iterations: Max iterations for Stage 1

        Returns:
            dict with keys:
              - winner_index: Index of winning video in candidate_paths
              - winner_path: Path to winning video
              - stats: Detailed statistics for each stage
        """
        import time

        total_start_time = time.time()
        stats = {
            'total_candidates': len(candidate_paths),
            'stages': {},
            'total_time': 0
        }

        current_candidates = candidate_paths.copy()

        # Stage 0: First Frame Validation (optional)
        if use_stage0:
            print(f"\n{'='*80}")
            print("🎬 STAGE 0: FIRST FRAME VALIDATION")
            print(f"{'='*80}\n")

            from pipeline.stage1_filter import run_stage0_first_frame_validation

            stage0_start = time.time()

            stage0_results, grounded_specs = run_stage0_first_frame_validation(
                video_paths=current_candidates,
                spec_json_str=prompt_alignment_checklist,
                model_name=stage0_model_name,
                num_gpus=stage0_num_gpus,
                workers_per_gpu=stage0_workers_per_gpu
            )

            # Filter passed videos
            passed_videos = [video_path for video_path, passed, reason in stage0_results if passed]

            stage0_time = time.time() - stage0_start

            print(f"\n📊 Stage 0 Results:")
            print(f"   Input: {len(current_candidates)} videos")
            print(f"   Passed: {len(passed_videos)} videos")
            print(f"   Failed: {len(current_candidates) - len(passed_videos)} videos")
            print(f"   Time: {stage0_time:.2f}s")

            stats['stages']['stage0'] = {
                'time': stage0_time,
                'total': len(current_candidates),
                'passed': len(passed_videos),
                'failed': len(current_candidates) - len(passed_videos)
            }

            current_candidates = passed_videos

            if len(current_candidates) == 0:
                print("\n❌ No videos passed Stage 0 validation!")
                return {
                    'winner_index': 0,
                    'winner_path': candidate_paths[0],
                    'stats': stats
                }

        # Stage 1: Hallucination Filter (optional)
        if use_stage1_filter:
            print(f"\n{'='*80}")
            print("🔍 STAGE 1: VISUAL HALLUCINATION FILTER")
            print(f"{'='*80}\n")

            stage1_start = time.time()

            # Use instance method instead of importing non-existent function
            passed_videos = self.run_stage1_filter(
                video_paths=current_candidates,
                text_prompt=text_prompt,
                checklist=prompt_alignment_checklist,
                num_gpus=stage1_num_gpus,
                workers_per_gpu=stage1_workers_per_gpu,
                max_iterations=stage1_max_iterations
            )

            stage1_time = time.time() - stage1_start

            print(f"\n📊 Stage 1 Results:")
            print(f"   Input: {len(current_candidates)} videos")
            print(f"   Passed: {len(passed_videos)} videos")
            print(f"   Max iterations: {stage1_max_iterations}")
            print(f"   Time: {stage1_time:.2f}s")

            stats['stages']['stage1'] = {
                'time': stage1_time,
                'total': len(current_candidates),
                'passed': len(passed_videos),
                'max_iterations': stage1_max_iterations,
                'final_passed': len(passed_videos)
            }

            current_candidates = passed_videos

            if len(current_candidates) == 0:
                print("\n❌ No videos passed Stage 1 filter!")
                return {
                    'winner_index': 0,
                    'winner_path': candidate_paths[0],
                    'stats': stats
                }

            # Stage 1 (SAM3 path) includes Stage 1c tournament — already down to 1 winner
            if len(current_candidates) == 1:
                print(f"\n   ✅ Stage 1 (SAM3 + tournament) produced a single winner — skipping Stages 2 & 3")
                winner_path = current_candidates[0]
                try:
                    winner_index = candidate_paths.index(winner_path)
                except ValueError:
                    winner_index = 0
                stats['total_time'] = time.time() - total_start_time
                return {
                    'winner_index': winner_index,
                    'winner_path': winner_path,
                    'stats': stats
                }

        # Stages 2 & 3: Forensic Observation + Tournament
        print(f"\n{'='*80}")
        print("🎯 STAGES 2 & 3: FORENSIC OBSERVATION + TOURNAMENT")
        print(f"{'='*80}\n")

        stage23_start = time.time()

        # Use compare_videos_gemini_tournament which handles both Stage 2 & 3
        winner_path = self.compare_videos_gemini_tournament(
            video_paths=current_candidates,
            text_prompt=text_prompt
        )

        stage23_time = time.time() - stage23_start

        # Find winner index in original candidate_paths
        try:
            winner_index = candidate_paths.index(winner_path)
        except ValueError:
            # Winner not in original list (shouldn't happen)
            winner_index = 0
            winner_path = candidate_paths[0]

        print(f"\n📊 Stages 2 & 3 Results:")
        print(f"   Candidates evaluated: {len(current_candidates)}")
        print(f"   Winner: candidate_{winner_index}")
        print(f"   Time: {stage23_time:.2f}s")

        stats['stages']['stage2'] = {
            'time': stage23_time / 2,  # Approximate
            'videos_analyzed': len(current_candidates)
        }

        stats['stages']['stage3'] = {
            'time': stage23_time / 2,  # Approximate
            'rounds': len(current_candidates) - 1 if len(current_candidates) > 1 else 0,
            'comparisons': len(current_candidates) * (len(current_candidates) - 1) // 2 if len(current_candidates) > 1 else 0
        }

        total_time = time.time() - total_start_time
        stats['total_time'] = total_time

        print(f"\n{'='*80}")
        print(f"✅ PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {total_time:.2f}s")
        print(f"{'='*80}\n")

        return {
            'winner_index': winner_index,
            'winner_path': winner_path,
            'stats': stats
        }