#!/usr/bin/env python3
"""
Stage 1 Hallucination Filter for Video Evaluation
Supports multi-GPU parallelization with batched processing and multi-threading

Architecture:
- Stage 0: First Frame Validation & Grounding (validate + fill UNKNOWN fields)
- Phase 1a: Static Element Check (10 random frames per video)
- Phase 1b: Temporal Consistency Check (12 frames: beginning→middle→end)
- Multi-GPU: Load model replica on each GPU, distribute videos round-robin
- Batching: Process all frames in single inference call
- Multi-threading: 5 workers per GPU for parallel video processing
"""

import torch
import os
import random
import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoProcessor
try:
    from transformers import Qwen3VLForConditionalGeneration
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False
from qwen_vl_utils import process_vision_info
import re


def sample_random_frames(video_path: str, count: int = 10, seed: int = None) -> Tuple[List[Image.Image], List[int]]:
    """
    Sample random frames from video.

    Args:
        video_path: Path to video file
        count: Number of frames to sample
        seed: Random seed for reproducibility (different per iteration)

    Returns:
        Tuple of (frames, frame_indices)
    """
    if seed is not None:
        random.seed(seed)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= count:
        # Sample all frames if video is short
        frame_indices = list(range(total_frames))
    else:
        # Random sampling
        frame_indices = sorted(random.sample(range(total_frames), count))

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)

    cap.release()
    return frames, frame_indices


def sample_consecutive_frames(video_path: str, start: str = "middle", count: int = 8,
                              seed: int = None) -> Tuple[List[Image.Image], List[int]]:
    """
    Sample consecutive frames from video.

    Args:
        video_path: Path to video file
        start: Starting position ("beginning", "middle", "end", or random with seed)
        count: Number of consecutive frames
        seed: Random seed for random start position

    Returns:
        Tuple of (frames, frame_indices)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine start frame
    if start == "beginning":
        start_idx = 0
    elif start == "end":
        start_idx = max(0, total_frames - count)
    elif start == "middle":
        start_idx = max(0, (total_frames - count) // 2)
    else:
        # Random start with seed
        if seed is not None:
            random.seed(seed)
        start_idx = random.randint(0, max(0, total_frames - count))

    # Sample consecutive frames
    frames = []
    frame_indices = list(range(start_idx, min(start_idx + count, total_frames)))
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)

    cap.release()
    return frames, frame_indices


def sample_beginning_middle_end_frames(video_path: str, frames_per_section: int = 4) -> Tuple[List[Image.Image], List[int]]:
    """
    Sample frames from beginning, middle, and end of video to capture full temporal progression.

    This sampling strategy is ideal for prompt-aware temporal consistency checking because it captures:
    - BEGINNING: Initial state (e.g., whole egg, ball at top)
    - MIDDLE: Transition/action (e.g., egg cracking, ball bouncing)
    - END: Final state (e.g., broken pieces, ball settled)

    Args:
        video_path: Path to video file
        frames_per_section: Number of frames to sample from each section (default: 4)
                           Total frames = 3 * frames_per_section

    Returns:
        Tuple of (frames, frame_indices)
        - frames: List of PIL Images (beginning frames + middle frames + end frames)
        - frame_indices: List of frame indices in video

    Example:
        For a 100-frame video with frames_per_section=4:
        - Beginning: frames [0, 8, 16, 24] (first third)
        - Middle: frames [38, 46, 54, 62] (middle third)
        - End: frames [76, 84, 92, 100] (last third)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Divide video into 3 sections
    section_size = total_frames // 3

    # Define ranges for each section
    beginning_range = (0, section_size)
    middle_range = (section_size, 2 * section_size)
    end_range = (2 * section_size, total_frames)

    all_frames = []
    all_indices = []

    # Sample from each section
    for section_name, (start, end) in [
        ("beginning", beginning_range),
        ("middle", middle_range),
        ("end", end_range)
    ]:
        # Evenly space frames within this section
        section_length = end - start
        if section_length < frames_per_section:
            # If section too small, sample all available frames
            indices = list(range(start, end))
        else:
            # Evenly space frames_per_section frames
            step = section_length / frames_per_section
            indices = [int(start + i * step) for i in range(frames_per_section)]

        # Extract frames
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                all_frames.append(pil_image)
                all_indices.append(idx)

    cap.release()
    return all_frames, all_indices


def extract_first_frame(video_path: str) -> Image.Image:
    """
    Extract the first frame from a video.

    Args:
        video_path: Path to video file

    Returns:
        PIL Image of the first frame
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read first frame from {video_path}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def visualize_bounding_boxes(video_path: str, grounded_spec: Dict, output_path: Optional[str] = None) -> Image.Image:
    """
    Visualize bounding boxes from grounded specification on the first frame.

    Draws bounding boxes for all detected entities (subject, interactive_entities, background)
    with color-coded boxes and labels.

    Args:
        video_path: Path to video file
        grounded_spec: Grounded specification dict (from validate_and_ground_first_frame)
        output_path: Optional path to save annotated image (if None, only returns PIL Image)

    Returns:
        PIL Image with bounding boxes drawn

    Example:
        result, _ = validate_and_ground_first_frame(video_path, spec_json, model, processor, device)
        annotated_img = visualize_bounding_boxes(video_path, result['grounded_spec'], "output.jpg")
    """
    # Extract first frame
    frame = extract_first_frame(video_path)
    img_width, img_height = frame.size

    # Create drawing context
    draw = ImageDraw.Draw(frame)

    # Try to load a font (fallback to default if not available)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except:
            font = ImageFont.load_default()

    # Color mapping for different entity roles
    role_colors = {
        "subject": "#FF0000",           # Red
        "interactive_entity": "#00FF00", # Green
        "background": "#0080FF"          # Blue
    }

    # Function to parse bbox - supports normalized [x1,y1,x2,y2] and old percentage format
    def parse_bbox_string(bbox_data):
        if not bbox_data or bbox_data == "UNKNOWN":
            return None
        try:
            # Handle both array and string formats
            if isinstance(bbox_data, list):
                # Direct array format: [0.28, 0.0, 0.67, 0.35]
                coords = bbox_data
            elif isinstance(bbox_data, str):
                # String format: "[0.28, 0.0, 0.67, 0.35]"
                list_match = re.search(r'\[([0-9.,\s]+)\]', bbox_data)
                if list_match:
                    coords = [float(x.strip()) for x in list_match.group(1).split(',')]
                else:
                    coords = None
            else:
                coords = None

            # Process coordinates if we got them
            if coords and len(coords) == 4:
                x1, y1, x2, y2 = coords
                # Convert normalized to pixel coordinates
                x1_px = int(float(x1) * img_width)
                y1_px = int(float(y1) * img_height)
                x2_px = int(float(x2) * img_width)
                y2_px = int(float(y2) * img_height)
                return (x1_px, y1_px, x2_px, y2_px)

            # Fall back to old percentage format if string
            if isinstance(bbox_data, str):
                # Old percentage format: x=40%, y=20%, w=15%, h=15%
                x_match = re.search(r'x=(\d+(?:\.\d+)?)%', bbox_data)
                y_match = re.search(r'y=(\d+(?:\.\d+)?)%', bbox_data)
                w_match = re.search(r'w=(\d+(?:\.\d+)?)%', bbox_data)
                h_match = re.search(r'h=(\d+(?:\.\d+)?)%', bbox_data)

                if all([x_match, y_match, w_match, h_match]):
                    x_pct = float(x_match.group(1)) / 100.0
                    y_pct = float(y_match.group(1)) / 100.0
                    w_pct = float(w_match.group(1)) / 100.0
                    h_pct = float(h_match.group(1)) / 100.0

                    # Convert to pixel coordinates
                    x = int(x_pct * img_width)
                    y = int(y_pct * img_height)
                    w = int(w_pct * img_width)
                    h = int(h_pct * img_height)

                    return (x, y, x + w, y + h)  # (x1, y1, x2, y2)
        except:
            pass
        return None

    # Extract and draw bounding boxes for all entities
    entities = grounded_spec.get("entities", {})
    detection_count = 0

    # DEBUG: Print structure to help diagnose bounding box issues
    print(f"\n[DEBUG] Grounded spec entities keys: {list(entities.keys())}")
    if "subject" in entities:
        subject = entities["subject"]
        print(f"[DEBUG] Subject keys: {list(subject.keys())}")
        if "appearance" in subject:
            print(f"[DEBUG] Subject appearance keys: {list(subject['appearance'].keys())}")
            print(f"[DEBUG] Subject exact_bbox: {subject['appearance'].get('exact_bbox', 'NOT FOUND')}")

    # Process subject
    if "subject" in entities:
        subject = entities["subject"]
        # Check both appearance and geometry sections for exact_bbox
        bbox_str = subject.get("appearance", {}).get("exact_bbox", "UNKNOWN")
        if bbox_str == "UNKNOWN":
            bbox_str = subject.get("geometry", {}).get("exact_bbox", "UNKNOWN")
        bbox = parse_bbox_string(bbox_str)

        if bbox:
            color = role_colors["subject"]
            draw.rectangle(bbox, outline=color, width=3)

            label = f"SUBJECT: {subject.get('name', 'unknown')}"
            # Draw label background
            text_bbox = draw.textbbox((bbox[0], bbox[1] - 20), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((bbox[0], bbox[1] - 20), label, fill="white", font=font)
            detection_count += 1

    # Process interactive_entities (can be dict or list)
    interactive_entities = entities.get("interactive_entities", entities.get("interactive_entity"))

    if interactive_entities:
        # Handle both single dict and list of dicts
        if isinstance(interactive_entities, dict):
            interactive_entities = [interactive_entities]

        for idx, entity in enumerate(interactive_entities):
            # Check both appearance and geometry sections for exact_bbox
            bbox_str = entity.get("appearance", {}).get("exact_bbox", "UNKNOWN")
            if bbox_str == "UNKNOWN":
                bbox_str = entity.get("geometry", {}).get("exact_bbox", "UNKNOWN")
            bbox = parse_bbox_string(bbox_str)

            if bbox:
                color = role_colors["interactive_entity"]
                draw.rectangle(bbox, outline=color, width=3)

                entity_name = entity.get('name', f'interactive_{idx}')
                interaction_type = entity.get('interaction_type', '')
                label = f"INTERACTIVE: {entity_name}"
                if interaction_type:
                    label += f" ({interaction_type})"

                # Draw label background
                text_bbox = draw.textbbox((bbox[0], bbox[1] - 20), label, font=font)
                draw.rectangle(text_bbox, fill=color)
                draw.text((bbox[0], bbox[1] - 20), label, fill="white", font=font)
                detection_count += 1

    # Process background
    if "background" in entities:
        background = entities["background"]
        # Check both appearance and geometry sections for exact_bbox
        bbox_str = background.get("appearance", {}).get("exact_bbox", "UNKNOWN")
        if bbox_str == "UNKNOWN":
            bbox_str = background.get("geometry", {}).get("exact_bbox", "UNKNOWN")
        bbox = parse_bbox_string(bbox_str)

        if bbox:
            color = role_colors["background"]
            draw.rectangle(bbox, outline=color, width=2)  # Thinner line for background

            label = f"BACKGROUND: {background.get('name', 'unknown')}"
            text_bbox = draw.textbbox((bbox[0], bbox[1] - 20), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((bbox[0], bbox[1] - 20), label, fill="white", font=font)
            detection_count += 1

    # Add legend
    legend_y = 10
    legend_items = [
        ("SUBJECT", role_colors["subject"]),
        ("INTERACTIVE", role_colors["interactive_entity"]),
        ("BACKGROUND", role_colors["background"])
    ]

    for label, color in legend_items:
        # Draw color box
        box_coords = (img_width - 150, legend_y, img_width - 130, legend_y + 15)
        draw.rectangle(box_coords, fill=color)
        # Draw label
        draw.text((img_width - 125, legend_y), label, fill=color, font=font)
        legend_y += 25

    # Add detection count
    count_text = f"Detections: {detection_count}"
    draw.text((10, img_height - 30), count_text, fill="#FFFFFF", font=font)

    # Save if output path provided
    if output_path:
        frame.save(output_path)
        print(f"✅ Saved annotated image to: {output_path}")

    return frame


def validate_and_ground_first_frame(video_path: str, spec_json_str: str, model, processor, device) -> Tuple[Dict, str]:
    """
    Stage 0: First Frame Validation & Grounding

    Tasks:
    1. Detect and segment all entities in the first frame
    2. Validate against specification (counts, geometry, materials)
    3. Fill UNKNOWN fields (colors, bounding boxes, positions)
    4. Detect unexpected entities not in specification

    Args:
        video_path: Path to video file
        spec_json_str: JSON specification string (from generate_prompt_alignment_checklist)
        model: Qwen2.5-VL model
        processor: Qwen2.5-VL processor
        device: Device

    Returns:
        Tuple of (result_dict, raw_response)
        - result_dict: {"passed": bool, "grounded_spec": dict, "reason": str}
        - raw_response: Full VLM response text
    """
    # Extract first frame
    first_frame = extract_first_frame(video_path)

    # Parse JSON specification
    try:
        spec_dict = json.loads(spec_json_str)
    except json.JSONDecodeError:
        # If not JSON, skip Stage 0 (backward compatibility with old text checklists)
        return {
            "passed": True,
            "grounded_spec": None,
            "reason": "Specification not in JSON format - skipping first frame validation"
        }, "SKIPPED"

    # Extract key requirements from specification
    entities = spec_dict.get("entities", {})
    validation_constraints = spec_dict.get("validation_constraints", {})
    count_constraints = validation_constraints.get("count_constraints", {})
    geometry_constraints = validation_constraints.get("geometry_constraints", {})

    # Build entity detection prompt
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": first_frame},
            {"type": "text", "text": f"""
You are a forensic video analyst performing FIRST FRAME VALIDATION and GROUNDING.

TASK: Analyze this first frame and validate it against the specification below, then fill in all UNKNOWN fields.

CRITICAL: You are validating whether the OBSERVED frame matches the REQUIRED specification.
- If specification requires count=1 but you observe count=0 → This is a MISMATCH → FAIL
- If specification requires "sphere" but you observe "cube" → This is a MISMATCH → FAIL

=== SPECIFICATION ===
{json.dumps(spec_dict, indent=2)}

=== YOUR TASKS ===

**TASK 1: ENTITY DETECTION & SEGMENTATION**
For each entity in the specification (subject, interactive_entity, background):
1. Detect if the entity exists in the frame
2. Count how many instances you see
3. Describe its visual appearance (color name, hex code estimate, shape, material evidence)
4. Estimate bounding box coordinates (x, y, width, height) as percentages of image dimensions
5. Describe its exact position in the frame

**TASK 2: VALIDATION CHECKS**
Check if the frame satisfies ALL validation constraints FROM THE SPECIFICATION:
- ✓ Count constraints: Compare YOUR DETECTED count against validation_constraints.count_constraints.exact
  Example: If spec says ball.count.exact=1 but you detected 0 balls → FAIL
  Example: If spec says ball.count.exact=1 and you detected 1 ball → PASS
- ✓ Geometry constraints: Does the shape/geometry match? (check entities.*.geometry and validation_constraints.geometry_constraints)
- ✓ Material evidence: Does visual appearance suggest expected material? (check entities.*.material)
- ✓ Initial state: Is the entity in the expected initial state? (check entities.*.initial_state.description)

**TASK 3: UNEXPECTED ENTITIES**
- Are there any MAJOR objects in the frame that are NOT mentioned in the specification?
- List any unexpected entities (ignore minor background details)

**TASK 4: GROUNDING (Fill UNKNOWN fields)**
For each entity, extract:
- Exact color (name + hex code estimate + RGB estimate)
- Exact bounding box (x%, y%, w%, h%)
- Exact position description
- Visual signature (distinctive features for tracking)

=== RESPONSE FORMAT ===

ENTITY DETECTIONS:
[For each entity in spec]
- Entity: [entity name]
  - Found: [YES/NO]
  - Count: [number detected]
  - Color: [color name] (hex: #XXXXXX, rgb: R,G,B)
  - Bounding Box: [x_min, y_min, x_max, y_max] (normalized 0.0-1.0, e.g., [0.2, 0.1, 0.6, 0.5])
  - Position: [describe exact location in frame]
  - Geometry: [describe observed shape - matches expected?]
  - Material Evidence: [visual clues about material - matches expected?]
  - Initial State: [describe current state - matches expected initial state?]

UNEXPECTED ENTITIES:
[List any major objects not in specification, or write "None"]

VALIDATION RESULT: [PASS or FAIL]
(Output ONLY "PASS" or "FAIL" on this line. If ANY check below fails, output FAIL.)

VALIDATION CHECKS:
- Count Constraints: [PASS/FAIL - explain by comparing detected count vs spec count]
- Geometry Constraints: [PASS/FAIL - explain]
- Material Evidence: [PASS/FAIL - explain]
- Initial State: [PASS/FAIL - explain]

FAILURE REASON (if FAIL): [Detailed explanation of what doesn't match. If VALIDATION RESULT is FAIL, you MUST provide a reason here.]

GROUNDED SPECIFICATION:
```json
[Provide the UPDATED JSON specification with all UNKNOWN fields filled in with the values you detected.

CRITICAL: For each entity, you MUST include exact_bbox in the APPEARANCE section (NOT in geometry):
  "appearance": {{
    "color": "color name",
    "color_hex": "#XXXXXX",
    "color_rgb": [R, G, B],
    "exact_bbox": [x_min, y_min, x_max, y_max],  <- Array format, NOT string! Put in appearance, NOT geometry!
    "exact_position": "description of location",
    "typical_colors": [...]
  }}

IMPORTANT:
- Put exact_bbox in the "appearance" section, NOT in "geometry" section!
- Output ONLY valid JSON - do NOT include HTML comments like <!-- ... --> or any other non-JSON syntax.
]
```
"""}
        ]
    }]

    # Process with VLM
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8192,  # Very large output for detailed grounding with full JSON spec
            temperature=0.1,
            do_sample=False
        )

    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    # Parse response - handle PASS on same line or next line
    # Matches "VALIDATION RESULT: PASS", "VALIDATION RESULT:PASS", or "VALIDATION RESULT:\nPASS"
    validation_match = re.search(r"VALIDATION RESULT:\s*(PASS|FAIL)", response, re.IGNORECASE)
    if validation_match:
        passed = validation_match.group(1).upper() == "PASS"
    else:
        # Fallback to old string matching
        passed = "VALIDATION RESULT: PASS" in response or "VALIDATION RESULT:PASS" in response

    # Extract failure reason if failed
    reason = ""
    if not passed:
        reason_match = re.search(r"FAILURE REASON.*?:\s*(.+?)(?=\n\n|GROUNDED|$)", response, re.IGNORECASE | re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip()
        else:
            reason = "First frame validation failed (see VLM response for details)"

    # Try to extract grounded specification
    grounded_spec = None
    # Try with code block first (```json ... ```)
    # Use non-greedy match until closing ``` (not until first })
    grounded_match = re.search(r"GROUNDED SPECIFICATION:\s*```json\s*(.+?)```", response, re.DOTALL)
    if not grounded_match:
        # Fall back to plain JSON format
        grounded_match = re.search(r"GROUNDED SPECIFICATION:\s*(\{.+\})", response, re.DOTALL)

    print(f"[DEBUG] Attempting to parse grounded spec (match found: {grounded_match is not None})")
    if grounded_match:
        print(f"[DEBUG] Extracted JSON length: {len(grounded_match.group(1))} chars")
        try:
            json_text = grounded_match.group(1).strip()
            # Remove HTML comments that VLM might add (<!-- ... -->)
            json_text = re.sub(r'<!--.*?-->', '', json_text)

            # Fix malformed multi-color RGB arrays: [0,255,0],[0,0,255] -> [[0,255,0],[0,0,255]]
            # Match multiple RGB triplets and wrap them in an outer array
            json_text = re.sub(
                r'"color_rgb":\s*(\[[0-9,\s]+\])(?:,\s*(\[[0-9,\s]+\]))+',
                lambda m: '"color_rgb": [' + m.group(0).split(':', 1)[1] + ']',
                json_text
            )

            # Fix malformed hex strings: "#00FF00,#0000FF" -> ["#00FF00","#0000FF"]
            json_text = re.sub(
                r'"color_hex":\s*"(#[0-9A-Fa-f]+(?:,#[0-9A-Fa-f]+)+)"',
                lambda m: '"color_hex": ["' + m.group(1).replace(',', '","') + '"]',
                json_text
            )

            grounded_spec = json.loads(json_text)
            print(f"[DEBUG] ✅ Successfully parsed grounded JSON spec")
        except json.JSONDecodeError as e:
            print(f"[DEBUG] ❌ JSON parse error: {e}")
            print(f"[DEBUG] Failed JSON text (first 500 chars): {grounded_match.group(1).strip()[:500]}")
            grounded_spec = None

    # If we got a grounded spec, try to fill in missing bounding boxes from text analysis
    if grounded_spec and "entities" in grounded_spec:
        # Extract bounding boxes from the ENTITY DETECTIONS text section
        entity_detections = re.search(r"ENTITY DETECTIONS:(.*?)(?:UNEXPECTED ENTITIES:|VALIDATION RESULT:)", response, re.DOTALL)
        if entity_detections:
            detection_text = entity_detections.group(1)
            print(f"[DEBUG] Extracted ENTITY DETECTIONS text (length: {len(detection_text)} chars)")

            # Extract bounding box for subject
            subject_bbox = re.search(r"(?:Subject|Ball|subject).*?Bounding Box:\s*\[([0-9.,\s]+)\]", detection_text, re.DOTALL | re.IGNORECASE)
            if subject_bbox and "subject" in grounded_spec["entities"]:
                print(f"[DEBUG] Found subject bbox in text: {subject_bbox.group(1)}")
                if "appearance" not in grounded_spec["entities"]["subject"]:
                    grounded_spec["entities"]["subject"]["appearance"] = {}
                if grounded_spec["entities"]["subject"]["appearance"].get("exact_bbox", "UNKNOWN") == "UNKNOWN" or \
                   "exact_bbox" not in grounded_spec["entities"]["subject"]["appearance"]:
                    bbox_coords = subject_bbox.group(1).strip()
                    grounded_spec["entities"]["subject"]["appearance"]["exact_bbox"] = f"[{bbox_coords}]"
                    print(f"[DEBUG] Filled subject bbox from text: [{bbox_coords}]")

            # Extract bounding boxes for interactive entities
            interactive_pattern = re.compile(r"(?:Interactive Entity|interactive_entity).*?Bounding Box:\s*\[([0-9.,\s]+)\]", re.DOTALL | re.IGNORECASE)
            interactive_bboxes = interactive_pattern.findall(detection_text)

            if interactive_bboxes and "interactive_entities" in grounded_spec["entities"]:
                entities_list = grounded_spec["entities"]["interactive_entities"]
                if isinstance(entities_list, list):
                    for idx, bbox_coords in enumerate(interactive_bboxes):
                        if idx < len(entities_list):
                            if "appearance" not in entities_list[idx]:
                                entities_list[idx]["appearance"] = {}
                            if entities_list[idx]["appearance"].get("exact_bbox", "UNKNOWN") == "UNKNOWN" or \
                               "exact_bbox" not in entities_list[idx]["appearance"]:
                                entities_list[idx]["appearance"]["exact_bbox"] = f"[{bbox_coords.strip()}]"
                                print(f"[DEBUG] Filled interactive_entity[{idx}] bbox from text: [{bbox_coords.strip()}]")

            # Extract bounding box for background
            bg_bbox = re.search(r"(?:Background|background).*?Bounding Box:\s*\[([0-9.,\s]+)\]", detection_text, re.DOTALL | re.IGNORECASE)
            if bg_bbox and "background" in grounded_spec["entities"]:
                print(f"[DEBUG] Found background bbox in text: {bg_bbox.group(1)}")
                if "appearance" not in grounded_spec["entities"]["background"]:
                    grounded_spec["entities"]["background"]["appearance"] = {}
                if grounded_spec["entities"]["background"]["appearance"].get("exact_bbox", "UNKNOWN") == "UNKNOWN" or \
                   "exact_bbox" not in grounded_spec["entities"]["background"]["appearance"]:
                    bbox_coords = bg_bbox.group(1).strip()
                    grounded_spec["entities"]["background"]["appearance"]["exact_bbox"] = f"[{bbox_coords}]"
                    print(f"[DEBUG] Filled background bbox from text: [{bbox_coords}]")

    result = {
        "passed": passed,
        "grounded_spec": grounded_spec if grounded_spec else spec_dict,  # Fall back to original if parsing failed
        "reason": reason if not passed else "First frame validation passed"
    }

    return result, response


def extract_random_frames(video_path: str, num_frames: int = 30, seed: Optional[int] = None) -> List[Tuple[Image.Image, int]]:
    """
    Extract frames uniformly distributed across video duration.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default: 30)
        seed: Random seed for reproducibility (optional, not used for uniform sampling)

    Returns:
        List of (PIL.Image, frame_index) tuples
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= num_frames:
        # If video has fewer frames than requested, sample all frames
        frame_indices = list(range(total_frames))
    else:
        # Uniform sampling: evenly distribute frames across video duration
        # E.g., for 30 frames from 100 total: [0, 3, 6, 9, ..., 96, 99]
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            frames.append((pil_frame, frame_idx))

    cap.release()
    return frames


def build_phase_grounding_prompt(total_frames: int, phase: Dict, subject_name: str, interactive_entities: List[str]) -> str:
    """
    Build grounding prompt to DETECT when a specific phase occurs in the video.
    Generic approach that works for any subject-entity interaction.

    Args:
        total_frames: Total number of frames in sampled video
        phase: Phase dict with 'phase_name' and 'description'
        subject_name: Name of the subject entity
        interactive_entities: List of interactive entity names

    Returns:
        Grounding prompt for detecting this phase
    """
    phase_name = phase.get('phase_name', '')
    phase_desc = phase.get('description', '')

    # Determine phase type and build type-specific grounding instructions
    if phase_name == 'initial':
        grounding_task = f"""
**DETECTION TASK: Identify "INITIAL STATE" frames**

Detect frames where **{subject_name}** is in STARTING configuration BEFORE interacting with any entities.

Visual indicators to look for:
- {subject_name} positioned/held/ready in starting state
- NO contact with interactive entities yet: {', '.join(interactive_entities)}
- {subject_name} may be stationary or beginning motion
- Phase description: {phase_desc}

What to detect: ALL frames showing the initial state before first interaction
"""
    elif phase_name == 'final':
        grounding_task = f"""
**DETECTION TASK: Identify "FINAL STATE" frames**

Detect frames where **{subject_name}** is in ENDING configuration AFTER all interactions complete.

Visual indicators to look for:
- {subject_name} at rest/settled in final position
- All interactions with entities finished
- {subject_name} stationary or in stable end state
- Phase description: {phase_desc}

What to detect: ALL frames showing the final state after interactions end
"""
    else:
        # Contact phase - extract entity name from phase_name (e.g., "impact_floor" -> "floor")
        entity_name = phase_name.split('_', 1)[1] if '_' in phase_name else 'entity'

        grounding_task = f"""
**DETECTION TASK: Identify "CONTACT/INTERACTION" frames**

Phase name: "{phase_name}"
Phase description: {phase_desc}

🔴 **MANDATORY SCANNING PROCEDURE:**
1. You MUST watch ALL {total_frames} frames from start (frame 0) to end (frame {total_frames-1})
2. DO NOT stop after finding the first contact - keep scanning!
3. Mark EVERY frame where you see {subject_name} touching {entity_name}

**What you're looking for:**
Frames where **{subject_name}** is ACTIVELY TOUCHING/CONTACTING **{entity_name}**

Visual clues:
- {subject_name} physically touching {entity_name}
- Compression/deformation at contact point
- {subject_name} at lowest point (if bouncing)
- Squishing, flattening visible
- Reaction forces: bouncing back, rebounding

🚨 **IF THE DESCRIPTION MENTIONS "BOUNCES" OR "BOUNCING":**

The {subject_name} will likely bounce MULTIPLE TIMES throughout the video (typically 3-10 bounces).

Pattern you'll see:
- Bounce 1: {subject_name} falls → CONTACTS {entity_name} → rises
- Bounce 2: {subject_name} falls again → CONTACTS {entity_name} → rises
- Bounce 3: {subject_name} falls again → CONTACTS {entity_name} → rises
- ... continues until energy dissipates

Each bounce happens at a DIFFERENT time in the video:
- First bounce might be around frame 3-5
- Second bounce around frame 8-10
- Third bounce around frame 13-15
- etc.

🔴 **YOU MUST DETECT ALL OF THEM!**

❌ **WRONG OUTPUT:** {{"detected_frames": [3, 4, 5, 6, 7]}} - This only captures the FIRST bounce!

✅ **CORRECT OUTPUT:** {{"detected_frames": [4, 5, 9, 10, 14, 15, 18, 19]}} - This captures MULTIPLE bounces across the video!

**Your task:** Find EVERY moment where {subject_name} touches {entity_name}, not just the first one.
"""

    return f"""You are analyzing a video with {total_frames} frames (indices 0 to {total_frames-1}).

TASK: GROUND/DETECT when a specific phase occurs by identifying ALL relevant frames.

{grounding_task}

=== DETECTION STRATEGY ===

1. **Scan ALL {total_frames} frames** - don't stop after finding first occurrence
2. **Identify visual markers** - look for the specific indicators listed above
3. **Mark EVERY matching frame** - if phase repeats, detect all repetitions
4. **Group if needed** - if phase spans multiple consecutive frames, mark them all

=== OUTPUT FORMAT ===

Return JSON:
{{
  "phase_name": "{phase_name}",
  "detected_frames": [<list of ALL frame indices where this phase is detected>],
  "frame_groups": [
    {{
      "occurrence": 1,
      "frames": [<frame indices for this occurrence>],
      "description": "what's happening"
    }}
  ],
  "total_occurrences": <number of times phase occurs>
}}

=== EXAMPLES ===

**Example 1: Initial phase**
{{
  "phase_name": "initial",
  "detected_frames": [0, 1, 2, 3],
  "frame_groups": [
    {{"occurrence": 1, "frames": [0, 1, 2, 3], "description": "{subject_name} in starting state before interaction"}}
  ],
  "total_occurrences": 1
}}

**Example 2: Contact phase that repeats**
{{
  "phase_name": "{phase_name}",
  "detected_frames": [4, 5, 8, 9, 12, 13, 16, 17],
  "frame_groups": [
    {{"occurrence": 1, "frames": [4, 5], "description": "first interaction - high energy"}},
    {{"occurrence": 2, "frames": [8, 9], "description": "second interaction - medium energy"}},
    {{"occurrence": 3, "frames": [12, 13], "description": "third interaction - lower energy"}},
    {{"occurrence": 4, "frames": [16, 17], "description": "fourth interaction - minimal energy"}}
  ],
  "total_occurrences": 4
}}

**Example 3: Final phase**
{{
  "phase_name": "final",
  "detected_frames": [18, 19, 20, 21, 22, 23],
  "frame_groups": [
    {{"occurrence": 1, "frames": [18, 19, 20, 21, 22, 23], "description": "{subject_name} in final state after all interactions"}}
  ],
  "total_occurrences": 1
}}

=== MANDATORY REQUIREMENTS ===

1. **Frame indices only**: All numbers must be integers 0 to {total_frames-1}
2. **Detect ALL occurrences**: Don't stop at first match if phase repeats
3. **Be exhaustive**: Include every frame where visual markers are present
4. **Group logically**: If multiple consecutive frames show same occurrence, group them

NOW: Analyze all {total_frames} frames and detect when "{phase_name}" occurs."""


def build_bounding_box_grounding_prompt(subject_name: str, entity_name: str, phase_description: str = "") -> str:
    """
    Build prompt for bounding box grounding AND contact point localization.

    Uses bounding boxes to locate objects, then asks VLM to identify the exact contact point.

    Args:
        subject_name: Name of subject entity
        entity_name: Name of interactive entity
        phase_description: Description of the phase/interaction for context

    Returns:
        Prompt string
    """
    context = f"\n**CONTEXT**: {phase_description}\n" if phase_description else ""

    return f"""Locate the **{subject_name}** and **{entity_name}** in this image.
{context}
**TASK**:
1. Provide precise bounding boxes for both objects
2. If they are physically touching, identify the exact (x,y) position of the contact point

**OUTPUT FORMAT** (normalized coordinates 0.0 to 1.0):

```json
{{
  "{subject_name}": {{
    "bbox": [x_min, y_min, x_max, y_max],
    "visible": true
  }},
  "{entity_name}": {{
    "bbox": [x_min, y_min, x_max, y_max],
    "visible": true
  }},
  "contact_point": [x, y]
}}
```

**INSTRUCTIONS:**
- Bounding box format: [x_min, y_min, x_max, y_max] in normalized coordinates (0.0 to 1.0)
- x_min, y_min: top-left corner
- x_max, y_max: bottom-right corner
- Set "visible": false if object is not visible in frame
- Be precise - the box should tightly fit the object
- **contact_point**: If {subject_name} and {entity_name} are physically touching, provide the exact [x, y] coordinate where they touch
- If NOT touching (any gap between objects), set "contact_point": null
- Contact point must be at the boundary where both objects meet

**EXAMPLES:**

Ball touching floor at specific point:
```json
{{
  "{subject_name}": {{"bbox": [0.4, 0.6, 0.5, 0.7], "visible": true}},
  "{entity_name}": {{"bbox": [0.0, 0.7, 1.0, 0.8], "visible": true}},
  "contact_point": [0.45, 0.70]
}}
```

Ball in air (not touching):
```json
{{
  "{subject_name}": {{"bbox": [0.4, 0.3, 0.5, 0.4], "visible": true}},
  "{entity_name}": {{"bbox": [0.0, 0.7, 1.0, 0.8], "visible": true}},
  "contact_point": null
}}
```

Now analyze this image and provide the bounding boxes and contact point (if any)."""


def calculate_bbox_overlap(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate overlap between two bounding boxes (IoU metric).

    Args:
        bbox1: [x_min, y_min, x_max, y_max]
        bbox2: [x_min, y_min, x_max, y_max]

    Returns:
        IoU (Intersection over Union) value between 0.0 and 1.0
    """
    # Extract coordinates
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Calculate intersection
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    # Check if there's actually an intersection
    if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min:
        return 0.0  # No overlap

    # Calculate areas
    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate union
    union_area = bbox1_area + bbox2_area - inter_area

    # Calculate IoU
    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou


def draw_bounding_boxes_on_frame(frame, subject_bbox: List[float], entity_bbox: List[float],
                                  subject_name: str, entity_name: str, iou: float, is_contact: bool,
                                  contact_point: Optional[List[float]] = None):
    """
    Draw bounding boxes and contact point on frame with labels.

    Args:
        frame: OpenCV frame (BGR format)
        subject_bbox: [x_min, y_min, x_max, y_max] in normalized coords
        entity_bbox: [x_min, y_min, x_max, y_max] in normalized coords
        subject_name: Name of subject
        entity_name: Name of entity
        iou: IoU value
        is_contact: Whether contact was detected
        contact_point: [x, y] position of contact point in normalized coords (if any)

    Returns:
        Annotated frame (BGR format)
    """
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    # Convert to PIL for drawing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(pil_image)

    img_width, img_height = pil_image.size

    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except:
            font = ImageFont.load_default()

    # Draw subject bbox (RED)
    x1, y1, x2, y2 = subject_bbox
    x1_px = int(x1 * img_width)
    y1_px = int(y1 * img_height)
    x2_px = int(x2 * img_width)
    y2_px = int(y2 * img_height)
    draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline="red", width=3)
    draw.text((x1_px, y1_px - 20), subject_name, fill="red", font=font)

    # Draw entity bbox (GREEN)
    x1, y1, x2, y2 = entity_bbox
    x1_px = int(x1 * img_width)
    y1_px = int(y1 * img_height)
    x2_px = int(x2 * img_width)
    y2_px = int(y2 * img_height)
    draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline="green", width=3)
    draw.text((x1_px, y1_px - 20), entity_name, fill="green", font=font)

    # Draw contact point if exists (YELLOW circle)
    if contact_point and len(contact_point) == 2:
        cp_x, cp_y = contact_point
        cp_x_px = int(cp_x * img_width)
        cp_y_px = int(cp_y * img_height)
        # Draw a circle at contact point
        radius = 8
        draw.ellipse(
            [cp_x_px - radius, cp_y_px - radius, cp_x_px + radius, cp_y_px + radius],
            fill="yellow",
            outline="orange",
            width=2
        )
        # Draw crosshair
        draw.line([cp_x_px - radius - 5, cp_y_px, cp_x_px + radius + 5, cp_y_px], fill="orange", width=2)
        draw.line([cp_x_px, cp_y_px - radius - 5, cp_x_px, cp_y_px + radius + 5], fill="orange", width=2)

    # Draw status text
    status_color = "red" if is_contact else "white"
    if contact_point and len(contact_point) == 2:
        status_text = f"Contact: YES at ({contact_point[0]:.2f}, {contact_point[1]:.2f}) | IoU: {iou:.2%}"
    else:
        status_text = f"Contact: NO | IoU: {iou:.2%}"
    draw.text((10, 10), status_text, fill=status_color, font=font)

    # Convert back to OpenCV BGR
    annotated_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return annotated_frame


def process_single_frame_with_bbox_grounding(frame, subject_name: str, entity_name: str, model, processor, device,
                                               save_dir: Optional[str] = None, frame_idx: int = 0, video_name: str = "video",
                                               phase_description: str = "") -> bool:
    """
    Process a single frame to detect contact using bounding box grounding + VLM binary decision.

    Uses bounding boxes to help VLM locate objects, then asks VLM for binary contact judgment.

    Args:
        frame: OpenCV frame (BGR format)
        subject_name: Name of subject entity
        entity_name: Name of interactive entity
        model: Qwen3-VL model
        processor: Qwen3-VL processor
        device: Device
        save_dir: Optional directory to save annotated frames with bounding boxes
        frame_idx: Frame index (for naming saved files)
        video_name: Video name (for naming saved files)
        phase_description: Description of the phase/interaction for context

    Returns:
        True if VLM determines objects are at point of contact, False otherwise
    """
    from PIL import Image

    # Convert OpenCV frame (BGR) to PIL Image (RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Build grounding prompt with phase context
    prompt = build_bounding_box_grounding_prompt(subject_name, entity_name, phase_description)

    # Build messages
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": prompt}
        ]
    }]

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Process images
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    # Prepare inputs
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        return_tensors="pt",
        do_resize=False
    ).to(device)

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False
        )

    # Decode response
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    # Parse bounding boxes
    try:
        # Extract JSON
        json_text = response
        if "```json" in response:
            json_text = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_text = response.split("```")[1].split("```")[0].strip()

        bbox_data = json.loads(json_text)

        # Extract bboxes
        subject_data = bbox_data.get(subject_name, {})
        entity_data = bbox_data.get(entity_name, {})

        subject_bbox = subject_data.get("bbox", [0, 0, 0, 0])
        entity_bbox = entity_data.get("bbox", [0, 0, 0, 0])

        subject_visible = subject_data.get("visible", False)
        entity_visible = entity_data.get("visible", False)

        # Both must be visible for contact
        if not (subject_visible and entity_visible):
            return False

        # Get contact point from VLM
        # If VLM identifies a specific contact point, then there IS contact
        contact_point = bbox_data.get("contact_point", None)

        # Contact detected if VLM provides a valid contact point
        contact = contact_point is not None and isinstance(contact_point, list) and len(contact_point) == 2

        # Calculate IoU for visualization purposes
        iou = calculate_bbox_overlap(subject_bbox, entity_bbox)

        # Save annotated frame if save_dir provided
        if save_dir:
            # Create bounding_box directory
            bbox_dir = os.path.join(save_dir, "bounding_boxes")
            os.makedirs(bbox_dir, exist_ok=True)

            # Draw bounding boxes and contact point on frame
            annotated_frame = draw_bounding_boxes_on_frame(
                frame=frame,
                subject_bbox=subject_bbox,
                entity_bbox=entity_bbox,
                subject_name=subject_name,
                entity_name=entity_name,
                iou=iou,
                is_contact=contact,
                contact_point=contact_point
            )

            # Save annotated frame
            if contact and contact_point:
                contact_str = f"CONTACT_at_{contact_point[0]:.2f}_{contact_point[1]:.2f}"
            else:
                contact_str = "no_contact"
            filename = f"{video_name}_frame_{frame_idx:04d}_{contact_str}_iou_{iou:.3f}.jpg"
            save_path = os.path.join(bbox_dir, filename)
            cv2.imwrite(save_path, annotated_frame)

        return contact

    except (json.JSONDecodeError, KeyError, ValueError):
        # Failed to parse - assume no contact
        return False


def _vlm_infer_images(model, processor, device, image_paths, prompt):
    """
    Run VLM on a sequence of image paths + text prompt.
    Supports LLaVA-Video (processor is a tuple) and Qwen3/2.5-VL.
    Returns raw string output.
    """
    import torch as _torch
    import numpy as _np
    from PIL import Image as _PILImage

    # ── LLaVA-Video path ──────────────────────────────────────────────────────
    if isinstance(processor, tuple):
        tokenizer, image_processor, max_length = processor

        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        import copy

        # Load images as numpy frames [N, H, W, 3]
        valid_paths = [p for p in image_paths if p and os.path.isfile(p)]
        if not valid_paths:
            return ""
        frames = []
        for p in valid_paths:
            img = _PILImage.open(p).convert("RGB").resize((336, 336))
            frames.append(_np.array(img))
        frames_np = _np.stack(frames)  # [N, 336, 336, 3]

        video_tensor = image_processor.preprocess(
            frames_np, return_tensors="pt")["pixel_values"].to(device).half()
        video_tensor = [video_tensor]

        video_time  = len(frames_np)   # treat frame count as "seconds" for prompt
        frame_times = ", ".join([f"{i}f" for i in range(len(frames_np))])
        time_instr  = (f"The sequence has {len(frames_np)} frames in temporal order "
                       f"(frames: {frame_times}).")

        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instr}\n{prompt}"
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        with _torch.no_grad():
            cont = model.generate(
                input_ids,
                images=video_tensor,
                modalities=["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=512,
            )
        return tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

    # ── Qwen2.5/Qwen3-VL path — pass frames as VIDEO for temporal reasoning ─────
    # Pass list of file:// paths as "video" — Qwen2.5-VL treats them as sequential
    # frames and attends across the whole sequence for object permanence reasoning.
    # No fps needed when passing image list (fps only applies to actual video files).
    valid_paths = [p for p in image_paths if p and os.path.isfile(p)]
    if not valid_paths:
        return ""

    # Resize all crops to uniform size to avoid tensor stack size mismatch in Qwen processor
    import tempfile as _tempfile
    _resized_dir = _tempfile.mkdtemp(prefix="qwen_crops_")
    resized_paths = []
    for i, p in enumerate(valid_paths):
        img = _PILImage.open(p).convert("RGB").resize((336, 336))
        out = os.path.join(_resized_dir, f"frame_{i:04d}.jpg")
        img.save(out)
        resized_paths.append(out)

    file_uris = [f"file://{os.path.abspath(p)}" for p in resized_paths]

    messages = [{"role": "user", "content": [
        {"type": "video", "video": file_uris},
        {"type": "text",  "text": prompt},
    ]}]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with _torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=1536, temperature=0.1, do_sample=True, top_p=0.9)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()


def _llm_infer_causal(model, tokenizer, prompt, max_new_tokens=1024):
    """
    Run text-only inference on a causal LM (e.g. Qwen2.5-14B-Instruct).
    Uses AutoModelForCausalLM / AutoTokenizer pattern.
    Returns raw string output.
    """
    import torch as _torch
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    with _torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.1, do_sample=True, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id)
    # Decode only the newly generated tokens
    new_tokens = generated_ids[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_tournament_model(model_name: str = "Qwen/Qwen3-8B",
                          device: str = "cuda:0"):
    """
    Load a text-only causal LM on a specific GPU for tournament selection.
    One copy per GPU — avoids device_map="auto" pipeline parallelism which
    ties up both GPUs for a single forward pass and prevents parallelism.
    Returns (model, tokenizer).
    """
    import torch as _torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"   Loading tournament LLM: {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=_torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"   ✓ Tournament LLM loaded ({model_name} on {device})")
    return model, tokenizer


def _find_bounce_arcs(win_pts, frame_height=480):
    """
    Find all bounce arcs in a sliding window using raw subj_y directly.

    No smoothing — the windows are short (~9 frames) so SG filter distorts peaks.
    Uses find_peaks on raw signal with auto-prominence (10% of range).

    Contacts = local MAX subj_y (ball at floor).
    Apexes   = local MIN subj_y (ball in air).

    For each contact:
      pb = last apex before it, OR min subj_y in pre-segment if no apex found
      pa = first apex after it, OR min subj_y in post-segment if no apex found

    All contacts included (no edge truncation). Edge flag set for caller.

    Returns list of dicts:
      { 'c_idx', 'c_frame', 'c_y',
        'pb': (frame, subj_y) or None,
        'pa': (frame, subj_y) or None }
    Returns [] if < 3 points.
    """
    import numpy as _np
    try:
        from scipy.signal import find_peaks as _find_peaks
    except ImportError:
        return []

    if len(win_pts) < 3:
        return []

    frames  = _np.array([f for f, _ in win_pts])
    subj_ys = _np.array([y for _, y in win_pts], dtype=float)

    # Auto prominence: 10% of signal range, min 2px
    sig_range  = float(subj_ys.max() - subj_ys.min())
    prominence = max(2.0, sig_range * 0.10)

    # Contacts = local MAX subj_y; apexes = local MIN subj_y
    # plateau_size=1 ensures equal-height adjacent points are also detected
    contact_idxs, _ = _find_peaks( subj_ys, prominence=prominence, plateau_size=1)
    apex_idxs,    _ = _find_peaks(-subj_ys, prominence=prominence, plateau_size=1)

    if len(contact_idxs) == 0:
        contact_idxs = _np.array([int(_np.argmax(subj_ys))])

    arcs = []
    for ci in contact_idxs:
        # pb: last apex before contact, else min subj_y in pre-segment
        before_apexes = [i for i in apex_idxs if i < ci]
        if before_apexes:
            pi = before_apexes[-1]
            pb = (int(frames[pi]), int(subj_ys[pi]))
        elif ci > 0:
            pi = int(_np.argmin(subj_ys[:ci]))
            pb = (int(frames[pi]), int(subj_ys[pi]))
        else:
            pb = None

        # pa: first apex after contact, else min subj_y in post-segment
        after_apexes = [i for i in apex_idxs if i > ci]
        if after_apexes:
            pi = after_apexes[0]
            pa = (int(frames[pi]), int(subj_ys[pi]))
        elif ci < len(frames) - 1:
            pi = ci + 1 + int(_np.argmin(subj_ys[ci + 1:]))
            pa = (int(frames[pi]), int(subj_ys[pi]))
        else:
            pa = None

        c_frame = int(frames[ci])
        pb_degen = (pb is None) or (pb[0] == c_frame)
        pa_degen = (pa is None) or (pa[0] == c_frame)
        if pb_degen and pa_degen:
            continue  # no usable flanking points — skip entirely

        arcs.append({
            'c_idx':   int(ci),
            'c_frame': c_frame,
            'c_y':     int(subj_ys[ci]),
            'pb':      pb,
            'pa':      pa,
        })
    return arcs


def _check_motion_physics(win_pts, arcs, orig_total=0,
                          stall_ratio=0.05,
                          glitch_std_factor=2.0):
    """
    Check velocity physics per arc using central-difference velocity.

    - Velocity: central difference vy[i] = (y[i+1]-y[i-1])/(f[i+1]-f[i-1])
                first and last positions have no velocity.
    - Per arc, split into descending (vy>0, falling) and ascending (vy<0, rising).
    - Monotonicity: descending should be non-decreasing, ascending non-increasing.
      Violations are glitch candidates.
    - Threshold: median ± glitch_std_factor*std of vy-diffs per side.
    - Stall: |vy| < stall_ratio * median(|vy| in arc).
    - Arcs whose contact is in last 5% of video are skipped.

    Returns:
        (arc_results, vy_list)
        arc_results: list of {c_frame, skip, n_violations, issues}
        vy_list: [(frame, vy), ...]  interior frames only
    """
    import numpy as _np

    if len(win_pts) < 4:
        return [], []

    frames  = _np.array([f for f, _ in win_pts], dtype=float)
    subj_ys = _np.array([y for _, y in win_pts], dtype=float)
    n = len(frames)
    frame_to_idx = {int(frames[i]): i for i in range(n)}

    # Central-difference velocity — interior points only
    vy_cd = _np.full(n, _np.nan)
    for i in range(1, n - 1):
        dt = frames[i + 1] - frames[i - 1]
        if dt > 0:
            vy_cd[i] = (subj_ys[i + 1] - subj_ys[i - 1]) / dt

    vy_list = [(int(frames[i]), round(float(vy_cd[i]), 3))
               for i in range(1, n - 1) if not _np.isnan(vy_cd[i])]

    arc_results = []
    for arc in arcs:
        ci      = arc['c_idx']
        c_frame = arc['c_frame']
        pb      = arc.get('pb')
        pa      = arc.get('pa')

        skip = (orig_total > 0 and c_frame >= orig_total * 0.95)
        if skip:
            arc_results.append({'c_frame': c_frame, 'skip': True,
                                 'n_violations': 0, 'issues': []})
            continue


        pb_idx = frame_to_idx.get(pb[0]) if pb else None
        pa_idx = frame_to_idx.get(pa[0]) if pa else None
        start  = (pb_idx if pb_idx is not None else ci)
        end    = (pa_idx if pa_idx is not None else ci)
        # interior indices with valid vy for this arc segment
        # exclude pb and pa indices themselves (boundary points, not usable for monotonicity)
        pb_pa = {i for i in (pb_idx, pa_idx) if i is not None}
        arc_idxs = [i for i in range(max(1, start), min(n - 1, end + 1))
                    if not _np.isnan(vy_cd[i]) and i not in pb_pa]

        if not arc_idxs:
            arc_results.append({'c_frame': c_frame, 'skip': False,
                                 'n_violations': 0, 'issues': []})
            continue

        vy_arc = _np.array([vy_cd[i] for i in arc_idxs])

        # Split at first non-positive vy: everything before = descending, rest = ascending
        split = next((k for k in range(len(arc_idxs)) if vy_arc[k] <= 0), len(arc_idxs))
        desc_idxs = [arc_idxs[k] for k in range(split)]
        asc_idxs  = [arc_idxs[k] for k in range(split, len(arc_idxs))]

        issues    = []
        n_violations = 0

        # Monotonicity check: gravity should make vy increase in both phases.
        # Any non-increasing step (diff <= 0) is a violation.
        n_violations = 0
        for side_idxs, label in [(desc_idxs, 'desc'), (asc_idxs, 'asc')]:
            if len(side_idxs) < 2:
                continue
            vy_side = _np.array([vy_cd[i] for i in side_idxs])
            diffs   = _np.diff(vy_side)
            for k, d in enumerate(diffs):
                if d <= 0:
                    n_violations += 1
                    issues.append({'type': 'NON_INCREASING', 'frame': int(frames[side_idxs[k + 1]]),
                                   'side': label,
                                   'vy_prev': round(float(vy_side[k]), 3),
                                   'vy_curr': round(float(vy_side[k + 1]), 3),
                                   'diff': round(float(d), 3)})

        n_steps = max(0, len(desc_idxs) - 1) + max(0, len(asc_idxs) - 1)
        arc_results.append({'c_frame': c_frame, 'skip': False,
                             'n_violations': n_violations,
                             'n_steps': n_steps,
                             'issues': issues})

    return arc_results, vy_list


def _fit_parabola(frames, ys):
    """Fit y = a*x^2 + b*x + c to (frames, ys). Returns (a, r2).
    Since subj_y increases downward, a valid bounce arc has subj_y peaking at contact:
      a < 0  → opens downward (∩) = valid bounce arc ✓
      a > 0  → opens upward (∪) = invalid ✗
    r2 close to 1.0 means the arc is well-approximated by a parabola.
    Returns (None, None) if fewer than 3 points.
    """
    import numpy as _np
    if len(frames) < 3:
        return None, None
    x = _np.array(frames, dtype=float)
    y = _np.array(ys, dtype=float)
    coeffs = _np.polyfit(x, y, 2)
    a = coeffs[0]
    y_pred = _np.polyval(coeffs, x)
    ss_res = _np.sum((y - y_pred) ** 2)
    ss_tot = _np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(a), float(r2)


def _save_bounce_plot(win_pts, bounce_idx, video_name, save_dir, arcs=None):
    """Save a subj_y-vs-frame plot for one window showing ALL detected arcs.
    Shows full window in gray, with contact/pb/pa markers for every arc.
    arcs: list of arc dicts from _find_bounce_arcs.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        import numpy as _np

        win_frames = [f for f, _ in win_pts]
        win_ys     = [y for _, y in win_pts]

        fig, ax = _plt.subplots(figsize=(10, 3))
        # Full window in light gray (context)
        ax.plot(win_frames, win_ys, "o-", color="lightgray", linewidth=1, label="window")

        # Mark all arcs
        arc_list = arcs if arcs else []
        colors_contact = ["red", "crimson", "darkred"]
        colors_pb      = ["green", "seagreen", "darkgreen"]
        colors_pa      = ["orange", "darkorange", "goldenrod"]

        for ai, arc in enumerate(arc_list):
            ci   = arc['c_frame']
            ci_y = arc.get('c_y')
            apb  = arc.get('pb')
            apa  = arc.get('pa')
            cc   = colors_contact[min(ai, len(colors_contact) - 1)]
            cp   = colors_pb[min(ai, len(colors_pb) - 1)]
            ca   = colors_pa[min(ai, len(colors_pa) - 1)]
            lbl  = f"arc{ai+1}" if len(arc_list) > 1 else ""

            ax.axvline(ci, color=cc, linestyle="--",
                       label=f"contact{lbl} f={ci}" + (f"(y={ci_y})" if ci_y else ""))
            if apb is not None:
                ax.axvline(apb[0], color=cp, linestyle=":",
                           label=f"pb{lbl} f={apb[0]}(y={apb[1]})")
            if apa is not None:
                ax.axvline(apa[0], color=ca, linestyle="-.",
                           label=f"pa{lbl} f={apa[0]}(y={apa[1]})")

        ax.invert_yaxis()
        ax.set_xlabel("frame")
        ax.set_ylabel("subj_y (px)")
        n_arcs = len(arc_list)
        ax.set_title(f"{video_name}  window {bounce_idx}  ({n_arcs} arc{'s' if n_arcs != 1 else ''} detected)",
                     fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        _plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{video_name}_window{bounce_idx}.png")
        fig.savefig(out_path, dpi=90)
        _plt.close(fig)
        return out_path
    except Exception as exc:
        return f"(plot failed: {exc})"


def _build_video_physics_profile(
    video_path: str,
    sam3_result: Dict,
    grounded_spec: Dict,
    qwen_check: Optional[Dict],
    video_label: str = "?",
    sam3_coarse_fps: float = 3.0,
    plots_dir: Optional[str] = None,
) -> tuple:
    """
    Build a compact text physics profile for tournament LLM reasoning.

    Returns:
        (profile_str, motion_viol, motion_arcs, energy_viol, energy_pairs, rotation_viol, cor_collected, final_settled)
        motion_viol   — total NON_INCREASING vy violations; divide by motion_arcs for rate
        motion_arcs   — non-skipped arcs checked
        energy_viol   — cross-window energy gain violations; divide by energy_pairs for rate
        energy_pairs  — consecutive bounce pairs checked
        rotation_viol — frames with hist > 2.0
        cor_collected — list of (contact_frame, cor_value) for all bounces
        final_settled — whether video ends with subject settled on floor
    """
    import cv2 as _cv2

    # ── Video metadata ───────────────────────────────────────────────────────
    cap = _cv2.VideoCapture(video_path)
    orig_fps   = cap.get(_cv2.CAP_PROP_FPS) or 16.0
    orig_total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_s = orig_total / orig_fps if orig_fps > 0 else 0.0

    subject_name = sam3_result.get("subject_name", "subject")
    entity_names = sam3_result.get("entity_names", [])
    entity_str   = ", ".join(entity_names) if entity_names else "none"
    min_dist     = sam3_result.get("min_contact_distance_px")
    min_dist_str = f"{min_dist:.0f}px" if min_dist is not None else "N/A"

    # ── Action phases from grounded_spec ────────────────────────────────────
    phases = grounded_spec.get("action_phases", {}).get("phases", [])
    phase_lines = []
    for ph in phases:
        name = ph.get("phase_name", "?")
        desc = ph.get("description", "")
        phase_lines.append(f"  [{name}] {desc}")
    phases_str = "\n".join(phase_lines) if phase_lines else "  (none)"

    # ── Coarse trajectory: downsample to ≤20 rows ───────────────────────────
    coarse = sam3_result.get("coarse_trajectory", [])

    # initial_y: subj_y in the first frame the ball is visible (for CoR denominator)
    initial_y = None
    for entry in coarse:
        com = entry.get(f"{subject_name}_center_of_mass")
        if com:
            initial_y = int(com[1])
            break

    traj_rows = []
    step = max(1, len(coarse) // 20)
    for entry in coarse[::step]:
        orig_fi  = entry.get("frame", 0)
        subj_com = entry.get(f"{subject_name}_center_of_mass")
        subj_y   = subj_com[1] if subj_com else None
        ent_y    = (entry.get(f"{entity_names[0]}_top") if entity_names else None)
        dist     = entry.get("contact_distance_px")
        row = f"  f={orig_fi}"
        if subj_y is not None:
            row += f" subj_y={subj_y}"
        if ent_y is not None:
            row += f" ent_top={ent_y}"
        if dist is not None:
            row += f" gap={dist:.0f}px"
        traj_rows.append(row)
    traj_str = "\n".join(traj_rows) if traj_rows else "  (no trajectory data)"

    # ── Key frame analysis — pre-compute all physics metrics ────────────────
    kf_analysis = sam3_result.get("key_frame_analysis", [])
    cv_kf_map = {}
    if qwen_check:
        for kf in qwen_check.get("per_keyframe", []):
            idx = kf.get("key_frame_orig_idx")
            cv_kf_map[idx] = {
                "hist_distances": kf.get("hist_distances", []),
            }

    # Collect per-keyframe physics results
    contact_frames = []
    peak_seq_collected = []  # (contact_frame, peak_frame, peak_subj_y)
    cor_collected = []       # (contact_frame, cor_value)
    kf_lines = []
    seen_contact_frames: set = set()
    motion_viol   = 0   # NON_INCREASING vy violations
    motion_arcs   = 0   # non-skipped arcs checked
    energy_viol   = 0   # cross-window energy gain violations
    energy_pairs  = 0   # consecutive bounce pairs checked
    rotation_viol = 0   # frames with hist > 2.0
    final_settled  = True
    last_bounce_post_pts = []   # full-FPS post_pts from chronologically last bounce (for final state)

    for kf in kf_analysis:
        idx  = kf.get("key_frame_orig_idx", 0)
        win  = kf.get("sliding_window_trajectory", [])

        # Build list of (frame, subj_y) for this window
        win_pts = []
        for e in win:
            com = e.get(f"{subject_name}_center_of_mass")
            if com:
                win_pts.append((e["frame"], int(com[1])))

        if not win_pts:
            kf_lines.append(f"  KEY_FRAME f={idx}: (no window data)")
            continue

        # Use scipy signal processing to find all bounce arcs in this window
        arcs = _find_bounce_arcs(win_pts)
        if not arcs:
            # fallback: single arc using global max subj_y as contact
            c_pos, (c_frame, c_y) = max(enumerate(win_pts), key=lambda t: t[1][1])
            arcs = [{'c_idx': c_pos, 'c_frame': c_frame, 'c_y': c_y,
                     'pb': None, 'pa': None}]

        # ── Hist distances for this window (keyed by key_frame_orig_idx) ──────
        hist_dists = cv_kf_map.get(idx, {}).get("hist_distances", [])
        hist_str = ""
        if hist_dists:
            max_hist = max(hist_dists)
            n_high = sum(1 for d in hist_dists if d > 2.0)
            hist_str = f"hist_change_rate max={max_hist:.3f}"
            if n_high > 0:
                hist_str += f" ⚠ {n_high} frames high (possible artifact)"
                rotation_viol += n_high

        # ── Motion physics check (central-diff vy, stall/glitch per arc) ────────
        arc_motion, vy_list = _check_motion_physics(win_pts, arcs, orig_total=orig_total)
        if vy_list:
            import json as _json
            vy_map = {f: v for f, v in vy_list}
            print(f"   [MOTION][{video_label}] window f={idx}", flush=True)
            for ai, (arc, ar) in enumerate(zip(arcs, arc_motion)):
                pb      = arc.get('pb')
                pa      = arc.get('pa')
                f_start = pb[0] if pb else arc['c_frame']
                f_end   = pa[0] if pa else arc['c_frame']
                arc_report = []
                for f, _ in win_pts:
                    if f_start <= f <= f_end:
                        entry = {'f': f}
                        if f in vy_map: entry['vy'] = vy_map[f]
                        issue_types = [iss['type'] for iss in ar['issues'] if iss.get('frame') == f]
                        if issue_types: entry['issues'] = issue_types
                        arc_report.append(entry)
                skip_str = " [SKIPPED-last10%]" if ar['skip'] else \
                           f" violations={ar['n_violations']}"
                print(f"     arc{ai+1} f{f_start}→c{arc['c_frame']}→f{f_end}{skip_str}: "
                      f"{_json.dumps(arc_report)}", flush=True)
            for ar in arc_motion:
                if not ar['skip']:
                    motion_viol += ar['n_violations']
                    motion_arcs += ar.get('n_steps', 0)



        # ── Save one plot per window showing all arcs ────────────────────────
        win_plot_str = ""
        if plots_dir:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            win_plot_path = _save_bounce_plot(
                win_pts, len(kf_lines) + 1,
                video_name, plots_dir,
                arcs=arcs,
            )
            win_plot_str = f"  window_plot → {win_plot_path}"

        n_arcs = len(arcs)
        for ai, arc in enumerate(arcs):
            c_frame   = arc['c_frame']
            c_y       = arc['c_y']
            pb        = arc['pb']
            pa        = arc['pa']

            # Skip duplicate contact frames across windows
            if c_frame in seen_contact_frames:
                continue
            seen_contact_frames.add(c_frame)

            # Skip arc check if contact is in last 5% of video
            check_rest = (orig_total > 0 and c_frame >= orig_total * 0.95)

            if check_rest:
                arc_str = "arc check skipped (contact in last 5%)"
            else:
                pb_str = f"pb=f{pb[0]}" if pb else "no pb"
                pa_str = f"pa=f{pa[0]}" if pa else "no pa"
                arc_str = f"arc: {pb_str} → contact f={c_frame} → {pa_str}"

            peak_before_str = f"peak_before f={pb[0]} subj_y={pb[1]}" if pb else "peak_before: not found"
            if pa is not None and not check_rest:
                peak_seq_collected.append((c_frame, pa[0], pa[1]))
                if pb is not None and (c_y - pb[1]) > 0:
                    cor = (c_y - pa[1]) / (c_y - pb[1])
                    cor_str = f"CoR={cor:.2f} (pb={pb[1]}, pa={pa[1]})"
                elif initial_y is not None and c_y > initial_y:
                    cor = (c_y - pa[1]) / (c_y - initial_y)
                    cor_str = f"CoR={cor:.2f} (initial_y={initial_y})"
                else:
                    cor = (c_y - pa[1]) / c_y if c_y > 0 else 0.0
                    cor_str = f"CoR={cor:.2f} (fallback)"
                # pa subj_y <= pb subj_y means ball rebounded to same height or higher (CoR >= 1)
                if pb is not None and pa[1] <= pb[1]:
                    cor_str += " ⚠ pa≤pb (impossible rebound)"
                peak_after_str = f"peak_after f={pa[0]} subj_y={pa[1]}, {cor_str}"
                cor_collected.append((c_frame, round(cor, 3)))
            else:
                peak_after_str = "peak_after: not found"

            contact_frames.append((c_frame, c_y))


            kf_line = (
                f"  BOUNCE f={c_frame} (contact subj_y={c_y})\n"
                f"    {arc_str}\n"
                f"    {peak_before_str}\n"
                f"    {peak_after_str}"
            )
            if hist_str:
                kf_line += f"\n    {hist_str}"
            kf_lines.append(kf_line)

        if win_plot_str:
            kf_lines.append(win_plot_str)

    # ── Cross-window energy decay check ──────────────────────────────────────
    # Aggregate all peak_after subj_y values sorted by contact frame.
    # peak_after subj_y must INCREASE across bounces (ball rebounds less = energy decay).
    # Skip first and last bounce (edge effects) — only check middle transitions.
    if len(peak_seq_collected) >= 2:
        peak_seq = sorted(peak_seq_collected, key=lambda t: t[0])
        peak_summary = "  Cross-window rebound peaks (subj_y must INCREASE = less rebound):\n"
        energy_violations = []
        # Check all consecutive pairs (including first/last — cross-window is global view)
        for i, (cf, pf, py) in enumerate(peak_seq):
            tag = ""
            if i > 0:
                energy_pairs += 1
                if py <= peak_seq[i - 1][2]:
                    tag = " ✗ ENERGY GAIN"
                    energy_violations.append(f"contact_f={cf}")
                    energy_viol += 1
                else:
                    tag = " ✓"
            peak_summary += f"    bounce {i+1}: contact_f={cf} peak_after_f={pf} peak_after_y={py}{tag}\n"
        if energy_violations:
            peak_summary += f"  ⚠ Energy gain at: {', '.join(energy_violations)} (energy violations)"
        else:
            peak_summary += "  ✓ Energy dissipating correctly"
    elif len(peak_seq_collected) == 1:
        peak_summary = "  (only one bounce — cannot check cross-window decay)"
    else:
        peak_summary = "  (no rebound peaks detected)"

    kf_str = "\n".join(kf_lines) if kf_lines else "  (no key frames)"

    # ── Final state check using last_phase_trajectory (full-FPS from last contact → end) ──
    # last_phase_trajectory is populated by run_sam3_contact Pass 3.
    # Use the last 10 frames of it (true end of video).
    final_state_str = "(final state: skipped — no last_phase_trajectory or contacts)"
    last_phase = sam3_result.get("last_phase_trajectory", [])
    last_phase_pts = [(e.get("frame", 0), int(e[f"{subject_name}_center_of_mass"][1]))
                      for e in last_phase if e.get(f"{subject_name}_center_of_mass")]
    if contact_frames and last_phase_pts:
        # floor_y = max subj_y across all contacts = lowest position = floor level
        floor_y   = max(y for _, y in contact_frames)
        check_pts = last_phase_pts
        final_ys  = [y for _, y in check_pts]
        y_range   = max(final_ys) - min(final_ys)
        pos_threshold  = max(10, int(floor_y * 0.10))
        # Settled = last-phase frames are near the floor level AND not bouncing much
        near_floor = all(abs(y - floor_y) <= pos_threshold for y in final_ys)
        low_bounce = y_range <= pos_threshold
        final_ok   = near_floor and low_bounce
        f_start = check_pts[0][0]
        f_end   = check_pts[-1][0]
        final_state_str = (
            f"last {len(check_pts)} last-phase frames (f={f_start}–{f_end}): "
            f"subj_y range [{min(final_ys)}–{max(final_ys)}] (spread={y_range}px) "
            f"vs floor_y={floor_y} (threshold±{pos_threshold}px)"
            + (" → SETTLED ✓" if final_ok else " → NOT SETTLED ✗")
            + ("" if near_floor else " [not near floor]")
            + ("" if low_bounce else " [still bouncing]")
        )
        final_settled = final_ok

    # Build a compact LLM-facing summary: only CoR + hist per bounce (kinematic checks already done)
    llm_bounce_lines = []
    for kf_line in kf_lines:
        # Extract just the lines with peak_before, peak_after/CoR, and hist
        relevant = [l.strip() for l in kf_line.split("\n")
                    if any(tok in l for tok in
                           ["BOUNCE f=", "peak_before", "peak_after", "CoR", "hist_change_rate",
                            "AT REST", "rest check"])]
        llm_bounce_lines.append("\n  ".join(relevant))
    llm_bounces_str = "\n".join(llm_bounce_lines) if llm_bounce_lines else "  (no bounces)"

    cor_summary = ("  " + "  ".join(f"bounce_f={cf} CoR={cv}" for cf, cv in
                   sorted(cor_collected, key=lambda t: t[0]))) if cor_collected else "  (no CoR data)"

    profile_str = (
        f"=== VIDEO {video_label}: {os.path.basename(video_path)} ===\n"
        f"duration={duration_s:.2f}s  initial_y={initial_y}  min_contact_gap={min_dist_str}\n"
        f"\nBounces (CoR + artifact check):\n"
        f"{llm_bounces_str}\n"
        f"\nCoR per bounce:\n{cor_summary}\n"
    )

    # ── Debug: always print full kinematic profile + violation count ──────────
    motion_rate = f"{motion_viol/motion_arcs:.2f}" if motion_arcs > 0 else "n/a"
    energy_rate = f"{energy_viol/energy_pairs:.2f}" if energy_pairs > 0 else "n/a"
    full_profile = (
        f"=== [DEBUG] VIDEO {video_label}: {os.path.basename(video_path)} ===\n"
        f"motion_viol={motion_viol}/{motion_arcs}arcs(rate={motion_rate})  "
        f"energy_viol={energy_viol}/{energy_pairs}pairs(rate={energy_rate})  "
        f"rotation_viol={rotation_viol}  cor_bounces={len(cor_collected)}\n"
        f"total_frames={orig_total}  fps={orig_fps:.1f}  duration={duration_s:.2f}s\n"
        f"initial_y={initial_y}  min_contact_gap={min_dist_str}\n"
        f"\nPer-bounce physics:\n{kf_str}\n"
        f"\nCoR per bounce:\n{cor_summary}\n"
        f"\nEnergy dissipation:\n{peak_summary}\n"
        f"\nFinal state:\n  {final_state_str}\n"
    )
    print(full_profile, flush=True)

    return profile_str, motion_viol, motion_arcs, energy_viol, energy_pairs, rotation_viol, cor_collected, final_settled


def run_tournament_selection(
    survivors: List[str],
    passed_video_data: Dict[str, Dict],
    tournament_model_name: str = "Qwen/Qwen3-8B",
    batch_size: int = 3,
    plots_dir: Optional[str] = None,
) -> List[str]:
    """
    Stage 1c: Score-based physical plausibility selection.

    Uses Qwen3-8B once to determine expected CoR range for the material pair,
    then scores each video deterministically:
      - CoR out of range → deduct points (skip last-bounce CoR, expected to be low)
      - CoR > 1 → hard deduction per bounce
      - soft_violations → deduct points
      - hist_change_rate > 1.5 → deduct points per bounce
    Highest score wins. No tournament brackets.

    Args:
        survivors: List of video paths that passed Stage 1b
        passed_video_data: {video_path: {"sam3": sam3_result, "spec": grounded_spec,
                                         "qwen": qwen_check}}
        tournament_model_name: Causal LM to use (default: Qwen/Qwen3-8B)

    Returns:
        List with single winner path (or all survivors if ≤1 or load fails)
    """
    print(f"\n{'='*80}")
    print(f"STAGE 1c: Score-based Selection ({len(survivors)} candidates)")
    print(f"{'='*80}")

    if len(survivors) <= 1:
        print(f"   Only {len(survivors)} survivor(s) — skipping selection")
        return survivors

    # ── Extract material/event info from spec ────────────────────────────────
    first_spec    = passed_video_data.get(survivors[0], {}).get("spec", {})
    event_desc    = (first_spec.get("action_description", "")
                     or first_spec.get("prompt", "")
                     or first_spec.get("text_prompt", ""))
    subj_ent      = first_spec.get("entities", {}).get("subject", {})
    subject_name  = subj_ent.get("name", "subject")
    subj_mat      = subj_ent.get("material", {})
    subj_mat_type = subj_mat.get("type", "unknown")
    subj_mat_props= ", ".join(subj_mat.get("properties", []))
    interact_ents = first_spec.get("entities", {}).get("interactive_entities", [])
    entity_descs  = [
        f"{e.get('name','entity')} ({e.get('material',{}).get('type','unknown')}, "
        f"{', '.join(e.get('material',{}).get('properties',[]))})"
        for e in interact_ents
    ]
    entity_desc_str = "; ".join(entity_descs) if entity_descs else "unknown surface"

    # ── One LLM call: get expected CoR range + hist_change_rate range ────────
    import torch as _torch
    import gc as _gc
    cor_min, cor_max   = 0.3, 0.9   # fallback
    hist_max_expected  = 1.0        # fallback: above this is abnormal rotation

    llm_model, llm_tok = None, None
    for gpu_id in range(_torch.cuda.device_count()):
        try:
            llm_model, llm_tok = load_tournament_model(
                tournament_model_name, device=f"cuda:{gpu_id}")
            print(f"   ✓ Scoring LLM loaded on cuda:{gpu_id}")
            break
        except Exception as exc:
            print(f"   ⚠️  cuda:{gpu_id} failed: {exc}")

    if llm_model is not None:
        physics_prompt = f"""You are a physics expert. Given this bouncing event, determine the expected physical ranges.

Event: "{event_desc}"
Subject: {subject_name} — material: {subj_mat_type} ({subj_mat_props})
Interactive entities: {entity_desc_str}

Answer these two questions based on the materials and the event:
1. What is the expected Coefficient of Restitution (CoR) range for the subject bouncing off the surface?
2. What is the expected maximum hist_change_rate (per-frame colour histogram change, 0–2 scale) for this subject during bouncing?
   - A colourful spinning rubber ball → higher hist (~0.3–0.8)
   - A plain heavy metal ball → near zero hist (~0.0–0.1)
   - Flickering/identity-swap artefact → hist > 1.5 (always abnormal)

Respond with ONLY valid JSON, no markdown:
{{"cor_min": <float>, "cor_max": <float>, "hist_max_normal": <float>, "reasoning": "<one sentence per field>"}}"""

        try:
            raw = _llm_infer_causal(llm_model, llm_tok, physics_prompt, max_new_tokens=512)
            think_match = re.search(r'<think>.*?</think>', raw, re.DOTALL)
            raw_json = raw[think_match.end():].strip() if think_match else raw
            if think_match:
                print(f"   [LLM reasoning]: {think_match.group(0)[:200]}...")
            m = re.search(r'\{[^}]+\}', raw_json, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                cor_min          = float(parsed.get("cor_min", cor_min))
                cor_max          = float(parsed.get("cor_max", cor_max))
                hist_max_expected= float(parsed.get("hist_max_normal", hist_max_expected))
                print(f"   LLM physics ranges — CoR: [{cor_min:.2f}, {cor_max:.2f}]  "
                      f"hist_max_normal: {hist_max_expected:.2f}  "
                      f"({parsed.get('reasoning','')})")
        except Exception as exc:
            print(f"   ⚠️  Physics range LLM failed ({exc}) — using fallbacks")

        del llm_model, llm_tok
        _gc.collect()
        _torch.cuda.empty_cache()
    else:
        print(f"   ⚠️  No LLM available — using fallback ranges")

    # ── Build physics profile + score each video ─────────────────────────────
    # Scoring (start at 100, deduct for violations):
    #   CoR > 1.0              : -10 per bounce  (physically impossible)
    #   CoR outside [min,max]  : -5  per bounce  (skip last bounce — settling CoR expected low)
    #   hist > hist_max_normal : -4  per bounce  (abnormal rotation / flickering)
    #   soft_violations        : -3  per count
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    scored = []

    for i, vp in enumerate(survivors):
        lbl  = labels[i % len(labels)]
        data = passed_video_data.get(vp, {})
        sam3 = data.get("sam3", {})
        spec = data.get("spec", {})
        qwen = data.get("qwen")
        _, motion_viol, motion_arcs, energy_viol, energy_pairs, rotation_viol, cor_collected, final_settled = _build_video_physics_profile(
            vp, sam3, spec, qwen, video_label=lbl, plots_dir=plots_dir)

        score      = 100
        deductions = []

        # ── CoR scoring (rate-based, max per category) ───────────────────────
        # Each category: rate = n_violations / n_bounces, deduct max * rate
        # CoR>2: max -20, CoR>1: max -15, out-of-range: max -10
        n_cor = max(len(cor_collected), 1)
        n_severe   = sum(1 for _, cor in cor_collected if cor > 2.0)
        n_impossi  = sum(1 for _, cor in cor_collected if 1.0 < cor <= 2.0)
        n_outrange = sum(1 for _, cor in cor_collected if cor <= 1.0 and (cor < cor_min or cor > cor_max))
        if n_severe > 0:
            ded = round(n_severe / n_cor * 40)
            score -= ded
            deductions.append(f"CoR>2: {n_severe}/{n_cor} bounces(-{ded})")
        if n_impossi > 0:
            ded = round(n_impossi / n_cor * 25)
            score -= ded
            deductions.append(f"CoR>1: {n_impossi}/{n_cor} bounces(-{ded})")
        if n_outrange > 0:
            ded = round(n_outrange / n_cor * 15)
            score -= ded
            deductions.append(f"CoR out-of-range: {n_outrange}/{n_cor} bounces(-{ded})")

        # ── Rotation/hist scoring (-5 each) ──────────────────────────────────
        if rotation_viol > 0:
            score -= rotation_viol * 5
            deductions.append(f"rotation_viol={rotation_viol}(-{rotation_viol*5})")

        # ── Energy violations: rate-based (max -25) ───────────────────────────
        if energy_pairs > 0:
            energy_rate = energy_viol / energy_pairs
            ded = round(energy_rate * 25)
            if ded > 0:
                score -= ded
                deductions.append(f"energy_viol={energy_viol}/{energy_pairs}(rate={energy_rate:.2f}, -{ded})")

        # ── Motion violations: rate-based (max -20) ───────────────────────────
        if motion_arcs > 0:
            motion_rate = motion_viol / motion_arcs
            ded = round(motion_rate * 20)
            if ded > 0:
                score -= ded
                deductions.append(f"motion_viol={motion_viol}/{motion_arcs}(rate={motion_rate:.2f}, -{ded})")

        # ── Final state (-15 flat) ────────────────────────────────────────────
        if not final_settled:
            score -= 15
            deductions.append("not_settled(-15)")

        ded_str = ", ".join(deductions) if deductions else "none"
        print(f"   [{lbl}] {os.path.basename(vp)}  score={score}  deductions: {ded_str}")
        scored.append((score, vp, lbl))

    # ── Pick winner ──────────────────────────────────────────────────────────
    scored.sort(key=lambda t: -t[0])
    winner_score, final_winner, winner_lbl = scored[0]
    print(f"\n{'─'*80}")
    print(f"SELECTION COMPLETE: 🏆 [{winner_lbl}] {os.path.basename(final_winner)}  score={winner_score}")
    for sc, vp, lbl in scored[1:]:
        print(f"   ✗ [{lbl}] {os.path.basename(vp)}  score={sc}")
    print(f"{'─'*80}")

    return [final_winner]


def verify_subject_in_video(
    video_path: str,
    grounded_spec: Dict,
    model,
    processor,
    device: str,
    num_frames: int = 5,
) -> Tuple[bool, str]:
    """
    Sample 5 frames evenly from the first 20% of the video and verify subject identity
    and initial action phase match the grounded spec.
    Returns (passed, reason).
    """
    import cv2 as _cv2
    import tempfile as _tempfile
    import shutil as _shutil
    import torch as _torch
    import re as _re2

    entities   = grounded_spec.get("entities", {})
    subject    = entities.get("subject", {})
    subj_name  = subject.get("name", "subject")

    geom       = subject.get("geometry", {})
    geom_type  = geom.get("type", "")
    geom_chars = ", ".join(geom.get("characteristics", []))
    appear     = subject.get("appearance", {})
    typ_colors = ", ".join(appear.get("typical_colors", []))
    mat        = subject.get("material", {})
    mat_type   = mat.get("type", "")
    mat_props  = ", ".join(mat.get("properties", []))

    ent_lines = []
    for ent in entities.get("interactive_entities", []):
        eg = ent.get("geometry", {})
        ea = ent.get("appearance", {})
        ent_lines.append(
            f"  - '{ent.get('name')}': shape={eg.get('type','?')} "
            f"({', '.join(eg.get('characteristics',[]))}), "
            f"typical_colors={', '.join(ea.get('typical_colors',[]))}"
        )
    ent_block = "\n".join(ent_lines) if ent_lines else "  (none)"
    ent_names = ", ".join(e.get("name", "") for e in entities.get("interactive_entities", []))

    phases = grounded_spec.get("action_phases", {}).get("phases", [])
    initial_phase = next(
        (p.get("description", "") for p in phases if p.get("phase_name") == "initial"), "")

    # Sample num_frames evenly from the first 10% of the video
    cap       = _cv2.VideoCapture(video_path)
    total     = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT)) or 1
    end_frame = max(num_frames, int(total * 0.10))
    step      = max(1, end_frame // num_frames)
    indices   = [i * step for i in range(num_frames) if i * step < total]

    tmp_dir     = _tempfile.mkdtemp(prefix="subj_verify_")
    frame_paths = []
    for idx in indices:
        cap.set(_cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        p = os.path.join(tmp_dir, f"f{idx:05d}.jpg")
        _cv2.imwrite(p, frame)
        frame_paths.append(p)
    cap.release()

    if not frame_paths:
        _shutil.rmtree(tmp_dir, ignore_errors=True)
        return False, "Could not extract frames from video"

    mat_line = f"Material: {mat_type} ({mat_props})" if mat_type else ""

    prompt = "\n".join(filter(None, [
        "You are a strict quality-control inspector for AI-generated videos. Your job is to FIND FAILURES.",
        "Default to false when in doubt. Only mark true when you are clearly confident.",
        f"You are shown {len(frame_paths)} frames sampled evenly from the first ~10% of the video, in temporal order "
        f"(Image 1 = earliest, Image {len(frame_paths)} = latest shown).",
        "Use the first 2 images to judge the initial phase. Use all frames together to judge geometry, color, material, and background.",
        "",
        "=== SUBJECT SPECIFICATION ===",
        f"Name: {subj_name}",
        f"Geometry: {geom_type} ({geom_chars})",
        f"Expected colors: {typ_colors}",
        mat_line,
        "",
        "=== INTERACTIVE ENTITIES ===",
        ent_block,
        "",
        "=== EXPECTED INITIAL PHASE ===",
        f"{initial_phase}",
        "",
        "=== CHECKS (answer each with true/false) ===",
        f"1. subject_present: Is '{subj_name}' CLEARLY and UNAMBIGUOUSLY visible?",
        "   → false if: subject is absent, barely visible, heavily occluded, or wrong object is shown.",
        f"2. geometry_matches: Does the shape EXACTLY match '{geom_type}' ({geom_chars})?",
        "   → false if: wrong shape (e.g. cube instead of sphere), clearly deformed, or shape is ambiguous.",
        f"3. color_matches: Are the colors a CLOSE match to '{typ_colors}'?",
        "   → false if: completely wrong color family (e.g. blue when red expected), or grayscale when color expected.",
        "   → true if colors are approximate or slightly off-shade.",
        f"4. entities_present: Are ALL of these entities clearly visible: {ent_names}?",
        "   → false if any listed entity is absent or clearly wrong.",
        f"5. initial_phase_matches: Does the scene show the BEGINNING of the action: '{initial_phase}'?",
        "   → true if: ball is falling, in early bounce, or just released — the action is clearly just starting.",
        "   → true if: the ball is mid-air or just making first contact with the floor.",
        "   → false ONLY if: the video clearly starts AFTER the action is complete — ball already at final rest, or multiple bounces already done.",
        (f"6. material_matches: Does the subject visually appear to be made of '{mat_type}' ({mat_props})?"
         if mat_type else None),
        (f"   → true if surface texture, sheen, and deformation cues match '{mat_type}' material."
         if mat_type else None),
        "   → false if: a rubber ball looks like metal/glass, or a rigid object shows jelly-like deformation." if mat_type else None,
        (f"{'7' if mat_type else '6'}. background_stable: Is the background consistent across all frames?"),
        "   → false if: severe flickering, scene cuts, camera wildly panning, background objects teleporting.",
        "   → true if only minor lighting changes.",
        (f"{'8' if mat_type else '7'}. observation: One paragraph — describe the subject shape, color, and material appearance you see,"),
        "   which entities are visible, what phase the scene appears to be in, and background stability.",
        "   Be specific: mention actual colors/shapes/material cues observed vs expected.",
        "",
        "Respond ONLY with a valid JSON object, no markdown fences.",
        (
            "Example (failing video):\n"
            "{\n"
            '  "subject_present": true,\n'
            '  "geometry_matches": false,\n'
            '  "color_matches": true,\n'
            '  "entities_present": true,\n'
            '  "initial_phase_matches": false,\n'
            '  "material_matches": true,\n'
            '  "background_stable": true,\n'
            '  "observation": "A red object is visible but it appears cubic rather than spherical, failing geometry. '
            'Colors are red as expected. The floor is present. The ball is already completely at rest on the floor '
            'with no motion — the action is finished, this is the final resting state not the start. '
            'Surface looks rubbery with slight sheen, consistent with rubber material. '
            'Background is a static room with no instability."\n'
            "}"
        ),
    ]))

    # Send frames as individual images (not video) so each frame gets full token budget
    # Use absolute paths directly — apply_chat_template does not support file:// URIs
    content = [{"type": "image", "image": os.path.abspath(p)} for p in frame_paths]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    with _torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0].strip()
    print(f"   [SubjectVerify] raw:\n{raw}", flush=True)

    _shutil.rmtree(tmp_dir, ignore_errors=True)

    try:
        m       = _re2.search(r'\{.*\}', raw, _re2.DOTALL)
        parsed  = json.loads(m.group()) if m else {}
        present      = bool(parsed.get("subject_present", False))
        geom_ok      = bool(parsed.get("geometry_matches", False))
        color_ok     = bool(parsed.get("color_matches", False))
        ents_ok      = bool(parsed.get("entities_present", False))
        init_ok      = bool(parsed.get("initial_phase_matches", False))
        mat_ok       = bool(parsed.get("material_matches", True))   # default True if not in spec
        bg_stable    = bool(parsed.get("background_stable", True))
        obs          = str(parsed.get("observation", ""))
    except Exception as exc:
        present = geom_ok = color_ok = ents_ok = init_ok = mat_ok = bg_stable = True
        obs = f"parse error: {exc}"

    # Hard checks: all fields
    checks = [
        ("subject_present",   present),
        ("geometry",          geom_ok),
        ("color",             color_ok),
        ("entities",          ents_ok),
        ("initial_phase",     init_ok),
        ("background_stable", bg_stable),
    ]
    if mat_type:
        checks.append(("material", mat_ok))

    failed = [k for k, v in checks if not v]

    if failed:
        return False, f"failed={failed} | {obs}"
    return True, f"all passed | {obs}"


def _cv_color_block_tracking(crop_paths: List[str],
                              n_colors: int = 4,
                              min_blob_area: int = 100) -> List[Dict]:
    """
    For each frame in the masked crop sequence, detect dominant colour blobs and
    record their centroids + areas. Track movement frame-to-frame.

    Returns a list of per-frame dicts:
      [{"frame": i+1,
        "blobs": [{"color": "red", "hue_deg": 5, "centroid_x": 45, "centroid_y": 30,
                   "area_px": 1200, "area_pct": 42.1}],
        "movement": [{"color": "red", "dx": 3, "dy": -5, "dist": 5.8}]  # vs prev frame
       }, ...]
    """
    import cv2 as _cv2
    import numpy as _np

    # HSV hue ranges for named colours (hue in 0-180 OpenCV scale)
    COLOR_RANGES = [
        ("red",          [  0,  10], [160, 180]),
        ("orange",       [ 10,  25]),
        ("yellow",       [ 25,  35]),
        ("yellow-green", [ 35,  50]),
        ("green",        [ 50,  75]),
        ("cyan",         [ 75,  95]),
        ("blue",         [ 95, 130]),
        ("violet",       [130, 145]),
        ("magenta",      [145, 160]),
    ]

    def _dominant_blobs(img_bgr):
        """Return list of detected blobs sorted by area desc."""
        hsv = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2HSV)
        fg_mask = _cv2.inRange(hsv, (0, 30, 30), (180, 255, 255))
        total_fg = float(fg_mask.sum() // 255) or 1.0
        blobs = []
        for entry in COLOR_RANGES:
            name = entry[0]
            ranges = entry[1:]  # one or two [lo, hi] pairs (red wraps)
            color_mask = _np.zeros(hsv.shape[:2], dtype=_np.uint8)
            for lo, hi in ranges:
                color_mask |= _cv2.inRange(hsv, (lo, 40, 40), (hi, 255, 255))
            color_mask &= fg_mask
            # Morphological cleanup
            kernel = _np.ones((3, 3), _np.uint8)
            color_mask = _cv2.morphologyEx(color_mask, _cv2.MORPH_OPEN, kernel)
            cnts, _ = _cv2.findContours(color_mask, _cv2.RETR_EXTERNAL,
                                         _cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = _cv2.contourArea(c)
                if area < min_blob_area:
                    continue
                M = _cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                hue_mean = int(hsv[color_mask > 0, 0].mean()) * 2 if color_mask.any() else 0
                blobs.append({
                    "color":      name,
                    "hue_deg":    hue_mean,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "area_px":    int(area),
                    "area_pct":   round(area / total_fg * 100, 1),
                })
        # Sort by area desc, keep top n_colors
        blobs.sort(key=lambda b: b["area_px"], reverse=True)
        return blobs[:n_colors]

    per_frame = []
    prev_blobs = None
    for i, p in enumerate(crop_paths):
        img = _cv2.imread(p)
        if img is None:
            per_frame.append({"frame": i + 1, "blobs": [], "movement": []})
            prev_blobs = None
            continue

        blobs = _dominant_blobs(img)

        # Compute movement vs previous frame by matching on colour name
        movement = []
        if prev_blobs:
            prev_map = {b["color"]: b for b in prev_blobs}
            for b in blobs:
                pb = prev_map.get(b["color"])
                if pb:
                    dx = b["centroid_x"] - pb["centroid_x"]
                    dy = b["centroid_y"] - pb["centroid_y"]
                    dist = round(float(_np.hypot(dx, dy)), 1)
                    movement.append({
                        "color": b["color"],
                        "dx": dx, "dy": dy, "dist": dist,
                    })

        per_frame.append({"frame": i + 1, "blobs": blobs, "movement": movement})
        prev_blobs = blobs

    return per_frame


def _save_color_track_plot(color_block_track: List[Dict],
                            hist_distances: List[float],
                            frame_indices: List[int],
                            kf_idx: int,
                            save_path: str,
                            video_name: str = "") -> None:
    """
    Save a two-panel figure for one keyframe window:
      Top: centroid trajectories per colour across frames (pixel space)
      Bottom: per-pair histogram distance bar chart
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt

        COLOR_HEX = {
            "red": "#e53935", "orange": "#fb8c00", "yellow": "#fdd835",
            "yellow-green": "#7cb342", "green": "#43a047", "cyan": "#00acc1",
            "blue": "#1e88e5", "violet": "#8e24aa", "magenta": "#d81b60",
        }

        fig, (ax_top, ax_bot) = _plt.subplots(2, 1, figsize=(10, 7),
                                               gridspec_kw={"height_ratios": [2, 1]})
        title = f"{video_name}\nkf={kf_idx}  colour block tracking" if video_name else f"kf={kf_idx}  colour block tracking"
        fig.suptitle(title, fontsize=9, wrap=True)

        # ── Top: centroid trails ──────────────────────────────────────────────
        # Collect per-colour trajectory
        trails = {}  # color → [(frame_label, cx, cy), ...]
        for fe in color_block_track:
            fi = fe.get("frame", "?")
            gfi = frame_indices[fi - 1] if (frame_indices and fi - 1 < len(frame_indices)) else fi
            for b in fe.get("blobs", []):
                c = b["color"]
                trails.setdefault(c, []).append((gfi, b["centroid_x"], b["centroid_y"], b["area_pct"]))

        for color, pts in trails.items():
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            labels = [str(p[0]) for p in pts]
            hex_c = COLOR_HEX.get(color, "#888888")
            ax_top.plot(xs, ys, "-o", color=hex_c, linewidth=1.5, markersize=5, label=color)
            for x, y, lbl in zip(xs, ys, labels):
                ax_top.annotate(f"f{lbl}", (x, y), textcoords="offset points",
                                xytext=(4, 4), fontsize=6, color=hex_c)

        ax_top.set_xlabel("centroid_x (px)", fontsize=8)
        ax_top.set_ylabel("centroid_y (px)", fontsize=8)
        ax_top.invert_yaxis()   # image coords: y increases downward
        ax_top.legend(fontsize=7, loc="upper right")
        ax_top.set_title("centroid movement (image pixel coords, y↓)", fontsize=9)
        ax_top.grid(True, linewidth=0.3, alpha=0.5)

        # ── Bottom: histogram distances ──────────────────────────────────────
        if hist_distances and len(frame_indices) >= 2:
            pair_labels = [f"f{frame_indices[i]}→f{frame_indices[i+1]}"
                           for i in range(len(hist_distances))]
        else:
            pair_labels = [str(i + 1) for i in range(len(hist_distances))]

        bar_colors = ["#e53935" if d >= 2.5 else "#1e88e5" for d in hist_distances]
        ax_bot.bar(pair_labels, hist_distances, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax_bot.axhline(y=2.5, color="red", linestyle="--", linewidth=1, label="threshold=2.5")
        ax_bot.set_ylabel("chi-sq dist", fontsize=8)
        ax_bot.set_title("HSV histogram distance per frame pair", fontsize=9)
        ax_bot.legend(fontsize=7)
        ax_bot.tick_params(axis="x", labelsize=7)
        ax_bot.grid(True, axis="y", linewidth=0.3, alpha=0.5)

        _plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        _plt.savefig(save_path, dpi=100, bbox_inches="tight")
        _plt.close(fig)
        print(f"   [CV-KF] saved colour track plot → {save_path}", flush=True)
    except Exception as exc:
        print(f"   [CV-KF] plot save failed: {exc}", flush=True)


def _cv_histogram_consistency(crop_paths: List[str],
                               hist_spike_thresh: float = 2.5,
                               area_ratio_thresh: float = 3.0,
                               frame_indices: List[int] = None) -> Dict:
    """
    OpenCV-based frame consistency check on masked crop image sequence.

    Metrics:
    - HSV histogram chi-squared distance between consecutive frames:
        gradual change (rotation) → low distance; sudden morph/flicker → high spike
    - Contour area ratio between consecutive frames:
        consistent object → ratio near 1.0; object exploding/vanishing → large ratio

    Returns:
      {"passed": bool, "confidence": float, "max_hist_dist": float,
       "max_area_ratio": float, "spike_frames": [...], "details": str}
    """
    import cv2 as _cv2
    import numpy as _np

    if len(crop_paths) < 2:
        return {"passed": True, "confidence": 1.0, "max_hist_dist": 0.0,
                "max_area_ratio": 1.0, "spike_frames": [], "details": "single frame — skip"}

    # Hue name lookup (18 buckets × 10° each)
    _HUE_NAMES = ["red","orange","yellow","yellow-green","green","green-cyan",
                  "cyan","cyan-blue","blue","blue-violet","violet","magenta",
                  "magenta-red","red2","red3","red4","red5","red6"]

    hists, areas, dominant_hues = [], [], []
    for p in crop_paths:
        img = _cv2.imread(p)
        if img is None:
            hists.append(None); areas.append(None); dominant_hues.append(None)
            continue
        hsv = _cv2.cvtColor(img, _cv2.COLOR_BGR2HSV)
        mask = _cv2.inRange(hsv, (0, 30, 30), (180, 255, 255))
        # Full hue histogram for chi-sq distance
        hist = _cv2.calcHist([hsv], [0, 1], mask, [18, 16], [0, 180, 0, 256])
        _cv2.normalize(hist, hist)
        hists.append(hist)
        # Contour area
        cnts, _ = _cv2.findContours(mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        areas.append(sum(_cv2.contourArea(c) for c in cnts) if cnts else 0.0)
        # Dominant hues: top-3 hue buckets by marginal sum
        hue_marginal = _cv2.calcHist([hsv], [0], mask, [18], [0, 180]).flatten()
        total_px = float(hue_marginal.sum()) or 1.0
        top3 = _np.argsort(hue_marginal)[::-1][:3]
        dominant_hues.append(
            [(f"{_HUE_NAMES[b]}", round(float(hue_marginal[b]) / total_px * 100, 1))
             for b in top3 if hue_marginal[b] > 0])

    hist_dists, area_ratios, spike_frames = [], [], []
    for i in range(1, len(crop_paths)):
        h1, h2 = hists[i - 1], hists[i]
        a1, a2 = areas[i - 1], areas[i]

        if h1 is not None and h2 is not None:
            d = _cv2.compareHist(h1, h2, _cv2.HISTCMP_CHISQR_ALT)
            hist_dists.append(d)
            if d > hist_spike_thresh:
                # Report as "fSRC→fDST" pair using global video frame indices
                src_fi = frame_indices[i - 1] if (frame_indices and i - 1 < len(frame_indices)) else i
                dst_fi = frame_indices[i]     if (frame_indices and i     < len(frame_indices)) else (i + 1)
                spike_frames.append(f"f{src_fi}→f{dst_fi}")
        else:
            hist_dists.append(0.0)

        if a1 and a2 and a1 > 0:
            r = max(a1, a2) / max(min(a1, a2), 1.0)
            area_ratios.append(r)
        else:
            area_ratios.append(1.0)

    max_dist  = max(hist_dists)  if hist_dists  else 0.0
    max_ratio = max(area_ratios) if area_ratios else 1.0

    hist_ok  = max_dist  < hist_spike_thresh
    shape_ok = max_ratio < area_ratio_thresh
    passed   = hist_ok and shape_ok

    conf = max(0.0, 1.0 - (max_dist / hist_spike_thresh)) if not hist_ok else \
           max(0.0, 1.0 - (max_dist / (hist_spike_thresh * 2)))

    # Per-frame colour distribution summary (top-2 hues per frame, using global indices)
    colour_dist_lines = []
    for fi, hues in enumerate(dominant_hues):
        if hues is None:
            continue
        gfi = frame_indices[fi] if (frame_indices and fi < len(frame_indices)) else (fi + 1)
        hue_str = ", ".join(f"{name} {pct}%" for name, pct in hues[:2])
        colour_dist_lines.append(f"f{gfi}:[{hue_str}]")
    colour_dist_str = "  colour: " + "  ".join(colour_dist_lines) if colour_dist_lines else ""

    details_parts = []
    if not hist_ok:
        details_parts.append(
            f"colour spike at frames {spike_frames} (max_dist={max_dist:.3f} > {hist_spike_thresh})")
    if not shape_ok:
        details_parts.append(
            f"shape discontinuity (max_area_ratio={max_ratio:.1f} > {area_ratio_thresh})")
    if passed:
        details_parts.append(
            f"consistent (max_hist_dist={max_dist:.3f}, max_area_ratio={max_ratio:.1f})")
    if colour_dist_str:
        details_parts.append(colour_dist_str)

    return {
        "passed":          passed,
        "confidence":      conf,
        "max_hist_dist":   max_dist,
        "max_area_ratio":  max_ratio,
        "spike_frames":    spike_frames,
        "hist_distances":  [round(d, 4) for d in hist_dists],   # per-pair change rate
        "dominant_hues":   dominant_hues,                        # per-frame top-3 hues + %
        "details":         "; ".join(details_parts),
    }


def evaluate_sam3_keyframes_with_qwen(
    sam3_result: Dict,
    grounded_spec: Dict,
    sam3_masks_dir: str,
    model=None,
    processor=None,
    device: str = "cuda",
    video_name: str = "",
) -> Dict:
    """
    OpenCV histogram-based frame consistency check on SAM3 masked crops.
    Replaces VLM-based approach — no model needed (model/processor args kept for API compat).

    For each key frame's sliding window, checks:
    - HSV histogram chi-squared distance between consecutive frames (colour morph detection)
    - Contour area ratio between consecutive frames (shape discontinuity detection)

    Returns:
      {"passed": bool, "confidence": float, "reason": str, "per_keyframe": [...]}
    """
    entities     = grounded_spec.get("entities", {})
    subject_name = entities.get("subject", {}).get("name", "subject")

    key_frame_analysis = sam3_result.get("key_frame_analysis", [])
    per_kf = []

    # Accumulate plot data; only flush to disk if ALL keyframes pass
    pending_plots = []  # [(block_track, hist_distances, crop_frame_indices, kf_idx)]

    for kf in key_frame_analysis:
        kf_idx   = kf.get("key_frame_orig_idx", "?")
        win_traj = kf.get("sliding_window_trajectory", [])

        # Collect subject masked-crop paths in frame order, keeping global frame indices
        subject_crops = []
        crop_frame_indices = []
        for fe in win_traj:
            subj_imgs = fe.get("images", {}).get(subject_name, {})
            if not isinstance(subj_imgs, dict):
                continue
            rel = subj_imgs.get("masked_crop")
            if rel:
                full = os.path.join(sam3_masks_dir, rel)
                if os.path.isfile(full):
                    subject_crops.append(full)
                    crop_frame_indices.append(fe.get("frame", None))

        print(f"   [CV-KF] kf={kf_idx}  {len(subject_crops)} masked crops → histogram check...",
              flush=True)

        if not subject_crops:
            kf_result = {
                "key_frame_orig_idx": kf_idx,
                "passed":             False,
                "confidence":         0.0,
                "details":            "No subject crop images available.",
            }
            per_kf.append(kf_result)
            print(f"   [CV-KF] kf={kf_idx} FAIL — no crops", flush=True)
            return {"passed": False, "confidence": 0.0,
                    "reason": f"kf={kf_idx}: no subject crops found",
                    "per_keyframe": per_kf}

        cv_result    = _cv_histogram_consistency(subject_crops, frame_indices=crop_frame_indices)
        block_track  = _cv_color_block_tracking(subject_crops)
        passed       = cv_result["passed"]
        conf         = cv_result["confidence"]
        details      = cv_result["details"]

        kf_result = {
            "key_frame_orig_idx": kf_idx,
            "passed":             passed,
            "confidence":         conf,
            "max_hist_dist":      cv_result["max_hist_dist"],
            "max_area_ratio":     cv_result["max_area_ratio"],
            "spike_frames":       cv_result["spike_frames"],
            "hist_distances":     cv_result.get("hist_distances", []),
            "dominant_hues":      cv_result.get("dominant_hues", []),
            "color_block_track":  block_track,
            "details":            details,
        }
        per_kf.append(kf_result)
        if passed:
            print(f"   [CV-KF] kf={kf_idx} passed=True conf={conf:.2f} | {details}",
                  flush=True)
            pending_plots.append((block_track, cv_result.get("hist_distances", []),
                                  crop_frame_indices, kf_idx))
        else:
            import json as _json
            slim = {k: v for k, v in kf_result.items() if k != "color_block_track"}
            print(f"   [CV-KF] kf={kf_idx} passed=False conf={conf:.2f}\n"
                  f"   {_json.dumps(slim, indent=4, default=str)}",
                  flush=True)

        if not passed:
            return {"passed": False, "confidence": conf,
                    "reason": f"kf={kf_idx} failed: {details}",
                    "per_keyframe": per_kf}

    if not per_kf:
        return {"passed": False, "confidence": 0.0,
                "reason": "No key frames evaluated.", "per_keyframe": []}

    avg_conf = sum(r["confidence"] for r in per_kf) / len(per_kf)

    # All keyframes passed — save all colour track plots to color_trajectory/
    if pending_plots:
        safe_name = os.path.splitext(video_name)[0][:80].replace(" ", "_") if video_name else "unknown"
        traj_dir  = os.path.join(sam3_masks_dir, "color_trajectory")
        os.makedirs(traj_dir, exist_ok=True)
        for bt, hd, cfi, kid in pending_plots:
            plot_path = os.path.join(traj_dir, f"{safe_name}_kf{kid}_color_track.png")
            _save_color_track_plot(bt, hd, cfi, kid, plot_path, video_name=video_name)

    return {"passed": True, "confidence": avg_conf,
            "reason": f"All {len(per_kf)} key frames passed (avg_conf={avg_conf:.2f})",
            "per_keyframe": per_kf}


def detect_contact_with_sam3(
    video_path: str,
    grounded_spec: Dict,
    device: str = "cuda",
    save_dir: Optional[str] = None,
    sam3_conda_env: str = "sam3",
    fps: float = 3.0,
    top_k: int = 5,
    window: int = 4,
) -> Tuple[Dict[str, any], str]:
    """
    Use SAM3 to detect and track subject and entities throughout video using text prompts.
    Runs as a subprocess in the `sam3` conda environment to avoid dependency conflicts.

    Args:
        video_path: Path to video file
        grounded_spec: Grounded specification from Stage 0 containing entity names
        device: Device to run on ("cuda" or "cpu")
        save_dir: Optional directory to save SAM3 masks
        sam3_conda_env: Name of the conda environment that has SAM3 installed (default: "sam3")

    Returns:
        Tuple of (result_dict, raw_response)
    """
    import json
    import subprocess

    print(f"\n   🎭 Using SAM3 for video object tracking (subprocess: conda env '{sam3_conda_env}')")
    print(f"      Video: {os.path.basename(video_path)}")

    # Build the subprocess script path relative to this file
    script_path = os.path.join(os.path.dirname(__file__), "run_sam3_contact.py")

    # Resolve the sam3 env's python by absolute path — never use `conda run`,
    # which resolves to the currently-active env, not the requested one.
    sam3_python = os.environ.get("SAM3_PYTHON", "")
    if not os.path.isfile(sam3_python):
        # Try common locations: same miniconda base as the running python
        import sys
        # e.g. /path/to/miniconda3/envs/vbch/bin/python -> /path/to/miniconda3
        _parts = sys.executable.split(os.sep)
        try:
            _envs_idx = _parts.index("envs")
            _conda_base = os.sep + os.path.join(*_parts[:_envs_idx])
        except ValueError:
            # running from base env: .../miniconda3/bin/python
            _conda_base = os.path.dirname(os.path.dirname(sys.executable))
        sam3_python = os.path.join(_conda_base, "envs", sam3_conda_env, "bin", "python")

    if not os.path.isfile(sam3_python):
        raise RuntimeError(
            f"Cannot find python for conda env '{sam3_conda_env}' at {sam3_python}. "
            f"Set SAM3_PYTHON=/path/to/envs/{sam3_conda_env}/bin/python"
        )
    print(f"      SAM3 python: {sam3_python}")
    python_cmd = [sam3_python]

    cmd = python_cmd + [
        script_path,
        "--video_path", video_path,
        "--grounded_spec", json.dumps(grounded_spec),
        "--device", device,
        "--fps", str(fps),
        "--top_k", str(top_k),
        "--window", str(window),
    ]
    if save_dir:
        cmd += ["--save_dir", save_dir]

    # Forward HF/cache env vars so the subprocess can find the model cache
    env = os.environ.copy()
    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                "PIP_CACHE_DIR", "TMPDIR", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if key in os.environ:
            env[key] = os.environ[key]

    print(f"      Running SAM3 subprocess...")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"SAM3 subprocess timed out for {video_path}")

    # stderr goes to our stdout for visibility
    if proc.stderr.strip():
        print(proc.stderr, end="")

    if proc.returncode != 0:
        raise RuntimeError(
            f"SAM3 subprocess failed (exit {proc.returncode}) for {video_path}:\n{proc.stderr[-2000:]}"
        )

    # Last line of stdout is the JSON result
    stdout_lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    if not stdout_lines:
        raise RuntimeError(f"SAM3 subprocess produced no output for {video_path}")

    output = json.loads(stdout_lines[-1])
    return output["result"], output["raw_response"]


def save_sam3_masks(
    video_path: str,
    outputs_per_frame: Dict,
    contact_frames: List[int],
    save_dir: str
):
    """
    Save SAM3 masks as visualization images.

    Args:
        video_path: Path to video
        outputs_per_frame: SAM3 outputs per frame
        contact_frames: List of contact frame indices
        save_dir: Directory to save masks
    """
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    sam3_dir = os.path.join(save_dir, "sam3_masks")
    os.makedirs(sam3_dir, exist_ok=True)

    print(f"      💾 Saving SAM3 masks to {sam3_dir}")

    # Save masks for contact frames only (to avoid too many files)
    for frame_idx in contact_frames[:50]:  # Limit to first 50 contact frames
        if frame_idx not in outputs_per_frame:
            continue

        frame_outputs = outputs_per_frame[frame_idx]
        object_ids = frame_outputs.get('object_ids', [])
        masks = frame_outputs.get('masks')
        scores = frame_outputs.get('scores', [])

        if masks is None or len(masks) == 0:
            continue

        # Read original frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            continue

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        overlay = frame_rgb.copy().astype(np.float32)

        # Apply colored masks
        colors = [
            [255, 0, 0],     # RED for first object (likely subject)
            [0, 255, 0],     # GREEN for second object
            [0, 0, 255],     # BLUE for third object
            [255, 255, 0],   # YELLOW for fourth
        ]

        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            color_overlay = np.zeros_like(overlay)
            color_overlay[:, :, 0] = color[0]
            color_overlay[:, :, 1] = color[1]
            color_overlay[:, :, 2] = color[2]

            # Apply mask with transparency
            mask_bool = mask > 0.5
            overlay[mask_bool] = overlay[mask_bool] * 0.6 + color_overlay[mask_bool] * 0.4

        overlay = overlay.astype(np.uint8)

        # Convert to PIL for text
        pil_image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(pil_image)

        # Try to load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("Arial.ttf", 16)
            except:
                font = ImageFont.load_default()

        # Draw info
        info_text = f"SAM3 | Frame {frame_idx} | CONTACT | Objects: {len(object_ids)}"
        draw.text((10, 10), info_text, fill="red", font=font)

        # Draw legend
        legend_y = 40
        for i, (obj_id, score) in enumerate(zip(object_ids, scores)):
            color_name = ["RED", "GREEN", "BLUE", "YELLOW"][i % 4]
            text = f"{color_name}: Object {obj_id} (score: {score:.2f})"
            draw.text((10, legend_y), text, fill="white", font=font)
            legend_y += 25

        # Save
        filename = f"{video_name}_frame_{frame_idx:04d}_CONTACT.jpg"
        save_path = os.path.join(sam3_dir, filename)
        pil_image.save(save_path)

    print(f"      ✅ Saved {min(len(contact_frames), 50)} mask visualizations")


def process_frame_batch_for_contact(frames: List, prompt: str, model, processor, device) -> List[bool]:
    """
    Process a batch of frames to detect contact in each.

    Args:
        frames: List of OpenCV frames (BGR format)
        prompt: Detection prompt
        model: Qwen3-VL model
        processor: Qwen3-VL processor
        device: Device

    Returns:
        List of booleans indicating contact detection for each frame
    """
    from PIL import Image

    # Convert OpenCV frames (BGR) to PIL Images (RGB)
    pil_images = []
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        pil_images.append(pil_image)

    # Build messages with multiple images
    content = []
    for img in pil_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})

    messages = [{
        "role": "user",
        "content": content
    }]

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Process images
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    # Prepare inputs
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        return_tensors="pt",
        do_resize=False
    ).to(device)

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2000,
            do_sample=False
        )

    # Decode response
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    # Parse JSON response
    contact_results = []
    try:
        # Extract JSON
        json_text = response
        if "```json" in response:
            json_text = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_text = response.split("```")[1].split("```")[0].strip()

        result = json.loads(json_text)
        results_list = result.get("results", [])

        # Extract contact boolean for each frame
        for item in results_list:
            contact = item.get("contact", False)
            contact_results.append(contact)

        # Ensure we have results for all frames
        while len(contact_results) < len(frames):
            contact_results.append(False)

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"      ⚠️  Failed to parse contact detection: {e}")
        print(f"      Response: {response[:200]}")
        # Default to no contact for all frames
        contact_results = [False] * len(frames)

    return contact_results



def detect_contact_frames_spatial(video_path: str, grounded_spec: Dict, model, processor, device, target_frames: int, save_dir: Optional[str] = None) -> List[int]:
    """
    Detect contact frames using bounding box overlap (IoU-based objective detection).

    NEW APPROACH: For each frame independently:
    1. Ground subject and entity with bounding boxes
    2. Calculate IoU (Intersection over Union)
    3. Mark as contact if IoU > 5%

    This is objective spatial grounding, not subjective temporal reasoning.

    Args:
        video_path: Path to video file
        grounded_spec: Grounded specification with action_phases, subject, and interactive_entities
        model: Qwen3-VL model
        processor: Qwen3-VL processor
        device: Device
        target_frames: Number of frames the video was sampled to
        save_dir: Optional directory to save annotated frames with bounding boxes

    Returns:
        List of detected contact frame indices (sampled indices 0 to target_frames-1)
    """
    import time
    import cv2

    # Get video name for saving files
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Extract information from grounded spec
    action_phases = grounded_spec.get("action_phases", {})
    phases = action_phases.get("phases", [])

    # Extract subject and interactive entities from entities section
    entities = grounded_spec.get("entities", {})
    subject_name = entities.get("subject", {}).get("name", "subject")
    interactive_entities_list = entities.get("interactive_entities", [])
    interactive_entities = [e.get("name", f"entity_{i}") for i, e in enumerate(interactive_entities_list)]

    if not phases:
        print("   ⚠️  No phases found in grounded spec")
        return []

    print(f"\n   🔍 Frame-by-frame spatial contact detection...")
    print(f"   📝 Subject: {subject_name}")
    print(f"   📝 Interactive entities: {', '.join(interactive_entities)}")

    # Only process contact phases (skip initial/final)
    contact_phases = [p for p in phases if p.get("phase_name") not in ["initial", "final"]]

    if not contact_phases:
        print("   ⚠️  No contact phases found")
        return []

    all_contact_frames = []

    # Extract frames from video
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampling_ratio = total_video_frames / target_frames if target_frames > 0 else 1.0

    print(f"   📹 Extracting {target_frames} frames for analysis...")

    # Extract all sampled frames
    frames = []
    frame_indices = []
    for sampled_idx in range(target_frames):
        original_idx = int(round(sampled_idx * sampling_ratio))
        original_idx = min(max(0, original_idx), total_video_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, original_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            frame_indices.append(sampled_idx)

    cap.release()

    print(f"   ✅ Extracted {len(frames)} frames")

    # Process each contact phase
    for phase in contact_phases:
        phase_name = phase.get("phase_name", "contact")
        phase_desc = phase.get("description", "")

        # Extract entity name from phase (e.g., "impact_floor" -> "floor")
        entity_name = phase_name.split('_', 1)[1] if '_' in phase_name else 'entity'

        print(f"\n   🎯 Detecting contact: {subject_name} ↔ {entity_name}")
        print(f"      Phase: '{phase_name}' - {phase_desc}")
        print(f"      Method: Multi-criteria adaptive contact detection (IoU + proximity + spatial)")

        # Process frames one at a time with bounding box grounding
        detected_frames_for_phase = []
        contact_count = 0

        t_start = time.time()

        for i, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
            # Detect contact via bounding box overlap
            is_contact = process_single_frame_with_bbox_grounding(
                frame=frame,
                subject_name=subject_name,
                entity_name=entity_name,
                model=model,
                processor=processor,
                device=device,
                save_dir=save_dir,
                frame_idx=frame_idx,
                video_name=video_name,
                phase_description=phase_desc
            )

            if is_contact:
                detected_frames_for_phase.append(frame_idx)
                contact_count += 1

            # Progress update every 8 frames
            if (i + 1) % 8 == 0 or (i + 1) == len(frames):
                elapsed = time.time() - t_start
                print(f"      Progress: {i+1}/{len(frames)} frames analyzed, {contact_count} contacts detected ({elapsed:.1f}s)")

        print(f"      ✅ Total contact frames: {len(detected_frames_for_phase)}")
        print(f"      📍 Frames: {detected_frames_for_phase}")

        all_contact_frames.extend(detected_frames_for_phase)

    # Remove duplicates and sort
    unique_contact_frames = sorted(list(set(all_contact_frames)))

    print(f"\n   🎯 Total contact frames across all phases: {len(unique_contact_frames)}")
    print(f"   📊 Frames: {unique_contact_frames}")

    return unique_contact_frames


def detect_key_frames_with_grounding(video_path: str, grounded_spec: Dict, model, processor, device, target_frames: int, sampling_fps: float, save_dir: Optional[str] = None) -> List[int]:
    """
    DEPRECATED: Temporal grounding approach - replaced by frame-by-frame spatial detection.

    This function kept for backward compatibility but now calls the new spatial detection approach.

    Args:
        video_path: Path to video file
        grounded_spec: Grounded specification with action_phases, subject, and interactive_entities
        model: Qwen3-VL model
        processor: Qwen3-VL processor
        device: Device
        target_frames: Number of frames the video was sampled to
        sampling_fps: FPS to sample video at (unused in new approach)
        save_dir: Optional directory to save annotated frames with bounding boxes

    Returns:
        List of detected frame indices (sampled indices 0 to target_frames-1)
    """
    print("   ℹ️  Using new frame-by-frame spatial contact detection...")

    # Call new bounding box based contact detection
    return detect_contact_frames_spatial(
        video_path=video_path,
        grounded_spec=grounded_spec,
        model=model,
        processor=processor,
        device=device,
        target_frames=target_frames,
        save_dir=save_dir
    )


def detect_key_frames_with_vlm(video_path: str, grounded_spec: Dict, model, processor, device, save_dir: Optional[str] = None) -> List[int]:
    """
    Stage 1a: Key Frame Detection

    Uses Qwen3-VL to watch the entire video and identify key frames corresponding to action phases.

    Args:
        video_path: Path to video file
        grounded_spec: Grounded specification with action_phases
        model: Qwen3-VL model
        processor: Qwen3-VL processor
        device: Device
        save_dir: Optional directory to save detected key frames as images

    Returns:
        List of key frame indices (sorted)
    """
    import time

    t0 = time.time()

    # Extract action phases from spec
    action_phases = grounded_spec.get("action_phases", {})
    phases = action_phases.get("phases", [])
    total_phases = action_phases.get("total_phases", len(phases))

    # Get video metadata
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / video_fps if video_fps > 0 else 0
    cap.release()

    # Calculate target sampling fps to get desired frame coverage
    # We want ~30% of frames, capped at 128 frames
    target_frames = min(int(total_frames * 0.3), 128)
    target_frames = max(target_frames, 16)  # At least 16 frames

    # Calculate sampling fps to achieve target frame count
    # sampling_fps = target_frames / duration
    if duration > 0:
        sampling_fps = target_frames / duration
        sampling_fps = min(sampling_fps, video_fps)  # Don't exceed original fps
    else:
        sampling_fps = 1.0

    print(f"   📹 Video: {total_frames} frames @ {video_fps:.1f}fps = {duration:.2f}s")
    print(f"   📊 Sampling at {sampling_fps:.1f} fps (target ~{target_frames} frames)")

    # Use grounding-based detection approach (replaces old selection-based approach)
    # This calls VLM multiple times (once per phase) to ground/detect when each phase occurs
    key_frames_sampled = detect_key_frames_with_grounding(
        video_path=video_path,
        grounded_spec=grounded_spec,
        model=model,
        processor=processor,
        device=device,
        target_frames=target_frames,
        sampling_fps=sampling_fps,
        save_dir=save_dir
    )

    # Map sampled frame indices back to original video frame indices
    # VLM sees sampled frames indexed 0-(target_frames-1), but we need original video indices 0-(total_frames-1)
    # Mapping: original_idx = sampled_idx * (total_frames / target_frames)
    sampling_ratio = total_frames / target_frames if target_frames > 0 else 1.0

    # Create mapping (before deduplication so we can show all mappings)
    mapping_pairs = []
    for sampled_idx in key_frames_sampled:
        original_idx = int(round(sampled_idx * sampling_ratio))
        original_idx = min(max(0, original_idx), total_frames - 1)  # Clamp to valid range
        mapping_pairs.append((sampled_idx, original_idx))

    # Extract unique original indices
    key_frames_original = sorted(list(set([orig for _, orig in mapping_pairs])))

    t3 = time.time()
    print(f"   ⏱️  Total key frame detection: {t3-t0:.1f}s")
    print(f"   🎯 Detected {len(key_frames_sampled)} key frames (sampled indices): {key_frames_sampled}")
    print(f"   🎯 Mapped to {len(key_frames_original)} original video frames: {key_frames_original}")
    if len(key_frames_original) < len(key_frames_sampled):
        print(f"   ⚠️  Note: {len(key_frames_sampled) - len(key_frames_original)} duplicate(s) removed after mapping")

    # Show detailed mapping for verification
    print(f"\n   🔍 Frame Mapping Verification:")
    print(f"      Sampled → Original (ratio: {sampling_ratio:.2f}x)")
    for sampled_idx, original_idx in mapping_pairs:
        print(f"      ✓ Frame {sampled_idx:3d} → {original_idx:3d}")

    # Save sampled frames and key frames as images if save_dir provided
    if save_dir:
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Create subfolders
        sampled_frames_dir = os.path.join(save_dir, "sampled_frames")
        key_frames_dir = os.path.join(save_dir, "key_frames")
        os.makedirs(sampled_frames_dir, exist_ok=True)
        os.makedirs(key_frames_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        # Save all sampled frames (what VLM sees)
        print(f"   💾 Saving {target_frames} sampled frames...")
        sampled_original_indices = []
        for sampled_idx in range(target_frames):
            original_idx = int(round(sampled_idx * sampling_ratio))
            original_idx = min(max(0, original_idx), total_frames - 1)
            sampled_original_indices.append(original_idx)

            cap.set(cv2.CAP_PROP_POS_FRAMES, original_idx)
            ret, frame = cap.read()
            if ret:
                save_path = os.path.join(sampled_frames_dir, f"{video_name}_sampled_{sampled_idx:04d}_orig_{original_idx:04d}.jpg")
                cv2.imwrite(save_path, frame)

        # Save selected key frames
        print(f"   💾 Saving {len(key_frames_original)} selected key frames...")
        for orig_frame_idx in key_frames_original:
            cap.set(cv2.CAP_PROP_POS_FRAMES, orig_frame_idx)
            ret, frame = cap.read()
            if ret:
                # Find which sampled index(es) this corresponds to
                sampled_indices = [s for s, o in mapping_pairs if o == orig_frame_idx]
                sampled_str = "_".join([f"s{s:04d}" for s in sampled_indices])
                save_path = os.path.join(key_frames_dir, f"{video_name}_keyframe_{sampled_str}_orig_{orig_frame_idx:04d}.jpg")
                cv2.imwrite(save_path, frame)

        cap.release()
        print(f"   ✅ Saved to:")
        print(f"      - {target_frames} sampled frames → {sampled_frames_dir}")
        print(f"      - {len(key_frames_original)} key frames → {key_frames_dir}")

    return key_frames_original


def extract_frames_around_keyframes(video_path: str, key_frames: List[int], window_size: int = 5) -> Dict[int, List[Tuple[Image.Image, int]]]:
    """
    Extract frames around each key frame.

    Args:
        video_path: Path to video file
        key_frames: List of key frame indices
        window_size: Number of frames to extract around each key frame (before and after)

    Returns:
        Dict mapping key_frame_idx -> List[(PIL.Image, frame_idx)]

    Example:
        If key_frame=50 and window_size=5:
        Extracts frames [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55] (11 frames total)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_dict = {}

    for key_frame_idx in key_frames:
        # Calculate window bounds
        start_frame = max(0, key_frame_idx - window_size)
        end_frame = min(total_frames - 1, key_frame_idx + window_size)

        frames = []
        for frame_idx in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frames.append((pil_frame, frame_idx))

        frames_dict[key_frame_idx] = frames

    cap.release()
    return frames_dict


def validate_key_frame_window(frames: List[Tuple[Image.Image, int]], grounded_spec: Dict,
                                model, processor, device, window_name: str = "Window") -> Tuple[Dict, str]:
    """
    Validate a single key frame window (10-11 frames around a key frame).

    Args:
        frames: List of (PIL.Image, frame_idx) tuples
        grounded_spec: Grounded specification
        model: Qwen VL model
        processor: Qwen VL processor
        device: Device
        window_name: Name of this window for logging

    Returns:
        Tuple of (result_dict, raw_response)
    """
    import time

    entities = grounded_spec.get("entities", {})

    # Build validation prompt for this window
    prompt_text = f"""
You are a forensic video analyst validating a KEY FRAME WINDOW from a video.

You will see {len(frames)} consecutive frames around a key moment. Your task is to validate consistency.

=== SPECIFICATION ===
{json.dumps(grounded_spec, indent=2)}

=== YOUR VALIDATION TASKS ===

**1. INVARIANT PROPERTIES (MUST NOT CHANGE)**
- Entity counts (e.g., ball count must remain {entities.get('subject', {}).get('count', 1)})
- Material types (e.g., ball must remain rubber, floor must remain tile)
- Base geometry (e.g., ball must remain sphere)
- Color signature (e.g., ball's color pattern must stay consistent)

**2. VARIANT PROPERTIES (CAN CHANGE within physics)**
- Position (ball can move)
- Velocity (ball can speed up/slow down)
- Shape deformation (ball can compress at impact, then return to sphere)

**3. ENTITY IDENTITY CONSISTENCY**
Is this the SAME ball across all {len(frames)} frames?
- Check: color pattern, size, texture, visual signature
- Any disappearing/appearing?
- Any extra balls when spec says count=1?

**4. BACKGROUND CONSISTENCY**
- Same room/environment throughout?
- Floor consistent?
- Any scene cuts?

=== RESPONSE FORMAT ===

CROSS-FRAME OBJECT IDENTITY:
For SUBJECT (ball):
- Same object across all {len(frames)} frames? [YES/NO/UNCERTAIN]
- Confidence: [0-100%]
- Evidence FOR same object: [list matching visual signatures]
- Evidence AGAINST: [list inconsistencies]
- Count consistent? Detected exactly {entities.get('subject', {}).get('count', 1)} ball(s) in EVERY frame? [YES/NO]

For FLOOR:
- Same throughout? [YES/NO]
- Any sudden changes? [list or "None"]

For BACKGROUND:
- Same throughout? [YES/NO]
- Any scene cuts? [list or "None"]

VALIDATION RESULT: [PASS or FAIL]

FAILURE REASON (if FAIL): [Explain what failed]

SUMMARY:
- Cross-frame identity confidence: [0-100%]
- Invariant violations: [list or "None"]
- Entity identity issues: [list or "None"]
"""

    # Prepare messages with all frames in this window
    content = [{"type": "text", "text": prompt_text}]
    for pil_frame, frame_idx in frames:
        content.append({"type": "image", "image": pil_frame})
        content.append({"type": "text", "text": f"Frame {frame_idx}"})

    messages = [{"role": "user", "content": content}]

    # Process with VLM
    t0 = time.time()
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    t1 = time.time()

    # Generate response (smaller token limit for faster processing)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1500,  # Smaller window = less output needed
            temperature=0.1,
            do_sample=False
        )

    t2 = time.time()

    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    num_tokens = len(generated_ids_trimmed[0])
    tokens_per_sec = num_tokens / (t2 - t1) if (t2 - t1) > 0 else 0

    print(f"      ⏱️  {window_name}: {t2-t0:.1f}s total ({num_tokens} tokens, {tokens_per_sec:.1f} tok/s)")

    # Parse response
    validation_match = re.search(r"VALIDATION RESULT:\s*(PASS|FAIL)", response, re.IGNORECASE)
    if validation_match:
        passed = validation_match.group(1).upper() == "PASS"
    else:
        passed = "VALIDATION RESULT: PASS" in response

    # Extract failure reason if failed
    reason = ""
    if not passed:
        reason_match = re.search(r"FAILURE REASON.*?:\s*(.+?)(?=\n\n|SUMMARY|$)", response, re.IGNORECASE | re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip()
        else:
            reason = "Window validation failed (see VLM response)"

    result = {
        "passed": passed,
        "reason": reason if not passed else f"{window_name} validated successfully",
        "num_frames": len(frames)
    }

    return result, response


def validate_temporal_consistency(
    video_path: str,
    grounded_spec: Dict,
    model,
    processor,
    device,
    num_frames: int = 30,
    window_size: int = 5
) -> Tuple[Dict, str]:
    """
    Stage 1: Two-Stage Temporal Consistency Validation with Early Stopping

    Stage 1a: Key frame detection using Qwen3-VL video understanding
    Stage 1b: Sequential validation of key frame windows with early stopping

    Args:
        video_path: Path to video file
        grounded_spec: Grounded specification from Stage 0
        model: Qwen VL model
        processor: Qwen VL processor
        device: Device
        num_frames: Deprecated (kept for backward compatibility)
        window_size: Frames to sample around each key frame (default: 5, creates 11-frame windows)

    Returns:
        Tuple of (result_dict, raw_response)
        - result_dict: {"passed": bool, "reason": str, "key_frames": list}
        - raw_response: Combined VLM responses from all windows
    """
    import time

    t0_total = time.time()

    # Create key frames directory
    video_dir = os.path.dirname(video_path)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    key_frames_dir = os.path.join(video_dir, "key_frames", video_basename)

    # STAGE 1A: Detect key frames
    print(f"\n   🔍 Stage 1a: Key Frame Detection")
    key_frames = detect_key_frames_with_vlm(video_path, grounded_spec, model, processor, device, save_dir=key_frames_dir)

    if not key_frames:
        print(f"   ⚠️  No key frames detected, falling back to uniform sampling")
        # Fallback: sample 3 uniform frames (beginning, middle, end)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        key_frames = [int(total_frames * 0.2), int(total_frames * 0.5), int(total_frames * 0.8)]

    # STAGE 1B: Extract frames around key frames
    print(f"\n   📸 Stage 1b: Extracting {window_size*2+1} frames around each of {len(key_frames)} key frames")
    t0 = time.time()
    frames_dict = extract_frames_around_keyframes(video_path, key_frames, window_size=window_size)
    t1 = time.time()
    total_sampled_frames = sum(len(frames) for frames in frames_dict.values())
    print(f"   ⏱️  Frame extraction: {t1-t0:.1f}s ({total_sampled_frames} total frames)")

    # STAGE 1C: Sequential validation with EARLY STOPPING
    print(f"\n   ✅ Stage 1c: Sequential Validation (with early stopping)")
    all_responses = []
    windows_checked = []
    all_frame_indices = []

    for window_idx, (key_frame_idx, frames) in enumerate(frames_dict.items(), 1):
        window_name = f"Window {window_idx}/{len(frames_dict)} (key frame {key_frame_idx})"
        print(f"      🔎 Validating {window_name}...")

        # Validate this key frame window
        result, response = validate_key_frame_window(
            frames, grounded_spec, model, processor, device, window_name
        )

        all_responses.append(f"=== {window_name} ===\n{response}")
        windows_checked.append(key_frame_idx)

        # Collect frame indices from this window
        frame_indices_in_window = [frame_idx for _, frame_idx in frames]
        all_frame_indices.extend(frame_indices_in_window)

        if not result["passed"]:
            # EARLY STOPPING: This window failed → reject video immediately
            print(f"      ❌ {window_name} FAILED - stopping validation")
            t_total = time.time() - t0_total

            return {
                "passed": False,
                "reason": f"Failed at {window_name}: {result['reason']}",
                "key_frames_checked": windows_checked,
                "num_frames_checked": len(all_frame_indices),
                "frame_indices": all_frame_indices,
                "total_windows": len(frames_dict),
                "windows_passed": window_idx - 1,
                "total_time": t_total
            }, "\n\n".join(all_responses)
        else:
            print(f"      ✅ {window_name} PASSED")

    # All windows passed!
    t_total = time.time() - t0_total
    print(f"\n   🎉 All {len(frames_dict)} key frame windows PASSED!")
    print(f"   ⏱️  Total validation time: {t_total:.1f}s")

    return {
        "passed": True,
        "reason": f"All {len(key_frames)} key frame windows passed validation",
        "key_frames_checked": windows_checked,
        "num_frames_checked": len(all_frame_indices),
        "frame_indices": all_frame_indices,
        "total_windows": len(frames_dict),
        "windows_passed": len(frames_dict),
        "total_time": t_total
    }, "\n\n".join(all_responses)


def load_stage1_model_on_gpu(gpu_id: int, model_name: str = "Qwen/Qwen3-VL-8B-Instruct"):
    """
    Load VLM on a specific GPU.
    Supports LLaVA-Video and Qwen3-VL models.

    Returns:
        (model, processor, device)
        For LLaVA-Video, processor is a tuple: (tokenizer, image_processor, max_length)
    """
    print(f"   📥 Loading {model_name} on GPU {gpu_id}...")
    device = f"cuda:{gpu_id}"

    if "LLaVA" in model_name or "llava" in model_name.lower():
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        lm_name = get_model_name_from_path(model_name)
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            model_name, None, lm_name,
            torch_dtype="bfloat16",
            device_map={"": gpu_id},
        )
        model.eval()
        processor = (tokenizer, image_processor, max_length)
    elif "Qwen3" in model_name:
        if not QWEN3_AVAILABLE:
            raise ImportError("Qwen3VLForConditionalGeneration not available. pip install transformers>=4.57.0")
        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map={"": gpu_id})
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    print(f"   ✓ Model loaded on GPU {gpu_id} ({device})")
    return model, processor, device


def batch_check_frames_against_checklist(frames: List[Image.Image], checklist: str,
                                         model, processor, device) -> Tuple[List[Dict], str]:
    """
    Check multiple frames against checklist in single batch inference.

    Args:
        frames: List of PIL Images (e.g., 10 random frames)
        checklist: Prompt alignment checklist
        model: Qwen2.5-VL model
        processor: Qwen2.5-VL processor
        device: Device (e.g., "cuda:0")

    Returns:
        Tuple of (parsed_results, raw_response)
        - parsed_results: List of dicts [{"passed": bool, "reason": str}, ...]
        - raw_response: Full VLM response text
    """
    # Build multi-image prompt
    messages = [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": frame} for frame in frames],
            {"type": "text", "text": f"""
You are a forensic video analyst checking {len(frames)} video frames against a specification.

SPECIFICATION (Required Elements):
{checklist}

INSTRUCTIONS:
For EACH image, perform a 2-step forensic analysis:

1. OBSERVE: Describe what you see in detail:
   - ENTITIES: All objects/subjects present - names, shapes, colors, counts, appearances
   - BACKGROUND/ENVIRONMENT: Floor, walls, setting, stability, lighting

2. VERIFY: Check if observations match the specification requirements:
   If the specification is JSON format (contains "entities", "validation_constraints", etc.):
   - ✓ All required entities are present with correct counts (check validation_constraints.count_constraints)
   - ✓ Subject entity matches its specifications (check entities.subject - geometry, material, appearance)
   - ✓ All interactive entities are present and correct (check entities.interactive_entity or entities.interactive_entities list)
   - ✓ Invariant properties are correct (check state_changes.invariant_properties - colors, materials shouldn't change)
   - ✓ Background/environment is appropriate (check entities.background)

   If the specification is text format (contains [VISUAL_SPECS], [ENVIRONMENT_SPECS]):
   - ✓ Subject matches VISUAL_SPECS (name, shape, color, count)
   - ✓ Background matches ENVIRONMENT_SPECS (type, stability)

Be STRICT: If ANY required element is missing, incorrect, or contradicts the specification → FAIL

RESPONSE FORMAT (provide for each image):
Image 1: [PASS/FAIL]
OBSERVATION: [Entities: describe all objects | Background: describe environment/setting]
REASON: [Why it passed or failed - reference specific requirements from specification]

Image 2: [PASS/FAIL]
OBSERVATION: [Entities: ... | Background: ...]
REASON: [Why it passed/failed - check against specification]

...

Image {len(frames)}: [PASS/FAIL]
OBSERVATION: [Entities: ... | Background: ...]
REASON: [Why it passed/failed]

IMPORTANT:
- Check ALL entities and background/environment for every frame
- Reference specific requirements from the specification (entity counts, geometry, invariant properties, etc.)
- Provide detailed observations and reasons for BOTH pass and fail cases
- Do NOT leave reason empty
"""}
        ]
    }]

    # Process with official pattern using process_vision_info
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Use official helper to process images
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1200,  # Increased for OBSERVATION + REASON per frame (10 frames)
            temperature=0.05,
            do_sample=True,
            top_p=0.95
        )

    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Parse results - new format with OBSERVATION and REASON
    results = []
    lines = response.strip().split('\n')

    for i in range(len(frames)):
        # Look for "Image X: PASS/FAIL"
        status_pattern = rf"Image\s+{i+1}\s*:\s*(PASS|FAIL)"
        status_match = None

        for line in lines:
            m = re.search(status_pattern, line, re.IGNORECASE)
            if m:
                status_match = m
                break

        if status_match:
            status = status_match.group(1).upper()

            # Extract OBSERVATION and REASON (look in subsequent lines)
            observation = ""
            reason = ""

            # Find the block for this image
            for j, line in enumerate(lines):
                if re.search(status_pattern, line, re.IGNORECASE):
                    # Found the image header, look at next lines
                    for k in range(j+1, min(j+5, len(lines))):  # Look ahead max 5 lines
                        obs_match = re.search(r"OBSERVATION:\s*(.+)", lines[k], re.IGNORECASE)
                        if obs_match:
                            observation = obs_match.group(1).strip()

                        reason_match = re.search(r"REASON:\s*(.+)", lines[k], re.IGNORECASE)
                        if reason_match:
                            reason = reason_match.group(1).strip()
                    break

            # Combine observation and reason for full context
            full_reason = f"{observation} | {reason}" if observation else reason

            results.append({
                "passed": status == "PASS",
                "reason": full_reason if status == "FAIL" else full_reason  # Show reason for both
            })
        else:
            # Couldn't parse - default to PASS to avoid false rejections
            results.append({"passed": True, "reason": "Could not parse VLM response"})

    return results, response


def check_temporal_consistency(frames: List[Image.Image], text_prompt: str, model, processor, device) -> Tuple[Dict, str]:
    """
    Check if object temporal changes are consistent with prompt description.

    Args:
        frames: List of 12 frames sampled from beginning, middle, and end of video
        text_prompt: Original generation prompt (e.g., "An egg breaks on the floor")
        model: Qwen2.5-VL model
        processor: Qwen2.5-VL processor
        device: Device

    Returns:
        Tuple of (parsed_result, raw_response)
        - parsed_result: {"consistent": bool, "reason": str}
        - raw_response: Full VLM response text
    """
    messages = [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": frame} for frame in frames],
            {"type": "text", "text": f"""
You are a forensic video analyst checking {len(frames)} frames sampled from the BEGINNING, MIDDLE, and END of a video for PROMPT ALIGNMENT and HALLUCINATIONS.

FRAME SAMPLING STRATEGY:
- Images 1-4: Beginning of video (initial state)
- Images 5-8: Middle of video (transition/action)
- Images 9-12: End of video (final state)

GENERATION PROMPT: "{text_prompt}"

TASK: Verify that the FULL TEMPORAL PROGRESSION (beginning → middle → end) matches the prompt description and detect hallucinations.

STEP 1 - DESCRIBE THE FULL PROGRESSION:
Analyze the complete temporal arc across all {len(frames)} frames:
- BEGINNING (Images 1-4): What is the object's initial state?
- MIDDLE (Images 5-8): What transition/action is happening?
- END (Images 9-12): What is the object's final state?

STEP 2 - PROMPT ALIGNMENT CHECK:
Does the observed temporal progression (initial → transition → final) MATCH what the prompt describes?

✅ CONSISTENT (Temporal changes match prompt):
- Prompt: "egg breaks" → Observe: whole egg (beginning) → cracking (middle) → broken pieces (end) ✓ MATCHES
- Prompt: "ball bounces" → Observe: ball falling (beginning) → hitting floor (middle) → bouncing up (end) ✓ MATCHES
- Prompt: "cube rotates" → Observe: cube at different rotation angles ✓ MATCHES
- Prompt: "ball drops" → Observe: ball high (beginning) → falling (middle) → on ground (end) ✓ MATCHES

❌ INCONSISTENT (Temporal changes DON'T match prompt OR show hallucinations):
- Prompt: "egg rolls" → Observe: egg breaks ✗ WRONG ACTION (hallucination)
- Prompt: "red ball" → Observe: ball changes red→blue across frames ✗ COLOR CHANGE (hallucination)
- Prompt: "one ball" → Observe: 1 ball → 2 balls ✗ COUNT CHANGE (hallucination)
- Prompt: "ball bounces" → Observe: ball morphs into cube ✗ SHAPE MORPH (hallucination)

CRITICAL RULES:
1. State changes described in prompt are EXPECTED and CONSISTENT (e.g., breaking, bouncing, rotating, dropping)
2. State changes NOT described in prompt are HALLUCINATIONS (e.g., color change, shape morph, count change, wrong action)
3. Normal physics (gravity, rotation, motion) are ALWAYS acceptable if prompt doesn't contradict them
4. You're seeing the FULL video arc (beginning→middle→end), so evaluate the COMPLETE progression

RESPONSE FORMAT:
STEP 1 - OBSERVATION: [Beginning: initial state | Middle: transition/action | End: final state]
STEP 2 - PROMPT ALIGNMENT: [Does this temporal progression match what "{text_prompt}" describes? Yes/No and why]
CONSISTENCY: [CONSISTENT/INCONSISTENT]
REASON: [If CONSISTENT: "Temporal progression matches prompt description". If INCONSISTENT: "Observed [X] but prompt describes [Y]" OR "Hallucination: [describe what's wrong]"]

Remember: If the prompt says "egg breaks", then observing whole egg → cracking → broken is CONSISTENT. Only flag changes that contradict or aren't described by the prompt.
"""}
        ]
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Use official helper to process images
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=500,  # Increased for OBSERVATION + ANALYSIS + REASON
            temperature=0.05,
            do_sample=True,
            top_p=0.95
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Parse response - extract all forensic fields
    consistent = "CONSISTENT" in response.upper() and "INCONSISTENT" not in response.upper()

    # Extract OBSERVATION
    observation_match = re.search(r"OBSERVATION:\s*(.+?)(?=STEP|PROMPT|CONSISTENCY|$)", response, re.IGNORECASE | re.DOTALL)
    observation = observation_match.group(1).strip() if observation_match else ""

    # Extract PROMPT ALIGNMENT
    alignment_match = re.search(r"PROMPT\s+ALIGNMENT:\s*(.+?)(?=CONSISTENCY|REASON|$)", response, re.IGNORECASE | re.DOTALL)
    alignment = alignment_match.group(1).strip() if alignment_match else ""

    # Extract REASON
    reason_match = re.search(r"REASON:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else ""

    # Combine all fields for full context
    full_reason = f"OBS: {observation} | ALIGNMENT: {alignment} | {reason}" if observation else reason

    result = {
        "consistent": consistent,
        "reason": full_reason
    }
    return result, response


def process_single_video_stage1a(video_path: str, checklist: str, model, processor, device,
                                  iteration_num: int = 1) -> Tuple[str, bool, str]:
    """
    Stage 1a: Check 10 random frames against checklist (STRICT - any fail = reject).

    Args:
        video_path: Path to video
        checklist: Prompt alignment checklist
        model: Qwen2.5-VL model
        processor: Qwen2.5-VL processor
        device: Device
        iteration_num: Iteration number (for different random seeds)

    Returns:
        (video_path, passed, reason)
    """
    try:
        video_name = os.path.basename(video_path)

        # Sample 10 random frames (different each iteration due to seed)
        frames, frame_indices = sample_random_frames(video_path, count=10, seed=iteration_num)

        if len(frames) == 0:
            return video_path, False, "Could not extract frames from video"

        # 🆕 Log sampled frames
        print(f"\n🎬 [{video_name}] Stage 1a - Sampled frames: {frame_indices}")

        # Batch check all frames
        results, raw_response = batch_check_frames_against_checklist(frames, checklist, model, processor, device)

        # 🆕 Log full VLM response
        print(f"📝 [{video_name}] VLM Response:")
        print(f"{'─'*80}")
        print(raw_response)
        print(f"{'─'*80}")

        # STRICT: If ANY frame fails, reject video
        for i, result in enumerate(results):
            if not result["passed"]:
                print(f"❌ [{video_name}] FAILED at frame {frame_indices[i]} (image {i+1}/10): {result['reason']}")
                return video_path, False, f"Frame {frame_indices[i]} (image {i+1}/10): {result['reason']}"

        # All frames passed
        print(f"✅ [{video_name}] All 10 frames passed")
        return video_path, True, "All frames passed checklist"

    except Exception as e:
        return video_path, False, f"Error processing video: {e}"


def process_single_video_stage1b(video_path: str, text_prompt: str, model, processor, device,
                                  iteration_num: int = 1) -> Tuple[str, bool, str]:
    """
    Stage 1b: Check temporal consistency with prompt alignment using beginning-middle-end sampling.

    Samples 12 frames total (4 from beginning, 4 from middle, 4 from end) to capture:
    - Initial state (beginning)
    - Transition/action (middle)
    - Final state (end)

    Args:
        video_path: Path to video
        text_prompt: Original generation prompt (e.g., "An egg breaks on the floor")
        model: Qwen2.5-VL model
        processor: Qwen2.5-VL processor
        device: Device
        iteration_num: Iteration number (currently unused, kept for API compatibility)

    Returns:
        (video_path, passed, reason)
    """
    try:
        video_name = os.path.basename(video_path)

        # Sample 12 frames: 4 from beginning + 4 from middle + 4 from end
        frames, frame_indices = sample_beginning_middle_end_frames(video_path, frames_per_section=4)

        if len(frames) < 12:
            return video_path, False, f"Could not extract enough frames (got {len(frames)}/12)"

        # 🆕 Log sampled frames
        print(f"\n🎬 [{video_name}] Stage 1b - Sampled frames (beginning→middle→end): {frame_indices}")

        # Check consistency with prompt alignment
        result, raw_response = check_temporal_consistency(frames, text_prompt, model, processor, device)

        # 🆕 Log full VLM response
        print(f"📝 [{video_name}] VLM Response:")
        print(f"{'─'*80}")
        print(raw_response)
        print(f"{'─'*80}")

        if not result["consistent"]:
            print(f"❌ [{video_name}] FAILED: {result['reason']}")
            return video_path, False, f"Temporal inconsistency: {result['reason']}"

        print(f"✅ [{video_name}] Temporally consistent")
        return video_path, True, "Temporally consistent"

    except Exception as e:
        return video_path, False, f"Error processing video: {e}"


def multi_gpu_stage1a(video_paths: List[str], checklist: str, num_gpus: int = 2,
                      workers_per_gpu: int = 5, iteration_num: int = 1,
                      model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
                      gpu_models: Dict = None) -> List[Tuple[str, bool, str]]:
    """
    Stage 1a: Multi-GPU parallel processing with batched inference.

    Args:
        video_paths: List of video paths
        checklist: Prompt alignment checklist
        num_gpus: Number of GPUs to use
        workers_per_gpu: Worker threads per GPU
        iteration_num: Iteration number (different random frames each iteration)
        model_name: Model to use
        gpu_models: Pre-loaded models {gpu_id: (model, processor, device)} (optional)

    Returns:
        List of (video_path, passed, reason) tuples
    """
    print(f"\n{'='*80}")
    print(f"STAGE 1a: Static Element Check (Iteration {iteration_num})")
    print(f"{'='*80}")
    print(f"   Videos: {len(video_paths)}")
    print(f"   GPUs: {num_gpus}")
    print(f"   Workers per GPU: {workers_per_gpu}")
    print(f"   Total workers: {num_gpus * workers_per_gpu}")

    # Use pre-loaded models or load if not provided (backward compatibility)
    if gpu_models is None:
        print(f"   📥 Loading models on {num_gpus} GPUs...")
        gpu_models_list = []
        for gpu_id in range(num_gpus):
            model, processor, device = load_stage1_model_on_gpu(gpu_id, model_name)
            gpu_models_list.append((model, processor, device))
    else:
        # Convert dict to list for compatibility with existing code
        gpu_models_list = [gpu_models[gpu_id] for gpu_id in range(num_gpus)]

    # Distribute videos across GPUs (round-robin)
    video_gpu_assignments = [
        (video, gpu_id % num_gpus)
        for gpu_id, video in enumerate(video_paths)
    ]

    # Process videos in parallel
    results = []
    total_workers = num_gpus * workers_per_gpu

    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        futures = {}

        for video_path, gpu_id in video_gpu_assignments:
            model, processor, device = gpu_models_list[gpu_id]
            future = executor.submit(
                process_single_video_stage1a,
                video_path, checklist, model, processor, device, iteration_num
            )
            futures[future] = video_path

        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            video_path, passed, reason = future.result()
            results.append((video_path, passed, reason))

            completed += 1
            if completed % 100 == 0 or completed == len(video_paths):
                passed_count = sum(1 for _, p, _ in results if p)
                print(f"   Progress: {completed}/{len(video_paths)} | Passed: {passed_count}")

    passed_count = sum(1 for _, p, _ in results if p)
    failed_count = len(results) - passed_count
    print(f"\n   ✓ Stage 1a Complete")
    print(f"   Passed: {passed_count} | Failed: {failed_count}")

    return results


def multi_gpu_stage1b(video_paths: List[str], text_prompt: str, num_gpus: int = 2,
                      workers_per_gpu: int = 5, iteration_num: int = 1,
                      model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
                      gpu_models: Dict = None) -> List[Tuple[str, bool, str]]:
    """
    Stage 1b: Multi-GPU temporal consistency check with prompt-aware analysis.

    Sampling strategy: 12 frames per video (4 from beginning, 4 from middle, 4 from end)
    to capture the full temporal progression: initial state → transition → final state.

    Args:
        video_paths: List of video paths (survivors from Stage 1a)
        text_prompt: Original generation prompt (e.g., "An egg breaks on the floor")
        num_gpus: Number of GPUs
        workers_per_gpu: Workers per GPU
        iteration_num: Iteration number
        model_name: Model to use
        gpu_models: Pre-loaded models {gpu_id: (model, processor, device)} (optional)

    Returns:
        List of (video_path, passed, reason) tuples
    """
    print(f"\n{'='*80}")
    print(f"STAGE 1b: Prompt-Aware Temporal Consistency Check (Iteration {iteration_num})")
    print(f"{'='*80}")
    print(f"   Videos: {len(video_paths)}")
    print(f"   Sampling: 12 frames/video (4 beginning + 4 middle + 4 end)")
    print(f"   GPUs: {num_gpus}")
    print(f"   Workers per GPU: {workers_per_gpu}")

    # Use pre-loaded models or load if not provided (backward compatibility)
    if gpu_models is None:
        print(f"   📥 Loading models on {num_gpus} GPUs...")
        gpu_models_list = []
        for gpu_id in range(num_gpus):
            model, processor, device = load_stage1_model_on_gpu(gpu_id, model_name)
            gpu_models_list.append((model, processor, device))
    else:
        # Convert dict to list for compatibility with existing code
        gpu_models_list = [gpu_models[gpu_id] for gpu_id in range(num_gpus)]

    video_gpu_assignments = [
        (video, gpu_id % num_gpus)
        for gpu_id, video in enumerate(video_paths)
    ]

    results = []
    total_workers = num_gpus * workers_per_gpu

    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        futures = {}

        for video_path, gpu_id in video_gpu_assignments:
            model, processor, device = gpu_models_list[gpu_id]
            future = executor.submit(
                process_single_video_stage1b,
                video_path, text_prompt, model, processor, device, iteration_num
            )
            futures[future] = video_path

        completed = 0
        for future in as_completed(futures):
            video_path, passed, reason = future.result()
            results.append((video_path, passed, reason))

            completed += 1
            if completed % 50 == 0 or completed == len(video_paths):
                passed_count = sum(1 for _, p, _ in results if p)
                print(f"   Progress: {completed}/{len(video_paths)} | Passed: {passed_count}")

    passed_count = sum(1 for _, p, _ in results if p)
    failed_count = len(results) - passed_count
    print(f"\n   ✓ Stage 1b Complete")
    print(f"   Passed: {passed_count} | Failed: {failed_count}")

    return results


def multi_gpu_stage0_first_frame_validation(video_paths: List[str], spec_json_str: str,
                                             num_gpus: int = 2, workers_per_gpu: int = 5,
                                              model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
                                             gpu_models: Dict = None) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Stage 0: Multi-GPU first frame validation and grounding.

    Validates first frame of each video and fills UNKNOWN fields in specification.

    Args:
        video_paths: List of video paths
        spec_json_str: JSON specification string
        num_gpus: Number of GPUs
        workers_per_gpu: Workers per GPU
        model_name: Model to use
        gpu_models: Pre-loaded models {gpu_id: (model, processor, device)} (optional)

    Returns:
        Tuple of (surviving_video_paths, grounded_specs_dict)
        - surviving_video_paths: Videos that passed Stage 0
        - grounded_specs_dict: {video_path: grounded_spec_dict}
    """
    print(f"\n{'='*80}")
    print(f"STAGE 0: First Frame Validation & Grounding")
    print(f"{'='*80}")
    print(f"   Videos: {len(video_paths)}")
    print(f"   GPUs: {num_gpus}")
    print(f"   Workers per GPU: {workers_per_gpu}")

    # Use pre-loaded models or load if not provided
    if gpu_models is None:
        print(f"   📥 Loading models on {num_gpus} GPUs...")
        gpu_models_list = []
        for gpu_id in range(num_gpus):
            model, processor, device = load_stage1_model_on_gpu(gpu_id, model_name)
            gpu_models_list.append((model, processor, device))
    else:
        # Convert dict to list
        gpu_models_list = [gpu_models[gpu_id] for gpu_id in range(num_gpus)]

    # Assign videos to GPUs (round-robin)
    video_gpu_assignments = [(video_path, i % num_gpus) for i, video_path in enumerate(video_paths)]

    # Multi-GPU + Multi-threading
    results = []
    grounded_specs = {}

    with ThreadPoolExecutor(max_workers=num_gpus * workers_per_gpu) as executor:
        futures = {}

        for video_path, gpu_id in video_gpu_assignments:
            model, processor, device = gpu_models_list[gpu_id]
            future = executor.submit(
                validate_and_ground_first_frame,
                video_path, spec_json_str, model, processor, device
            )
            futures[future] = video_path

        # Collect results
        for future in as_completed(futures):
            video_path = futures[future]
            video_name = os.path.basename(video_path)

            try:
                result, raw_response = future.result()

                print(f"\n🎬 [{video_name}] Stage 0 - First Frame Analysis")
                print(f"{'─'*80}")
                # Show full response
                print(raw_response)
                print(f"{'─'*80}")

                if result["passed"]:
                    print(f"✅ [{video_name}] PASSED: {result['reason']}")
                    results.append((video_path, True, result['reason']))
                    grounded_specs[video_path] = result['grounded_spec']

                    # Save grounded spec and visualized first frame
                    video_dir = os.path.dirname(video_path)
                    video_basename = os.path.splitext(os.path.basename(video_path))[0]
                    grounding_dir = os.path.join(video_dir, "stage0_grounding")
                    os.makedirs(grounding_dir, exist_ok=True)

                    # Save grounded spec as JSON
                    spec_path = os.path.join(grounding_dir, f"{video_basename}_grounded_spec.json")
                    with open(spec_path, 'w') as f:
                        json.dump(result['grounded_spec'], f, indent=2)
                    print(f"   💾 Saved grounded spec: {spec_path}")

                    # Debug: Check if bounding boxes are present (check both appearance and geometry)
                    if result['grounded_spec'] and 'entities' in result['grounded_spec']:
                        entities = result['grounded_spec']['entities']
                        bbox_count = 0

                        # Check subject in both appearance and geometry
                        if 'subject' in entities:
                            subj = entities['subject']
                            if (subj.get('appearance', {}).get('exact_bbox') or
                                subj.get('geometry', {}).get('exact_bbox')):
                                bbox_count += 1

                        # Check interactive_entities in both appearance and geometry
                        if 'interactive_entities' in entities and isinstance(entities['interactive_entities'], list):
                            for ent in entities['interactive_entities']:
                                if (ent.get('appearance', {}).get('exact_bbox') or
                                    ent.get('geometry', {}).get('exact_bbox')):
                                    bbox_count += 1

                        # Check background in both appearance and geometry
                        if 'background' in entities:
                            bg = entities['background']
                            if (bg.get('appearance', {}).get('exact_bbox') or
                                bg.get('geometry', {}).get('exact_bbox')):
                                bbox_count += 1

                        print(f"   🎯 Bounding boxes in grounded spec: {bbox_count}")

                    # Save visualized first frame with bounding boxes
                    try:
                        bbox_img_path = os.path.join(grounding_dir, f"{video_basename}_first_frame_bbox.jpg")
                        visualized_img = visualize_bounding_boxes(
                            video_path,
                            result['grounded_spec'],
                            output_path=bbox_img_path
                        )
                        print(f"   🖼️  Saved bbox visualization: {bbox_img_path}")
                    except Exception as viz_err:
                        print(f"   ⚠️  Failed to save bbox visualization: {viz_err}")

                else:
                    print(f"❌ [{video_name}] FAILED: {result['reason']}")
                    results.append((video_path, False, result['reason']))

            except Exception as e:
                print(f"❌ [{video_name}] ERROR: {e}")
                results.append((video_path, False, f"Error: {e}"))

    # Extract survivors
    survivors = [video_path for video_path, passed, _ in results if passed]

    passed_count = len(survivors)
    failed_count = len(video_paths) - passed_count

    print(f"\n{'─'*80}")
    print(f"STAGE 0 COMPLETE:")
    print(f"   Passed: {passed_count} | Failed: {failed_count}")
    print(f"{'─'*80}")

    return survivors, grounded_specs


def multi_gpu_stage1_temporal_validation(
    video_paths: List[str],
    grounded_specs: Dict[str, Dict] = None,
    spec_json_str: str = None,
    num_gpus: int = 2,
    workers_per_gpu: int = 5,
    sam3_masks_dir: str = "sam3_masks",
    model_name: str = None,  # Deprecated - kept for backward compatibility
    gpu_models: Dict = None,  # Deprecated - kept for backward compatibility
    num_frames: int = None,  # Deprecated - kept for backward compatibility
    qwen_precheck: bool = False,
    qwen_keyframe_check: bool = True,
    qwen_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
) -> List[str]:
    """
    Stage 1: Multi-GPU SAM3-based contact detection and validation.

    Uses SAM3 (Segment Anything Model 3) to track subject and entities throughout
    the video using text prompts, then detects contact via mask overlap.

    Args:
        video_paths: List of video paths to validate
        grounded_specs: Dict mapping video_path -> grounded_spec from Stage 0 (optional if spec_json_str provided)
        spec_json_str: JSON specification string to use for ALL videos (if grounded_specs not provided)
        num_gpus: Number of GPUs to use for parallel processing
        workers_per_gpu: Number of worker threads per GPU
        sam3_masks_dir: Directory to save SAM3 mask visualizations (default: "sam3_masks")
        model_name: (Deprecated - kept for backward compatibility)
        gpu_models: (Deprecated - kept for backward compatibility)
        num_frames: (Deprecated - kept for backward compatibility)
        qwen_precheck: Run Qwen3-VL subject verification pre-filter (default: True). Set False to skip and avoid loading VLM.

    Returns:
        List of video paths that passed Stage 1 SAM3 contact detection
    """
    import time as _time
    _t_stage1_start = _time.time()
    print(f"\n{'='*80}")
    print(f"STAGE 1: SAM3 Contact Detection")
    print(f"{'='*80}")
    print(f"   Videos: {len(video_paths)}")
    print(f"   GPUs: {num_gpus}")
    print(f"   Workers per GPU: {workers_per_gpu}")
    print(f"   SAM3 Masks Directory: {sam3_masks_dir}")

    # Parse spec if provided as JSON string (when Stage 0 is skipped)
    default_spec = None
    if grounded_specs is None and spec_json_str is not None:
        try:
            default_spec = json.loads(spec_json_str)
            print(f"   📄 Using provided JSON spec for all videos (Stage 0 skipped)")
        except json.JSONDecodeError as e:
            print(f"   ❌ Failed to parse spec_json_str: {e}")
            return []

    # ── Stage 1a-pre: Subject verification using VLM on first 20% frames ────────
    # Reuse scorer's pre-loaded gpu_models if available; otherwise load fresh.
    # This avoids loading a second copy of Qwen3-VL-8B (~18GB per GPU) when the
    # scorer already loaded one for Stage 1b.
    _t_prefilter_start = _time.time()
    print(f"\n{'='*80}")
    print(f"STAGE 1 PRE-FILTER: Subject Verification (6 frames, first 20%)")
    if not qwen_precheck:
        print(f"   ⏭️  qwen_precheck=False — skipping VLM load and subject verification")
    print(f"{'='*80}")
    import threading as _threading
    qwen_gpu_locks  = {}
    if gpu_models and (qwen_precheck or qwen_keyframe_check):
        # Reuse scorer's pre-loaded models — no reload needed
        qwen_gpu_models = dict(gpu_models)
        for gpu_id in qwen_gpu_models:
            qwen_gpu_locks[gpu_id] = _threading.Lock()
        print(f"   ✓ Reusing {len(qwen_gpu_models)} pre-loaded VLM instance(s) (no reload)")
    else:
        qwen_gpu_models = {}
        if qwen_precheck or qwen_keyframe_check:
            for gpu_id in range(num_gpus):
                try:
                    m, p, d = load_stage1_model_on_gpu(gpu_id, qwen_model_name)
                    qwen_gpu_models[gpu_id] = (m, p, d)
                    qwen_gpu_locks[gpu_id]  = _threading.Lock()
                    print(f"   ✓ {qwen_model_name} loaded on GPU {gpu_id} ({d})")
                except Exception as exc:
                    print(f"   ⚠️  GPU {gpu_id} {qwen_model_name} load failed: {exc}")

    # Assign videos to GPUs (round-robin)
    video_gpu_assignments = [(video_path, i % num_gpus) for i, video_path in enumerate(video_paths)]

    # Create SAM3 masks directory
    os.makedirs(sam3_masks_dir, exist_ok=True)

    # Multi-GPU + Multi-threading
    results = []
    # Dict to store per-video data for Stage 1c tournament
    # {video_path: {"sam3": sam3_result, "spec": grounded_spec, "qwen": qwen_check}}
    passed_video_data: Dict[str, Dict] = {}

    prefilter_passed = 0
    prefilter_failed = 0

    with ThreadPoolExecutor(max_workers=num_gpus * workers_per_gpu) as executor:
        futures = {}

        # ── Pass 1: subject verification for ALL videos before SAM3 starts ──
        verified_videos = []  # (video_path, grounded_spec, gpu_id)
        for video_path, gpu_id in video_gpu_assignments:
            if grounded_specs is not None:
                grounded_spec = grounded_specs.get(video_path)
            else:
                grounded_spec = default_spec

            video_name = os.path.basename(video_path)

            if not grounded_spec:
                print(f"⚠️  [{video_name}] Skipping - no spec available")
                results.append((video_path, False, "No specification available"))
                prefilter_failed += 1
                continue

            if qwen_precheck and qwen_gpu_models:
                qm, qp, qd = qwen_gpu_models[gpu_id % len(qwen_gpu_models)]
                passed_verify, verify_reason = verify_subject_in_video(
                    video_path, grounded_spec, qm, qp, qd)
                if passed_verify:
                    print(f"   ✅ [{video_name}] Subject verified: {verify_reason}")
                    prefilter_passed += 1
                    verified_videos.append((video_path, grounded_spec, gpu_id))
                else:
                    print(f"   ❌ [{video_name}] Subject FAILED: {verify_reason}")
                    results.append((video_path, False, f"Subject verification failed: {verify_reason}"))
                    prefilter_failed += 1
            else:
                # Precheck disabled or VLM not loaded — skip verification, count as passed
                prefilter_passed += 1
                verified_videos.append((video_path, grounded_spec, gpu_id))

        # ── Pre-filter summary (before SAM3 starts) ───────────────────────────
        prefilter_total = prefilter_passed + prefilter_failed
        if prefilter_total > 0:
            rate = 100.0 * prefilter_passed / prefilter_total
            _t_prefilter = _time.time() - _t_prefilter_start
            print(f"\n{'─'*80}")
            print(f"PRE-FILTER (Subject Verification): "
                  f"{prefilter_passed}/{prefilter_total} passed ({rate:.1f}%)  [{_t_prefilter:.1f}s]")
            print(f"{'─'*80}\n")

        # ── Pass 2: submit SAM3 only for verified videos ───────────────────────
        for video_path, grounded_spec, gpu_id in verified_videos:
            device = f"cuda:{gpu_id}"
            future = executor.submit(
                detect_contact_with_sam3,
                video_path, grounded_spec, device, sam3_masks_dir
            )
            futures[future] = (video_path, grounded_spec)

        # ── Collect SAM3 results (segmentation only, no filtering) ──────────
        # sam3_done: list of (video_path, sam3_result_dict, grounded_spec)
        sam3_done = []
        for future in as_completed(futures):
            video_path, grounded_spec = futures[future]
            video_name = os.path.basename(video_path)
            try:
                result, raw_response = future.result()
                print(f"\n🎬 [{video_name}] SAM3 segmentation done")
                print(f"{'─'*60}")
                print(raw_response)
                print(f"{'─'*60}")
                print(f"   Key frames: {result.get('key_frames', [])}")
                print(f"   Tracked: {result.get('subject_name','')} + "
                      f"{', '.join(result.get('entity_names', []))}")
                sam3_done.append((video_path, result, grounded_spec))
            except Exception as e:
                print(f"❌ [{video_name}] SAM3 ERROR: {e}")
                import traceback; traceback.print_exc()
                results.append((video_path, False, f"SAM3 error: {e}"))

    # ── SAM3 survivor summary ─────────────────────────────────────────────────
    sam3_submitted = prefilter_passed  # only verified videos were submitted to SAM3
    sam3_passed    = len(sam3_done)
    sam3_errored   = sam3_submitted - sam3_passed
    _t_sam3_end = _time.time()
    _t_sam3_elapsed = _t_sam3_end - _t_prefilter_start
    print(f"\n{'─'*80}")
    print(f"SAM3 COMPLETE: {sam3_passed}/{sam3_submitted} passed"
          + (f"  |  {sam3_errored} errored" if sam3_errored else "")
          + f"  [{_t_sam3_elapsed:.1f}s]")
    print(f"   Advancing to Qwen check: {[os.path.basename(vp) for vp, _, _ in sam3_done]}")
    print(f"{'─'*80}")

    # ── Stage 1b: Qwen2.5-VL keyframe consistency check (all GPUs, parallel) ────
    _t_1b_start = _time.time()
    if sam3_done:
        print(f"\n{'='*80}")
        print(f"STAGE 1b: CV Keyframe Consistency Check ({len(sam3_done)} videos)  [SAM3: {_t_sam3_end - _t_prefilter_start:.1f}s]")
        print(f"{'='*80}")

        # Stage 1b: OpenCV histogram consistency — no model needed
        print(f"   Stage 1b uses OpenCV histogram check (no VLM required)")

        def _cv_check_worker(video_path, sam3_result, grounded_spec):
            return evaluate_sam3_keyframes_with_qwen(
                sam3_result=sam3_result,
                grounded_spec=grounded_spec,
                sam3_masks_dir=sam3_masks_dir,
                video_name=os.path.basename(video_path),
            )

        import os as _os
        num_cv_workers = min(len(sam3_done), max(1, _os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=num_cv_workers) as qwen_exec:
            qwen_futures = {}
            for vp, sr, gs in sam3_done:
                f = qwen_exec.submit(_cv_check_worker, vp, sr, gs)
                qwen_futures[f] = (vp, sr, gs)

            for f in as_completed(qwen_futures):
                video_path, sam3_result, grounded_spec_kf = qwen_futures[f]
                video_name = os.path.basename(video_path)
                try:
                    qwen_check = f.result()
                    if qwen_check["passed"]:
                        print(f"✅ [{video_name}] CV-KF PASSED: {qwen_check['reason']}")
                        results.append((video_path, True, qwen_check["reason"]))
                        passed_video_data[video_path] = {
                            "sam3": sam3_result,
                            "spec": grounded_spec_kf,
                            "qwen": qwen_check,
                        }
                    else:
                        print(f"❌ [{video_name}] CV-KF FAILED: {qwen_check['reason']}")
                        for kf in qwen_check.get("per_keyframe", []):
                            print(f"      kf={kf['key_frame_orig_idx']} "
                                  f"conf={kf.get('confidence', 0):.2f} "
                                  f"max_hist={kf.get('max_hist_dist', 0):.3f} "
                                  f"max_area={kf.get('max_area_ratio', 1):.1f} "
                                  f"| {kf['details']}")
                        results.append((video_path, False, qwen_check["reason"]))
                except Exception as exc:
                    import traceback
                    print(f"❌ [{video_name}] CV-KF error ({exc}) — rejecting video")
                    traceback.print_exc()
                    results.append((video_path, False, f"CV-KF error: {exc}"))
    else:
        # No SAM3 survivors — nothing to store for tournament
        pass

    # Extract survivors
    survivors = [video_path for video_path, passed, _ in results if passed]

    passed_count = len(survivors)
    failed_count = len(video_paths) - passed_count

    _t_1b_elapsed = _time.time() - _t_1b_start
    print(f"\n{'─'*80}")
    print(f"STAGE 1b COMPLETE:  [{_t_1b_elapsed:.1f}s]")
    print(f"   Passed: {passed_count} | Failed: {failed_count}")
    print(f"   Remaining videos: {[os.path.basename(p) for p in survivors]}")
    print(f"{'─'*80}")

    # ── Free VLM models before loading tournament LLM ────────────────────────
    # qwen_gpu_models either points to gpu_models (reused) or is a fresh dict.
    # Free via qwen_gpu_models only — covers both cases without double-freeing.
    import torch as _torch
    import gc as _gc
    if qwen_gpu_models:
        print(f"\n   Freeing {len(qwen_gpu_models)} VLM instance(s) from VRAM...")
        for gpu_id, (m, p, d) in list(qwen_gpu_models.items()):
            del m
            del p
        qwen_gpu_models.clear()
        if gpu_models:
            gpu_models.clear()  # sync the scorer's dict too (same models already deleted)
        _gc.collect()
        _torch.cuda.empty_cache()
        print(f"   ✓ VLMs freed")

    # ── Stage 1c: Tournament selection (physical plausibility ranking) ────────
    _t_1c_start = _time.time()
    if len(survivors) > 1 and passed_video_data:
        survivors = run_tournament_selection(
            survivors=survivors,
            passed_video_data=passed_video_data,
            tournament_model_name="Qwen/Qwen3-8B",
            batch_size=3,
            plots_dir=os.path.join(sam3_masks_dir, "bounce_plots"),
        )
    _t_1c_elapsed = _time.time() - _t_1c_start
    _t_total = _time.time() - _t_stage1_start

    print(f"\n{'─'*80}")
    print(f"STAGE 1 COMPLETE:  [1c tournament: {_t_1c_elapsed:.1f}s | total: {_t_total:.1f}s]")
    print(f"   Final selection: {[os.path.basename(p) for p in survivors]}")
    print(f"{'─'*80}")

    return survivors


def run_stage1_until_convergence(video_paths: List[str], text_prompt: str, checklist: str,
                                  num_gpus: int = 2, workers_per_gpu: int = 5,
                                  max_iterations: int = 5,
                                  model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
                                  gpu_models: Dict = None,
                                  run_stage0: bool = False,
                                  num_frames: int = 30,
                                  qwen_precheck: bool = False) -> List[str]:
    """
    Run complete Stage 0 + Stage 1 pipeline.

    Stage 0: First frame validation & grounding
    Stage 1: SAM3-based contact detection
      - Uses SAM3 to track subject and entities with text prompts
      - Detects contact frames via mask overlap
      - Saves mask visualizations

    Args:
        video_paths: List of all candidate video paths
        text_prompt: Original generation prompt (e.g., "A ball bounces on the floor")
        checklist: Prompt alignment checklist (JSON format from generate_prompt_alignment_checklist)
        num_gpus: Number of GPUs
        workers_per_gpu: Workers per GPU
        max_iterations: (Deprecated - kept for backward compatibility)
        model_name: Model name (used for Stage 0 only)
        gpu_models: Pre-loaded models for Stage 0 {gpu_id: (model, processor, device)} (optional)
        run_stage0: Run Stage 0 first frame validation (default: True)
        num_frames: (Deprecated - kept for backward compatibility)

    Returns:
        List of video paths that passed all stages
    """
    # Load models for Stage 0 if not provided (backward compatibility)
    # Note: Stage 1 (SAM3) loads its own models and doesn't need these
    if gpu_models is None and run_stage0:
        print(f"\n   📥 Loading models on {num_gpus} GPUs for Stage 0 (one-time setup)...")
        gpu_models = {}
        for gpu_id in range(num_gpus):
            model, processor, device = load_stage1_model_on_gpu(gpu_id, model_name)
            gpu_models[gpu_id] = (model, processor, device)
        print(f"   ✓ Models loaded")
    elif gpu_models is not None:
        print(f"\n   ✅ Using pre-loaded models from scorer (no reload needed)")

    # 🆕 STAGE 0: First Frame Validation & Grounding (optional)
    grounded_specs = None
    if run_stage0:
        survivors_stage0, grounded_specs = multi_gpu_stage0_first_frame_validation(
            video_paths, checklist, num_gpus, workers_per_gpu, model_name, gpu_models
        )

        if len(survivors_stage0) == 0:
            print(f"\n❌ Stage 0 eliminated ALL videos")
            return []

        # Update video list to only Stage 0 survivors
        video_paths = survivors_stage0
        print(f"\n✓ Stage 0 Complete - {len(video_paths)} videos advance to Stage 1")
    else:
        print(f"\n⚠️  Stage 0 SKIPPED - using original JSON spec for all videos")

    # STAGE 1: SAM3 Contact Detection
    # Place sam3_masks inside the videos directory
    videos_dir = os.path.dirname(os.path.abspath(video_paths[0])) if video_paths else "."
    sam3_masks_dir = os.path.join(videos_dir, "sam3_masks")

    survivors_stage1 = multi_gpu_stage1_temporal_validation(
        video_paths,
        grounded_specs=grounded_specs,
        spec_json_str=checklist if not run_stage0 else None,
        num_gpus=num_gpus,
        workers_per_gpu=workers_per_gpu,
        sam3_masks_dir=sam3_masks_dir,
        qwen_precheck=qwen_precheck,
        # Deprecated parameters kept for backward compatibility
        model_name=model_name,
        gpu_models=gpu_models,
        num_frames=num_frames
    )

    # Return final survivors
    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETE:")
    print(f"   Final survivors: {len(survivors_stage1)}")
    print(f"{'='*80}")

    return survivors_stage1
