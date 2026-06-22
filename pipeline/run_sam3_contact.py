#!/usr/bin/env python3
"""
Standalone SAM3 contact detection script.
Runs in the `sam3` conda environment as a subprocess.

Two-pass approach:
  Pass 1 (coarse): Sample at ~3fps, track subject+entity positions, find top-K
                   key frames by minimum contact distance.
  Pass 2 (fine):   For each key frame, run a sliding window of
                   [key-W .. key .. key+W] original frames through SAM3,
                   record per-frame positions, save extraction images.

Outputs JSON to stdout: {"result": {...}, "raw_response": "..."}
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


# ── helpers ──────────────────────────────────────────────────────────────────

def get_mask_np(mask):
    if hasattr(mask, 'cpu'):
        mask = mask.cpu().numpy()
    else:
        mask = np.array(mask)
    return mask > 0.5


def mask_brightness(frame_raw, mask_np):
    """Mean brightness of pixels under mask_np (0–255)."""
    try:
        if hasattr(frame_raw, 'convert'):
            frame_np = np.array(frame_raw.convert('RGB'))
        else:
            frame_np = np.array(frame_raw)
            if frame_np.ndim == 2:
                frame_np = np.stack([frame_np] * 3, axis=-1)
            elif frame_np.shape[2] == 4:
                frame_np = frame_np[:, :, :3]
        pixels = frame_np[mask_np]
        if len(pixels) == 0:
            return 0.0
        return float(pixels.mean())
    except Exception:
        return 0.0


def filter_shadow_masks(masks, frame_raw, expected_count, frame_idx=None, debug=True,
                        subject_name=None, entity_names=None):
    """
    Keep the N largest masks (by pixel area) and preserve their original order.

    Shadows/reflections are always smaller than the real objects (ball, floor),
    so dropping the smallest masks correctly removes them regardless of order.
    Original index order is preserved so prompt-slot assignment stays correct.
    """
    if masks is None or len(masks) == 0:
        return masks

    tag = f"f{frame_idx} " if frame_idx is not None else ""
    labels = ([subject_name] if subject_name else ["subject"]) + (entity_names or [])

    # Compute pixel area for each mask
    areas = []
    for i, mask in enumerate(masks):
        m = get_mask_np(mask)
        rows = np.where(m.any(axis=1))[0]
        cols = np.where(m.any(axis=0))[0]
        bbox = (int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])) if len(rows) else None
        px_count = int(m.sum())
        areas.append(px_count)
        if debug:
            print(f"      {tag}mask[{i}]: bbox={bbox}  pixels={px_count}",
                  file=sys.stderr)

    if len(masks) > expected_count:
        # Find indices of the N largest masks
        ranked = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)
        keep_set = set(ranked[:expected_count])
        dropped  = [i for i in range(len(masks)) if i not in keep_set]
        if debug:
            print(f"      {tag}dropping mask(s) {dropped} "
                  f"(areas {[areas[i] for i in dropped]}) as shadow/extra",
                  file=sys.stderr)
        # Preserve original order among kept masks
        masks = [masks[i] for i in range(len(masks)) if i in keep_set]
        areas = [areas[i] for i in range(len(areas)) if i in keep_set]

    if debug:
        for li, px in zip(labels[:len(masks)], areas):
            print(f"      {tag}→ '{li}' pixels={px}", file=sys.stderr)

    return masks


def mask_stats(mask_np, label):
    """Return top/bottom/center_of_mass for a binary mask."""
    rows = np.where(mask_np.any(axis=1))[0]
    if len(rows) == 0:
        return {f"{label}_top": None, f"{label}_bottom": None,
                f"{label}_center_of_mass": None}
    ys, xs = np.where(mask_np)
    return {
        f"{label}_top":            int(rows[0]),
        f"{label}_bottom":         int(rows[-1]),
        f"{label}_center_of_mass": [int(xs.mean()), int(ys.mean())],
    }


def frame_entry(label_masks, subject_name, entity_names, frame_orig_idx,
                sampled_frame_idx=None, is_key=False):
    """Build one trajectory entry from label-keyed SAM3 mask dict."""
    entry = {"frame": frame_orig_idx, "is_key_frame": is_key}
    if sampled_frame_idx is not None:
        entry["sampled_frame"] = sampled_frame_idx

    for lbl in [subject_name] + list(entity_names):
        m = label_masks.get(lbl) if label_masks else None
        if m is not None:
            entry.update(mask_stats(get_mask_np(m), lbl))
        else:
            entry.update({f"{lbl}_top": None, f"{lbl}_bottom": None,
                          f"{lbl}_center_of_mass": None})

    s_bot = entry.get(f"{subject_name}_bottom")
    e_top = entry.get(f"{entity_names[0]}_top") if entity_names else None
    entry["contact_distance_px"] = (e_top - s_bot) if (
        s_bot is not None and e_top is not None) else None

    return entry


def run_sam3_session(model, processor, frames, subject_name, entity_names, device):
    """
    Run one SAM3 inference session. Returns per-frame dicts keyed by label name:
      { frame_idx: { "ball": mask_np, "floor": mask_np, ... } }

    Uses prompt_to_obj_ids from SAM3 output to correctly map each mask to its
    prompted label, regardless of the order SAM3 internally tracks them.
    """
    all_labels = [subject_name] + list(entity_names)

    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    session = processor.add_text_prompt(inference_session=session, text=subject_name)
    for ename in entity_names:
        session = processor.add_text_prompt(inference_session=session, text=ename)

    outputs = {}
    _debug_printed = False
    for out in model.propagate_in_video_iterator(
        inference_session=session,
        max_frame_num_to_track=len(frames)
    ):
        processed = processor.postprocess_outputs(session, out)

        if not _debug_printed:
            # Print prompt_to_obj_ids once to verify label→mask mapping
            p2o = processed.get("prompt_to_obj_ids", {})
            obj_ids = processed.get("object_ids")
            print(f"      [SAM3] prompt_to_obj_ids={p2o}  object_ids={obj_ids}",
                  file=sys.stderr)
            _debug_printed = True

        # Build label → mask mapping using prompt_to_obj_ids
        p2o = processed.get("prompt_to_obj_ids", {})
        obj_ids = processed.get("object_ids")   # tensor of tracked IDs
        masks_tensor = processed.get("masks")   # tensor [N, H, W]

        label_masks = {}
        if p2o and obj_ids is not None and masks_tensor is not None:
            obj_ids_list = obj_ids.cpu().tolist()
            for label in all_labels:
                # p2o values are lists e.g. {'ball': [0], 'floor': [1]}
                # or with shadow: {'ball': [0, 2], 'floor': [1]}
                candidate_ids = p2o.get(label)
                if candidate_ids is None:
                    idx = all_labels.index(label)
                    candidate_ids = p2o.get(idx)
                # Normalise to list
                if candidate_ids is None:
                    candidate_ids = []
                elif not isinstance(candidate_ids, (list, tuple)):
                    candidate_ids = [candidate_ids]

                # Find mask indices for all candidate obj_ids
                candidate_mask_indices = [
                    obj_ids_list.index(oid)
                    for oid in candidate_ids
                    if oid in obj_ids_list
                ]

                if not candidate_mask_indices:
                    print(f"      [SAM3] WARNING: label '{label}' has no valid "
                          f"obj_ids in {obj_ids_list} (p2o entry={candidate_ids})",
                          file=sys.stderr)
                    label_masks[label] = None
                elif len(candidate_mask_indices) == 1:
                    label_masks[label] = masks_tensor[candidate_mask_indices[0]]
                elif len(candidate_mask_indices) == 2:
                    # Two candidates — always keep the largest (real object), drop the smaller.
                    # Brightness-based rejection is unreliable for multicolored objects.
                    # Qwen downstream handles any remaining quality issues.
                    best_idx = max(candidate_mask_indices,
                                   key=lambda mi: int(get_mask_np(masks_tensor[mi]).sum()))
                    dropped = [i for i in candidate_mask_indices if i != best_idx]
                    print(f"      [SAM3] '{label}' has 2 candidates "
                          f"→ keeping mask[{best_idx}] (largest), dropping {dropped}",
                          file=sys.stderr)
                    label_masks[label] = masks_tensor[best_idx]
                else:
                    # 3+ candidates = severe hallucination — abort this video
                    print(f"      [SAM3] HALLUCINATION: label '{label}' has "
                          f"{len(candidate_mask_indices)} candidates {candidate_mask_indices} "
                          f"— too many objects, rejecting video",
                          file=sys.stderr)
                    raise ValueError(
                        f"SAM3 hallucination: '{label}' tracked "
                        f"{len(candidate_mask_indices)} objects (expected 1)"
                    )
        else:
            # Fallback: prompt order matches mask order
            print(f"      [SAM3] WARNING: no prompt_to_obj_ids — falling back to "
                  f"prompt order. p2o={p2o} obj_ids={obj_ids} masks={masks_tensor is not None}",
                  file=sys.stderr)
            for i, label in enumerate(all_labels):
                if masks_tensor is not None and i < len(masks_tensor):
                    label_masks[label] = masks_tensor[i]
                else:
                    label_masks[label] = None

        outputs[out.frame_idx] = label_masks
    return outputs


# ── visualisation ─────────────────────────────────────────────────────────────

def extract_object_images(frame_raw, label_masks, subject_name, entity_names,
                          video_name, frame_orig_idx, key_frame_orig_idx, save_dir):
    """
    For each object (subject + entities), save a masked crop and overlay only (no raw crop).
    label_masks: dict {label: mask_tensor_or_None}
    Returns: {label: {"masked_crop": rel_path}, "_overlay": rel_path}
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        if hasattr(frame_raw, 'convert'):
            frame_np = np.array(frame_raw.convert('RGB'))
        else:
            frame_np = np.array(frame_raw)
            if frame_np.ndim == 2:
                frame_np = np.stack([frame_np]*3, axis=-1)
            elif frame_np.shape[2] == 4:
                frame_np = frame_np[:, :, :3]

        H, W = frame_np.shape[:2]
        object_labels = [subject_name] + list(entity_names)
        colors = [[255, 80, 80], [80, 200, 80], [80, 120, 255], [255, 220, 50]]
        is_key = (frame_orig_idx == key_frame_orig_idx)

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        except Exception:
            font = ImageFont.load_default()

        paths = {}

        # Overlay — colour each label's mask
        overlay = frame_np.astype(np.float32)
        for i, lbl in enumerate(object_labels):
            mask = label_masks.get(lbl) if label_masks else None
            if mask is None:
                continue
            m = get_mask_np(mask)
            c = colors[i % len(colors)]
            co = np.zeros_like(overlay)
            co[:, :] = c
            overlay[m] = overlay[m] * 0.5 + co[m] * 0.5
        overlay_img = Image.fromarray(overlay.astype(np.uint8))
        draw = ImageDraw.Draw(overlay_img)
        tag = "[KEY] " if is_key else ""
        draw.text((6, 6), f"{tag}f{frame_orig_idx}", fill="yellow" if is_key else "white", font=font)
        for i, lbl in enumerate(object_labels):
            draw.text((6, 24 + i * 16),
                      f"{['R','G','B','Y'][i%4]}: {lbl}", fill="white", font=font)
        overlay_fname = f"{video_name}_kf{key_frame_orig_idx:04d}_f{frame_orig_idx:04d}_overlay.jpg"
        overlay_path = os.path.join(save_dir, overlay_fname)
        overlay_img.save(overlay_path)
        paths["_overlay"] = os.path.relpath(overlay_path, os.path.dirname(save_dir))

        # Per-object masked crops only (no raw crop saved)
        for label in object_labels:
            mask = label_masks.get(label) if label_masks else None
            if mask is None:
                paths[label] = {"masked_crop": None}
                continue
            m = get_mask_np(mask)
            rows = np.where(m.any(axis=1))[0]
            cols = np.where(m.any(axis=0))[0]
            if len(rows) == 0 or len(cols) == 0:
                paths[label] = {"masked_crop": None}
                continue

            pad = 8
            y1 = max(0, int(rows[0])  - pad)
            y2 = min(H, int(rows[-1]) + pad)
            x1 = max(0, int(cols[0])  - pad)
            x2 = min(W, int(cols[-1]) + pad)

            masked_np = frame_np.copy()
            masked_np[~m] = 0
            masked_img = Image.fromarray(masked_np[y1:y2, x1:x2])
            ImageDraw.Draw(masked_img).text((4, 4), f"{label} | f{frame_orig_idx}",
                                            fill="white", font=font)
            masked_fname = (f"{video_name}_kf{key_frame_orig_idx:04d}"
                            f"_f{frame_orig_idx:04d}_{label}_masked.jpg")
            masked_path = os.path.join(save_dir, masked_fname)
            masked_img.save(masked_path)

            paths[label] = {
                "masked_crop": os.path.relpath(masked_path, os.path.dirname(save_dir)),
            }

        return paths

    except Exception as e:
        import traceback
        print(f"      Warning: object extraction f{frame_orig_idx} failed: {e}",
              file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {}


def save_trajectory_plot(video_name, coarse_traj, subject_name, entity_names,
                         contact_thresh_px, key_frame_orig_indices, save_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(save_dir, exist_ok=True)

        frames      = [e["frame"] for e in coarse_traj]
        subj_cy     = [(e.get(f"{subject_name}_center_of_mass") or [None, None])[1]
                       for e in coarse_traj]
        subj_cy     = [v if v is not None else float('nan') for v in subj_cy]
        subj_bot    = [e.get(f"{subject_name}_bottom") or float('nan') for e in coarse_traj]
        ent_top     = ([e.get(f"{entity_names[0]}_top") or float('nan') for e in coarse_traj]
                       if entity_names else [])
        distances   = [e.get("contact_distance_px") if e.get("contact_distance_px") is not None
                       else float('nan') for e in coarse_traj]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        ax1.plot(frames, subj_cy,  color="royalblue",  linewidth=1.5,
                 label=f"{subject_name} center Y")
        ax1.plot(frames, subj_bot, color="dodgerblue", linewidth=1, linestyle="--",
                 label=f"{subject_name} bottom Y")
        if ent_top:
            ax1.plot(frames, ent_top, color="tomato", linewidth=1.5,
                     label=f"{entity_names[0]} top Y")

        kf_set = set(key_frame_orig_indices)
        kf_x = [e["frame"] for e in coarse_traj if e["frame"] in kf_set]
        kf_y = [(e.get(f"{subject_name}_center_of_mass") or [None, None])[1]
                for e in coarse_traj if e["frame"] in kf_set]
        kf_y = [v if v is not None else float('nan') for v in kf_y]
        ax1.scatter(kf_x, kf_y, color="red", zorder=5, s=60, label="Key frames")
        ax1.invert_yaxis()
        ax1.set_ylabel("Y position (px, ↓)")
        ax1.set_title(
            f"SAM3 Trajectory: {subject_name} ∩ {', '.join(entity_names)}\n{video_name}")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        ax2.plot(frames, distances, color="darkorange", linewidth=1.5,
                 label="Contact distance (px)")
        ax2.axhline(contact_thresh_px, color="red", linewidth=1, linestyle="--",
                    label=f"Threshold ({contact_thresh_px}px)")
        ax2.axhline(0, color="gray", linewidth=0.5)
        ax2.fill_between(frames, distances, contact_thresh_px,
                         where=[d <= contact_thresh_px for d in distances],
                         alpha=0.3, color="red", label="Contact zone")
        ax2.set_xlabel("Frame index")
        ax2.set_ylabel("Distance (px)")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        out = os.path.join(save_dir, f"{video_name}_trajectory.jpg")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"      Trajectory plot saved: {out}", file=sys.stderr)
    except Exception as e:
        print(f"      Warning: trajectory plot failed: {e}", file=sys.stderr)


# ── main run ──────────────────────────────────────────────────────────────────

def run(video_path, grounded_spec, device, save_dir, fps=3.0, top_k=5, window=4, last_phase_n=5):
    # HF auth — HF_TOKEN env var is picked up automatically by huggingface_hub.
    # Explicit login() is skipped because newer versions interpret the token value
    # as a stored token name rather than a raw token string.
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        print(f"      HF_TOKEN set, will be used automatically", file=sys.stderr)
    else:
        print(f"      Warning: HF_TOKEN not set", file=sys.stderr)

    from transformers import Sam3VideoModel, Sam3VideoProcessor
    from transformers.video_utils import load_video

    print(f"\n   SAM3 contact detection: {os.path.basename(video_path)}", file=sys.stderr)

    # Parse entity names from spec
    entities = grounded_spec.get("entities", {})
    subject_name = entities.get("subject", {}).get("name", "subject")

    phases = grounded_spec.get("action_phases", {}).get("phases", [])
    entity_names = []
    for p in phases:
        pname = p.get("phase_name", "")
        if pname not in ("initial", "final") and '_' in pname:
            ename = pname.split('_', 1)[1]
            if ename not in entity_names:
                entity_names.append(ename)
    if not entity_names:
        for ent in entities.get("interactive_entities", []):
            n = ent.get("name")
            if n and n not in entity_names:
                entity_names.append(n)

    if not entity_names:
        result_dict = {"passed": False,
                       "reason": "No contact entities found in spec",
                       "num_frames_checked": 0, "frame_indices": [],
                       "contact_frames": [], "subject_name": subject_name,
                       "entity_names": []}
        return result_dict, "SAM3: no entities found"

    print(f"      Subject: {subject_name}  Entities: {', '.join(entity_names)}",
          file=sys.stderr)

    # Load model once
    print(f"      Loading SAM3 model...", file=sys.stderr)
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

    # Load all frames
    print(f"      Loading video frames...", file=sys.stderr)
    all_frames, _ = load_video(video_path)
    total_loaded = len(all_frames)

    import cv2 as _cv2
    cap = _cv2.VideoCapture(video_path)
    video_fps_native = cap.get(_cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    step = max(1, round(video_fps_native / fps))
    sampled_indices = list(range(0, total_loaded, step))
    coarse_frames = [all_frames[i] for i in sampled_indices]
    frame_indices_map = {i: sampled_indices[i] for i in range(len(sampled_indices))}

    print(f"      {total_loaded} frames → {len(coarse_frames)} coarse at ~{fps}fps (step={step})",
          file=sys.stderr)

    # ── Pass 1: coarse trajectory ─────────────────────────────────────────────
    print(f"      Pass 1: coarse propagation...", file=sys.stderr)
    coarse_outputs = run_sam3_session(model, processor, coarse_frames,
                                      subject_name, entity_names, device)

    contact_thresh_px = 15
    coarse_traj = []
    for si in sorted(coarse_outputs.keys()):
        label_masks = coarse_outputs[si]   # {label: mask_tensor}
        orig_idx = frame_indices_map[si]
        entry = frame_entry(label_masks, subject_name, entity_names, orig_idx,
                            sampled_frame_idx=si)
        coarse_traj.append(entry)

    # Select top-K key frames: divide trajectory into K bins, pick best (smallest
    # contact distance) from each bin — guarantees temporal spread across video.
    has_dist = [(e["contact_distance_px"], e["frame"], e["sampled_frame"])
                for e in coarse_traj if e["contact_distance_px"] is not None]
    top_k = min(top_k, len(coarse_traj))
    if has_dist:
        n = len(has_dist)
        bin_size = max(1, n // top_k)
        key_frame_orig = []
        key_frame_samp = []
        for b in range(top_k):
            start = b * bin_size
            end   = start + bin_size if b < top_k - 1 else n  # last bin gets remainder
            bin_entries = has_dist[start:end]
            if bin_entries:
                best = min(bin_entries, key=lambda t: t[0])
                key_frame_orig.append(best[1])
                key_frame_samp.append(best[2])
    else:
        key_frame_orig = [e["frame"] for e in coarse_traj[-top_k:]]
        key_frame_samp = [e["sampled_frame"] for e in coarse_traj[-top_k:]]

    distances = [e["contact_distance_px"] for e in coarse_traj
                 if e["contact_distance_px"] is not None]
    min_dist = min(distances) if distances else None
    contact_frames = [e["sampled_frame"] for e in coarse_traj
                      if e["contact_distance_px"] is not None
                      and e["contact_distance_px"] <= contact_thresh_px]

    print(f"      Distance range: {min(distances) if distances else 'N/A'} – "
          f"{max(distances) if distances else 'N/A'} px", file=sys.stderr)
    print(f"      Key frames (orig): {sorted(key_frame_orig)}", file=sys.stderr)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    kf_info_dir = os.path.join(save_dir, "key_frame_info") if save_dir else None
    if kf_info_dir:
        os.makedirs(kf_info_dir, exist_ok=True)

    # ── Pass 2: sliding window around each key frame ──────────────────────────
    key_frame_analysis = []

    for kf_orig, kf_samp in zip(key_frame_orig, key_frame_samp):
        print(f"      Pass 2: sliding window around frame {kf_orig} "
              f"(±{window} frames)...", file=sys.stderr)

        win_orig_indices = list(range(
            max(0, kf_orig - window),
            min(total_loaded, kf_orig + window + 1)
        ))
        win_frames = [all_frames[i] for i in win_orig_indices]
        win_idx_map = {wi: win_orig_indices[wi] for wi in range(len(win_orig_indices))}

        win_outputs = run_sam3_session(model, processor, win_frames,
                                       subject_name, entity_names, device)

        win_traj = []
        for wi in sorted(win_outputs.keys()):
            label_masks = win_outputs[wi]   # {label: mask_tensor}
            orig_fi = win_idx_map[wi]
            is_key = (orig_fi == kf_orig)
            entry = frame_entry(label_masks, subject_name, entity_names, orig_fi,
                                is_key=is_key)

            # Extract per-object crops and store paths in entry
            if kf_info_dir and label_masks:
                img_paths = extract_object_images(
                    all_frames[orig_fi], label_masks, subject_name, entity_names,
                    video_name, orig_fi, kf_orig, kf_info_dir
                )
                # Store as: entry["images"]["ball"] = {"crop": ..., "masked_crop": ...}
                entry["images"] = img_paths
            else:
                entry["images"] = {}

            win_traj.append(entry)

        kf_coarse = next((e for e in coarse_traj if e["frame"] == kf_orig), {})
        key_frame_analysis.append({
            "key_frame_orig_idx":      kf_orig,
            "key_frame_sampled_idx":   kf_samp,
            "coarse_contact_distance_px": kf_coarse.get("contact_distance_px"),
            "sliding_window_trajectory": win_traj,
        })

    # ── Pass 3: last phase — last N frames of the video at full FPS ─────────
    last_phase_traj = []
    if total_loaded > 0:
        last_phase_indices = list(range(max(0, total_loaded - last_phase_n), total_loaded))
        last_phase_frames  = [all_frames[i] for i in last_phase_indices]
        print(f"      Pass 3: last phase — last {len(last_phase_frames)} frames "
              f"(f={last_phase_indices[0]}–{last_phase_indices[-1]})...", file=sys.stderr)
        lp_outputs = run_sam3_session(model, processor, last_phase_frames,
                                      subject_name, entity_names, device)
        lp_idx_map = {wi: last_phase_indices[wi] for wi in range(len(last_phase_indices))}
        for wi in sorted(lp_outputs.keys()):
            orig_fi = lp_idx_map[wi]
            label_masks = lp_outputs[wi]
            entry = frame_entry(label_masks, subject_name, entity_names, orig_fi)
            if kf_info_dir and label_masks:
                img_paths = extract_object_images(
                    all_frames[orig_fi], label_masks, subject_name, entity_names,
                    video_name, orig_fi, orig_fi, kf_info_dir
                )
                entry["images"] = img_paths
            else:
                entry["images"] = {}
            last_phase_traj.append(entry)

    # ── Save outputs ──────────────────────────────────────────────────────────
    if save_dir:
        traj_path = os.path.join(save_dir, f"{video_name}_trajectory.json")
        try:
            with open(traj_path, "w") as f:
                json.dump({
                    "video":               os.path.basename(video_path),
                    "subject":             subject_name,
                    "entities":            entity_names,
                    "fps_sampled":         fps,
                    "contact_threshold_px": contact_thresh_px,
                    "coarse_trajectory":   coarse_traj,
                    "key_frame_analysis":  key_frame_analysis,
                    "last_phase_trajectory": last_phase_traj,
                }, f, indent=2)
            print(f"      Trajectory JSON saved: {traj_path}", file=sys.stderr)
        except Exception as e:
            print(f"      Warning: trajectory JSON failed: {e}", file=sys.stderr)

        save_trajectory_plot(video_name, coarse_traj, subject_name, entity_names,
                             contact_thresh_px, key_frame_orig, save_dir)

    # ── Result ────────────────────────────────────────────────────────────────
    total_frames = len(coarse_traj)
    # SAM3 is segmentation only for now — always passes.
    # Filtering will be done downstream by Qwen3-VL keyframe check.
    passed = True
    reason = (
        f"SAM3 segmentation complete: {total_frames} coarse frames, "
        f"{len(key_frame_analysis)} key frames extracted "
        f"(min contact dist={min_dist}px)"
    )

    result_dict = {
        "passed":                  passed,
        "reason":                  reason,
        "num_frames_checked":      total_frames,
        "contact_frames":          contact_frames,
        "key_frames":              sorted(key_frame_orig),
        "frame_indices":           sorted(key_frame_orig),
        "subject_name":            subject_name,
        "entity_names":            entity_names,
        "min_contact_distance_px": min_dist,
        "contact_threshold_px":    contact_thresh_px,
        "coarse_trajectory":       coarse_traj,
        "key_frame_analysis":      key_frame_analysis,
        "last_phase_trajectory":   last_phase_traj,
    }

    kf_str = ", ".join(
        f"f{kf['key_frame_orig_idx']}={kf['coarse_contact_distance_px']}px"
        for kf in key_frame_analysis)
    raw_response = (
        f"SAM3 Trajectory & Contact Detection\n"
        f"{'='*80}\n"
        f"Video: {os.path.basename(video_path)}\n"
        f"Coarse frames: {total_frames} (~{fps}fps)  "
        f"Key frames: {top_k} (±{window} sliding window)\n\n"
        f"Tracked: {subject_name} → {', '.join(entity_names)}\n\n"
        f"Min contact distance: {min_dist}px  (threshold={contact_thresh_px}px)\n"
        f"Contact frames: {len(contact_frames)}/{total_frames}\n\n"
        f"Key frames: {kf_str}\n\n"
        f"Result: {'PASS' if passed else 'FAIL'}\n"
    )

    return result_dict, raw_response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",     required=True)
    parser.add_argument("--grounded_spec",  required=True, help="JSON string")
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--save_dir",       default=None)
    parser.add_argument("--fps",            type=float, default=3.0,
                        help="Coarse sampling rate in fps (default: 3.0)")
    parser.add_argument("--top_k",          type=int,   default=5,
                        help="Number of key frames by closest contact (default: 5)")
    parser.add_argument("--window",         type=int,   default=3,
                        help="Sliding window half-size around each key frame (default: 4)")
    args = parser.parse_args()

    grounded_spec = json.loads(args.grounded_spec)
    result_dict, raw_response = run(
        args.video_path, grounded_spec, args.device, args.save_dir,
        fps=args.fps, top_k=args.top_k, window=args.window)

    print(json.dumps({"result": result_dict, "raw_response": raw_response}))


if __name__ == "__main__":
    main()
