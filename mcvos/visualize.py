"""
OW-VIS Visualization Utilities.

Provides clean, paper-quality visualization of detection and segmentation results.
"""

import numpy as np
import cv2
from typing import List, Optional, Dict, Tuple


# Color scheme: known=green, unknown=red
KNOWN_COLOR = (0, 200, 0)      # green (BGR for cv2)
UNKNOWN_COLOR = (0, 0, 220)    # red (BGR)
TEXT_BG_COLOR = (0, 0, 0)      # black text background
TEXT_COLOR = (255, 255, 255)   # white text

# Palette for mask overlays (distinct colors for up to 32 instances)
INSTANCE_PALETTE = [
    (255, 56, 56),   (255, 157, 151), (255, 112, 31),  (255, 178, 29),
    (207, 210, 49),  (72, 249, 10),   (146, 204, 23),  (61, 219, 134),
    (26, 147, 52),   (0, 212, 187),   (44, 153, 168),  (0, 194, 255),
    (52, 69, 147),   (100, 115, 255), (0, 24, 236),    (132, 56, 255),
    (82, 0, 133),    (203, 56, 255),  (255, 149, 200), (255, 55, 199),
    (220, 70, 70),   (70, 220, 70),   (70, 70, 220),   (220, 220, 70),
    (220, 70, 220),  (70, 220, 220),  (180, 130, 70),  (70, 180, 130),
    (130, 70, 180),  (180, 180, 70),  (70, 130, 180),  (130, 180, 70),
]


def _get_instance_color(instance_id: int) -> Tuple[int, int, int]:
    """Get a consistent BGR color for a given instance ID."""
    color_rgb = INSTANCE_PALETTE[instance_id % len(INSTANCE_PALETTE)]
    return (color_rgb[2], color_rgb[1], color_rgb[0])  # RGB -> BGR


def draw_detections(
    frame_np: np.ndarray,
    known_boxes: np.ndarray,
    known_labels: List[str],
    unknown_boxes: np.ndarray,
    unknown_ids: List[int],
    masks: Optional[Dict[int, np.ndarray]] = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Draw detection results on a frame.

    - Green boxes + label text for known detections
    - Red boxes + "UNK#ID" text for unknown detections
    - If masks provided: colored semi-transparent overlay per instance
    - Returns annotated frame as numpy array (BGR)
    """
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    overlay = frame_bgr.copy()

    # Draw mask overlays first (behind boxes)
    if masks:
        for obj_id, mask in masks.items():
            if mask is None or not mask.any():
                continue
            color_bgr = _get_instance_color(abs(obj_id))
            colored_mask = np.zeros_like(frame_bgr)
            colored_mask[mask > 0] = color_bgr
            overlay = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color_bgr, 1)

    # Draw known detections (green boxes)
    for box, label in zip(known_boxes, known_labels):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), KNOWN_COLOR, 2)
        label_text = str(label)
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(overlay, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), KNOWN_COLOR, -1)
        cv2.putText(overlay, label_text, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_BG_COLOR, 1, cv2.LINE_AA)

    # Draw unknown detections (red boxes + ID)
    for box, inst_id in zip(unknown_boxes, unknown_ids):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = _get_instance_color(inst_id)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), UNKNOWN_COLOR, 2)
        id_text = f"UNK#{inst_id}"
        (tw, th), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), UNKNOWN_COLOR, -1)
        cv2.putText(overlay, id_text, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_BG_COLOR, 1, cv2.LINE_AA)
        cv2.circle(overlay, (x2 - 6, y1 + 6), 5, color, -1)
        cv2.circle(overlay, (x2 - 6, y1 + 6), 5, (255, 255, 255), 1)

    return overlay


def create_summary_grid(
    frames_list: List[np.ndarray],
    n_cols: int = 5,
    cell_size: Tuple[int, int] = (320, 180),
) -> np.ndarray:
    """Stack annotated frames into a grid image."""
    if not frames_list:
        return np.zeros((180, 320, 3), dtype=np.uint8)

    n = len(frames_list)
    n_rows = (n + n_cols - 1) // n_cols
    W, H = cell_size

    grid = np.zeros((n_rows * H, n_cols * W, 3), dtype=np.uint8)
    for i, frame in enumerate(frames_list):
        r = i // n_cols
        c = i % n_cols
        thumb = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f"F{i:03d}", (4, H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = thumb

    return grid


def save_demo_video(
    annotated_frames: List[np.ndarray],
    output_path: str,
    fps: float = 10.0,
) -> None:
    """Save annotated frames as an MP4 video."""
    if not annotated_frames:
        print("[WARN] No frames to save.")
        return

    H, W = annotated_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for frame in annotated_frames:
        writer.write(frame)
    writer.release()
    print(f"[VIZ] Saved demo video: {output_path} ({len(annotated_frames)} frames @ {fps}fps)")
