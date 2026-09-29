"""
OW-VIS Evaluation Metrics.

Provides J&F score, unknown recall, and ID consistency metrics
for evaluating open-world video instance segmentation.
"""

import numpy as np
from typing import Dict, List, Optional

try:
    from scipy.ndimage import binary_dilation, binary_erosion
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def compute_j_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Jaccard index (IoU) for a single instance mask."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def _mask_to_boundary(mask: np.ndarray, boundary_thresh: float = 0.008) -> np.ndarray:
    """Convert binary mask to boundary pixels."""
    h, w = mask.shape
    diag = np.sqrt(h * h + w * w)
    radius = max(1, int(np.round(boundary_thresh * diag)))
    if _SCIPY_AVAILABLE:
        struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        dilated = binary_dilation(mask, structure=struct)
        eroded = binary_erosion(mask, structure=struct)
        boundary = dilated & ~eroded
    else:
        # Fallback: use numpy-based erosion
        from numpy.lib.stride_tricks import sliding_window_view
        pad = radius
        padded = np.pad(mask.astype(float), pad, mode='constant')
        boundary = np.zeros_like(mask, dtype=bool)
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                rolled = padded[pad+dy:pad+dy+h, pad+dx:pad+dx+w].astype(bool)
                boundary |= (rolled != mask.astype(bool))
        boundary = boundary & mask.astype(bool)
    return boundary.astype(bool)


def compute_f_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    boundary_thresh: float = 0.008,
) -> float:
    """F-measure on contour accuracy."""
    pred_boundary = _mask_to_boundary(pred_mask.astype(bool), boundary_thresh)
    gt_boundary = _mask_to_boundary(gt_mask.astype(bool), boundary_thresh)

    if pred_boundary.sum() == 0:
        precision = 0.0
    else:
        precision = float((pred_boundary & gt_boundary).sum()) / pred_boundary.sum()

    if gt_boundary.sum() == 0:
        recall = 1.0
    else:
        recall = float((pred_boundary & gt_boundary).sum()) / gt_boundary.sum()

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_jf_mean(
    pred_masks_dict: Dict[int, np.ndarray],
    gt_masks_dict: Dict[int, np.ndarray],
) -> Dict[str, float]:
    """Mean J&F over all instances. Returns {'J': float, 'F': float, 'JF': float}."""
    common_ids = set(pred_masks_dict.keys()) & set(gt_masks_dict.keys())
    if not common_ids:
        return {"J": 0.0, "F": 0.0, "JF": 0.0}

    j_scores = []
    f_scores = []
    for iid in common_ids:
        j_scores.append(compute_j_score(pred_masks_dict[iid], gt_masks_dict[iid]))
        f_scores.append(compute_f_score(pred_masks_dict[iid], gt_masks_dict[iid]))

    mean_j = float(np.mean(j_scores))
    mean_f = float(np.mean(f_scores))
    return {"J": mean_j, "F": mean_f, "JF": (mean_j + mean_f) / 2.0}


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area) / float(union_area)


def compute_unknown_recall(
    pred_unknown_boxes: np.ndarray,
    gt_unknown_boxes: np.ndarray,
    iou_thresh: float = 0.5,
) -> float:
    """What fraction of true unknown objects did we detect?"""
    if len(gt_unknown_boxes) == 0:
        return 1.0
    if len(pred_unknown_boxes) == 0:
        return 0.0

    matched = np.zeros(len(gt_unknown_boxes), dtype=bool)
    for pred_box in pred_unknown_boxes:
        for gi, gt_box in enumerate(gt_unknown_boxes):
            if matched[gi]:
                continue
            if _box_iou(pred_box, gt_box) >= iou_thresh:
                matched[gi] = True
                break

    return float(matched.sum()) / len(gt_unknown_boxes)


def compute_id_consistency(instance_ids_per_frame: List[List[int]]) -> float:
    """Temporal ID consistency: fraction of consecutive frame-pair IDs that match."""
    if len(instance_ids_per_frame) < 2:
        return 1.0

    consistent_pairs = 0
    total_pairs = 0

    for f_idx in range(len(instance_ids_per_frame) - 1):
        ids_curr = set(instance_ids_per_frame[f_idx])
        ids_next = set(instance_ids_per_frame[f_idx + 1])
        if not ids_curr and not ids_next:
            continue
        consistent = len(ids_curr & ids_next)
        total = len(ids_curr | ids_next)
        if total > 0:
            consistent_pairs += consistent
            total_pairs += total

    if total_pairs == 0:
        return 1.0
    return float(consistent_pairs) / float(total_pairs)
