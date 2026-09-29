"""
OpenWorldDetector: Wraps DETR (facebook/detr-resnet-50) for open-world object detection.

KEY DESIGN:
- OWL-ViT (old): text queries in -> only text-matched objects out
- OpenWorldDetector (new): image in -> ALL objects out (known + unknown)

"Unknown" objects are detected by exploiting DETR's class probability distribution:
  - If max_class_prob < unknown_threshold -> detector sees *something* but can't
    classify it as a COCO class -> treat as "unknown"
  - Objectness score must still exceed objectness_threshold to be considered

This gives true open-world detection without ANY text input.
"""

import sys
import torch
import numpy as np
from typing import Optional
from PIL import Image

# COCO 80-class names (indices match DETR output label ordering)
COCO_CLASSES = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A",
    "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]


class OpenWorldDetector:
    """
    Wraps DETR (or OW-DETR if weights available) for open-world object detection.

    Returns BOTH:
    - known_detections: objects matching COCO categories (80 classes)
    - unknown_detections: objects detected but NOT matching any COCO class
                         (high objectness but low class confidence)

    This is the KEY DIFFERENCE from OWL-ViT:
    - OWL-ViT: text queries in -> matching objects out
    - OpenWorldDetector: image in -> all objects out (known + unknown)

    Args:
        device: 'cuda' or 'cpu' or torch.device
        unknown_threshold: max class probability below this -> object is "unknown"
        objectness_threshold: DETR end-prob score must exceed this to consider any box
        owdetr_checkpoint: optional path to OW-DETR .pth weights (if None, uses DETR fallback)
    """

    def __init__(
        self,
        device='cuda',
        unknown_threshold: float = 0.5,
        objectness_threshold: float = 0.1,
        owdetr_checkpoint: Optional[str] = None,
    ):
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.unknown_threshold = unknown_threshold
        self.objectness_threshold = objectness_threshold
        self._owdetr_loaded = False

        print(f'[OpenWorldDetector] Initializing on {self.device}')

        if owdetr_checkpoint is not None and self._try_load_owdetr(owdetr_checkpoint):
            print('[OpenWorldDetector] Loaded OW-DETR custom weights')
        else:
            self._load_detr_fallback()
            print('[OpenWorldDetector] Using DETR (facebook/detr-resnet-50) fallback')

    def _try_load_owdetr(self, checkpoint_path: str) -> bool:
        import os
        if not os.path.exists(checkpoint_path):
            print(f'[OpenWorldDetector] Checkpoint not found: {checkpoint_path}')
            return False
        size_mb = os.path.getsize(checkpoint_path) / 1e6
        if size_mb < 1.0:
            print(f'[OpenWorldDetector] Checkpoint too small ({size_mb:.2f}MB), skipping')
            return False
        try:
            from transformers import DetrImageProcessor, DetrForObjectDetection
            self.processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50', revision='no_timm')
            self.model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', revision='no_timm', ignore_mismatched_sizes=True)
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state_dict = ckpt.get('model', ckpt)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f'[OpenWorldDetector] OW-DETR load: {len(missing)} missing, {len(unexpected)} unexpected keys')
            self.model = self.model.to(self.device).eval()
            self._owdetr_loaded = True
            return True
        except Exception as e:
            print(f'[OpenWorldDetector] OW-DETR load failed: {e}')
            return False

    def _load_detr_fallback(self):
        from transformers import DetrImageProcessor, DetrForObjectDetection
        self.processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50', revision='no_timm')
        self.model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', revision='no_timm').to(self.device).eval()

    def detect(self, image_np: np.ndarray) -> dict:
        """
        Detect all objects in an image - both known COCO classes and unknowns.

        Args:
            image_np: HxWx3 RGB numpy array (uint8)

        Returns:
            dict with:
                'known_boxes':    (N_k, 4) float32 xyxy pixel coords
                'known_scores':   (N_k,)   float32 classification confidence
                'known_labels':   list of str  COCO category names
                'unknown_boxes':  (N_u, 4) float32 xyxy pixel coords
                'unknown_scores': (N_u,)   float32 objectness scores
        """
        H, W = image_np.shape[:2]
        pil_image = Image.fromarray(image_np)

        inputs = self.processor(images=pil_image, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # outputs.logits: (1, num_queries, num_classes+1)
        # outputs.pred_boxes: (1, num_queries, 4) cxcywh normalized [0,1]
        logits = outputs.logits[0]          # (num_queries, num_classes+1)
        pred_boxes = outputs.pred_boxes[0]  # (num_queries, 4)

        # Softmax over class+background dimension
        probs = logits.softmax(dim=-1)  # (num_queries, num_classes+1)
        # Last class = "no object" (background)
        # Objectness = 1 - P(background)
        objectness_scores = 1.0 - probs[:, -1]  # (num_queries,)

        # Max class probability excluding background
        class_probs = probs[:, :-1]  # (num_queries, num_classes)
        max_class_probs, class_indices = class_probs.max(dim=-1)

        # Convert cxcywh -> xyxy in pixel coords
        cx, cy, bw, bh = pred_boxes.unbind(-1)
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        x2 = (cx + bw / 2) * W
        y2 = (cy + bh / 2) * H
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)  # (num_queries, 4)

        # Filter by objectness threshold first
        obj_mask = objectness_scores > self.objectness_threshold

        # Known: above objectness threshold AND class confidence > unknown_threshold
        known_mask = obj_mask & (max_class_probs > self.unknown_threshold)

        # Unknown: above objectness threshold BUT class confidence <= unknown_threshold
        # These are "something is there" but DETR can't confidently classify it
        unknown_mask = obj_mask & (max_class_probs <= self.unknown_threshold)

        # Extract known detections
        known_boxes = boxes_xyxy[known_mask].cpu().numpy()
        known_scores = max_class_probs[known_mask].cpu().numpy()
        known_label_indices = class_indices[known_mask].cpu().numpy()
        known_labels = [
            COCO_CLASSES[int(i)] if int(i) < len(COCO_CLASSES) else f'class_{i}'
            for i in known_label_indices
        ]

        # Extract unknown detections
        unknown_boxes = boxes_xyxy[unknown_mask].cpu().numpy()
        unknown_scores = objectness_scores[unknown_mask].cpu().numpy()

        return {
            'known_boxes': known_boxes,
            'known_scores': known_scores,
            'known_labels': known_labels,
            'unknown_boxes': unknown_boxes,
            'unknown_scores': unknown_scores,
        }

    @property
    def backend_name(self) -> str:
        return 'OW-DETR' if self._owdetr_loaded else 'DETR-ResNet50'
