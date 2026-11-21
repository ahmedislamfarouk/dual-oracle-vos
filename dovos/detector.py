"""Class-agnostic proposals from a COCO-pretrained DETR-R50.

Every query whose objectness, 1 - p(no-object), exceeds `objectness_threshold` is kept, whatever its
class. Boxes are returned as [x1, y1, x2, y2] in pixels of the input image; confidently classified
queries (max class probability > 0.5) come first, then the rest, each group in query order.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image


class ObjectnessDetector:
    def __init__(self, device: str = "cuda", objectness_threshold: float = 0.05,
                 class_threshold: float = 0.5, model_name: str = "facebook/detr-resnet-50"):
        from transformers import DetrForObjectDetection, DetrImageProcessor
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.objectness_threshold = objectness_threshold
        self.class_threshold = class_threshold
        self.processor = DetrImageProcessor.from_pretrained(model_name, revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained(model_name, revision="no_timm").to(self.device).eval()

    @torch.no_grad()
    def __call__(self, image_rgb: np.ndarray) -> List[List[float]]:
        h, w = image_rgb.shape[:2]
        inputs = self.processor(images=Image.fromarray(image_rgb), return_tensors="pt")
        out = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
        probs = out.logits[0].softmax(-1)
        objectness = 1.0 - probs[:, -1]
        max_class, _ = probs[:, :-1].max(-1)
        cx, cy, bw, bh = out.pred_boxes[0].unbind(-1)
        boxes = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], -1)
        keep = objectness > self.objectness_threshold
        confident = keep & (max_class > self.class_threshold)
        uncertain = keep & (max_class <= self.class_threshold)
        return [list(b) for b in boxes[confident].cpu().numpy().tolist()] + \
               [list(b) for b in boxes[uncertain].cpu().numpy().tolist()]


class MaskRCNNDetector:
    """Second detector for the pre-registered replication: torchvision Mask R-CNN R50-FPN (COCO).

    Every box with score >= `score_threshold` is kept whatever its class, ordered by score.
    """

    def __init__(self, device: str = "cuda", score_threshold: float = 0.05):
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        self.model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                                           box_score_thresh=score_threshold).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, image_rgb: np.ndarray) -> List[List[float]]:
        x = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255).to(self.device)
        pred = self.model([x])[0]
        order = pred["scores"].argsort(descending=True)
        keep = order[pred["scores"][order] >= self.score_threshold]
        return [list(b) for b in pred["boxes"][keep].cpu().numpy().tolist()]


def build_detector(name: str, device: str, threshold: float):
    if name == "maskrcnn":
        return MaskRCNNDetector(device, threshold)
    return ObjectnessDetector(device, objectness_threshold=threshold)
