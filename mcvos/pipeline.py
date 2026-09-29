"""
OW-VIS Pipeline: Open-World Video Instance Segmentation.

Combines:
- OpenWorldDetector (OW-DETR/DETR) for TRUE open-world detection (no text input)
- SAM 2 for temporal mask propagation
- UnknownInstanceMemoryBank for consistent cross-frame instance IDs

The KEY improvement over OWL-ViT:
- OWL-ViT: text queries in -> only matching objects out (NOT open-world)
- OpenWorldDetector: image in -> ALL objects out (known COCO + unknown novel)

Usage:
    pipeline = OWVISPipeline(
        sam2_checkpoint_path="/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt",
        sam2_config="sam2.1_hiera_large",
        unknown_score_threshold=0.3,
        memory_bank=memory_bank,
        crop_embedder=crop_embedder,
        device=torch.device("cuda"),
        use_owdetr=True,  # use OpenWorldDetector instead of OWL-ViT
    )
    results = pipeline.process_video("video.mp4", known_categories=["person","car"])
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image

# SAM2 path
SAM2_PATH = "/path/to/ow_vis/sam2"
if SAM2_PATH not in sys.path:
    sys.path.insert(0, SAM2_PATH)


class OWVISPipeline:
    """
    Full OW-VIS inference pipeline.

    Per-frame detection loop (with use_owdetr=True):
    1. Run OpenWorldDetector (DETR-based) on each frame - NO text input needed
    2. DETR class conf > unknown_score_threshold -> known objects (COCO labels)
    3. DETR objectness > objectness_threshold but class conf low -> unknown candidates
    4. Crop unknown candidates, embed via CropEmbedder, match/register in MemoryBank
    5. SAM2 propagates all active instance masks through the video

    Args:
        sam2_checkpoint_path: path to SAM2 .pt checkpoint
        sam2_config: config name, e.g. "sam2.1_hiera_large" or "sam2.1_hiera_base_plus"
        unknown_score_threshold: classification confidence threshold (known vs unknown)
        unknown_objectness_threshold: minimum objectness score to consider any detection
        memory_bank: UnknownInstanceMemoryBank instance
        crop_embedder: CropEmbedder instance
        device: torch.device
        use_owdetr: if True, use OpenWorldDetector (DETR-based, no text required)
                    if False, fall back to OWL-ViT (requires text queries)
        owdetr_checkpoint: optional path to OW-DETR .pth weights file
    """

    def __init__(
        self,
        sam2_checkpoint_path: str,
        sam2_config: str,
        memory_bank,
        crop_embedder,
        unknown_score_threshold: float = 0.3,
        unknown_objectness_threshold: float = 0.1,
        device: Optional[torch.device] = None,
        use_owdetr: bool = True,
        owdetr_checkpoint: Optional[str] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.unknown_score_threshold = unknown_score_threshold
        self.unknown_objectness_threshold = unknown_objectness_threshold
        self.memory_bank = memory_bank.to(device)
        self.crop_embedder = crop_embedder.to(device)

        # Load detector: OpenWorldDetector (OW-DETR/DETR) or OWL-ViT fallback
        self.use_owdetr = use_owdetr
        if use_owdetr:
            SRC_PATH = "/path/to/ow_vis/src"
            if SRC_PATH not in sys.path:
                sys.path.insert(0, SRC_PATH)
            from owdetr_detector import OpenWorldDetector
            self.detector = OpenWorldDetector(
                device=device,
                unknown_threshold=unknown_score_threshold,
                objectness_threshold=unknown_objectness_threshold,
                owdetr_checkpoint=owdetr_checkpoint,
            )
            self._detector_feat_dim = 256  # DETR hidden dim (for CropEmbedder)
        else:
            # Legacy OWL-ViT fallback (requires text queries)
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
            self.owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            self.owl_model = OwlViTForObjectDetection.from_pretrained(
                "google/owlvit-base-patch32"
            ).to(device).eval()

        # Load SAM2 video predictor
        # Config must be a path understood by Hydra's pkg://sam2 search path.
        # Actual YAML files on disk: sam2.1_hiera_b+.yaml, sam2.1_hiera_l.yaml,
        # sam2.1_hiera_s.yaml, sam2.1_hiera_t.yaml (inside configs/sam2.1/).
        # Accept friendly aliases and map them to real file names.
        from sam2.build_sam import build_sam2_video_predictor
        _ALIASES = {
            "sam2.1_hiera_large":      "configs/sam2.1/sam2.1_hiera_l",
            "sam2.1_hiera_base_plus":  "configs/sam2.1/sam2.1_hiera_b+",
            "sam2.1_hiera_small":      "configs/sam2.1/sam2.1_hiera_s",
            "sam2.1_hiera_tiny":       "configs/sam2.1/sam2.1_hiera_t",
            "sam2_hiera_large":        "configs/sam2/sam2_hiera_l",
            "sam2_hiera_base_plus":    "configs/sam2/sam2_hiera_b+",
        }
        _cfg = _ALIASES.get(sam2_config, sam2_config)
        # If still a bare name (not starting with configs/), prepend path
        if not _cfg.startswith("configs/"):
            if "sam2.1" in _cfg:
                _cfg = f"configs/sam2.1/{_cfg}"
            else:
                _cfg = f"configs/sam2/{_cfg}"
        if not _cfg.endswith(".yaml"):
            _cfg = _cfg + ".yaml"
        self.sam2_predictor = build_sam2_video_predictor(
            config_file=_cfg,
            ckpt_path=sam2_checkpoint_path,
            device=device,
        )

        # CropEmbedder's backbone_dim must match OWL-ViT patch embedding dim (768 for base-patch32)
        # (Only used in OWL-ViT mode; OW-DETR mode uses DETR hidden dim 256 -> pooled to crop_embedder)
        self._owl_feat_dim = 768

    # ------------------------------------------------------------------
    # Per-frame detection and memory-bank interaction
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_np: np.ndarray,
        frame_idx: int,
        known_text_categories: List[str],
    ) -> Dict:
        """
        Detect known and unknown objects in a single frame.

        Args:
            frame_np: HxWx3 uint8 RGB numpy array
            frame_idx: temporal index (used by memory bank)
            known_text_categories: list of text labels for known classes

        Returns:
            dict with keys:
              'known_boxes'   : (N_k, 4) numpy array [x1,y1,x2,y2] in pixel coords
              'known_labels'  : list of str, one per known box
              'known_scores'  : (N_k,) numpy array
              'unknown_boxes' : (N_u, 4) numpy array [x1,y1,x2,y2] in pixel coords
              'unknown_ids'   : list of int instance IDs from memory bank
              'unknown_scores': (N_u,) numpy array objectness scores
        """
        self.memory_bank.step_frame()
        H, W = frame_np.shape[:2]

        if self.use_owdetr:
            # --- OpenWorldDetector path (no text input required) ---
            det = self.detector.detect(frame_np)
            known_boxes = det["known_boxes"]
            known_scores = det["known_scores"]
            known_label_names = det["known_labels"]
            unknown_boxes = det["unknown_boxes"]
            unknown_scores = det["unknown_scores"]

            # Embed unknown boxes using pixel crop + CropEmbedder
            unknown_ids = []
            if len(unknown_boxes) > 0:
                for box in unknown_boxes:
                    crop_emb = self._embed_box_crop_pixel(frame_np, box)
                    inst_id, _ = self.memory_bank.match_or_register(crop_emb)
                    unknown_ids.append(inst_id)
        else:
            # --- Legacy OWL-ViT path (requires text queries) ---
            pil_image = Image.fromarray(frame_np)
            text_queries = [known_text_categories]  # OWL-ViT expects list of list

            inputs = self.owl_processor(
                text=text_queries,
                images=pil_image,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.owl_model(**inputs)

            target_sizes = torch.tensor([[H, W]], device=self.device)
            results = self.owl_processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.unknown_objectness_threshold,
            )[0]

            boxes_all = results["boxes"].cpu().numpy()
            scores_all = results["scores"].cpu().numpy()
            labels_all = results["labels"].cpu().numpy()

            known_mask = scores_all >= self.unknown_score_threshold
            unknown_mask = ~known_mask

            known_boxes = boxes_all[known_mask]
            known_scores = scores_all[known_mask]
            known_label_indices = labels_all[known_mask]
            known_label_names = [known_text_categories[i] for i in known_label_indices]

            unknown_boxes = boxes_all[unknown_mask]
            unknown_scores = scores_all[unknown_mask]

            unknown_ids = []
            if len(unknown_boxes) > 0:
                with torch.no_grad():
                    vision_outputs = self.owl_model.owlvit.vision_model(
                        pixel_values=inputs["pixel_values"],
                        output_hidden_states=True,
                    )
                    img_features = vision_outputs.last_hidden_state[0]
                    img_feat_h = img_feat_w = int((img_features.shape[0] - 1) ** 0.5)
                    patch_features = img_features[1:].reshape(img_feat_h, img_feat_w, -1)

                for box in unknown_boxes:
                    crop_emb = self._embed_box_crop(frame_np, box, patch_features, img_feat_h, img_feat_w)
                    inst_id, _ = self.memory_bank.match_or_register(crop_emb)
                    unknown_ids.append(inst_id)

        return {
            "known_boxes": known_boxes,
            "known_labels": known_label_names,
            "known_scores": known_scores,
            "unknown_boxes": unknown_boxes,
            "unknown_ids": unknown_ids,
            "unknown_scores": unknown_scores,
        }

    def _embed_box_crop(
        self,
        frame_np: np.ndarray,
        box: np.ndarray,
        patch_features: torch.Tensor,
        feat_h: int,
        feat_w: int,
    ) -> torch.Tensor:
        """
        Extract and project a feature embedding for a detected bounding box.

        Uses bilinear interpolation of OWL-ViT patch features within the box ROI,
        then passes through the CropEmbedder projector.

        Args:
            frame_np: HxWx3 numpy image
            box: [x1,y1,x2,y2] in pixel coords
            patch_features: (feat_h, feat_w, D) patch-level features
            feat_h, feat_w: spatial dims of patch_features

        Returns:
            embedding: (D_embed,) normalized embedding tensor
        """
        H, W = frame_np.shape[:2]
        x1, y1, x2, y2 = box
        # Normalize box to [0,1]
        x1n, x2n = max(0, x1 / W), min(1, x2 / W)
        y1n, y2n = max(0, y1 / H), min(1, y2 / H)

        # Map to patch feature grid
        px1 = int(x1n * feat_w)
        px2 = max(px1 + 1, int(x2n * feat_w))
        py1 = int(y1n * feat_h)
        py2 = max(py1 + 1, int(y2n * feat_h))
        px2 = min(px2, feat_w)
        py2 = min(py2, feat_h)

        roi_features = patch_features[py1:py2, px1:px2, :]  # (rh, rw, D)
        pooled = roi_features.mean(dim=(0, 1))  # (D,)
        pooled = pooled.unsqueeze(0)  # (1, D)

        with torch.no_grad():
            embedding = self.crop_embedder(pooled)  # (1, D_embed)
        return embedding.squeeze(0)  # (D_embed,)

    def _embed_box_crop_pixel(
        self,
        frame_np: np.ndarray,
        box: np.ndarray,
        output_dim: int = 256,
    ) -> torch.Tensor:
        """
        Embed a bounding box crop using mean pixel features + CropEmbedder.
        Used in OW-DETR mode where we don't have OWL-ViT patch features.

        Extracts a pixel crop, converts to a simple mean color/spatial descriptor,
        and projects through CropEmbedder for memory-bank matching.

        Args:
            frame_np: HxWx3 RGB numpy array
            box: [x1,y1,x2,y2] pixel coords
            output_dim: embedding dimension (must match CropEmbedder output)

        Returns:
            embedding: (output_dim,) normalized tensor on self.device
        """
        H, W = frame_np.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(W-1, x1))
        x2 = max(x1+1, min(W, x2))
        y1 = max(0, min(H-1, y1))
        y2 = max(y1+1, min(H, y2))

        crop = frame_np[y1:y2, x1:x2]  # (h, w, 3) uint8

        # Resize crop to fixed size for consistent features
        try:
            from PIL import Image as PILImage
            pil_crop = PILImage.fromarray(crop).resize((16, 16), PILImage.BILINEAR)
            crop_resized = np.array(pil_crop, dtype=np.float32) / 255.0  # (16, 16, 3)
        except Exception:
            crop_resized = crop.astype(np.float32) / 255.0

        # Flatten to a feature vector; pad/truncate to self._owl_feat_dim (768)
        flat = crop_resized.flatten()  # (16*16*3,) = (768,)
        if len(flat) < self._owl_feat_dim:
            flat = np.pad(flat, (0, self._owl_feat_dim - len(flat)))
        else:
            flat = flat[:self._owl_feat_dim]

        feat_tensor = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 768)

        with torch.no_grad():
            embedding = self.crop_embedder(feat_tensor)  # (1, output_dim)
        return embedding.squeeze(0)

    # ------------------------------------------------------------------
    # SAM2 video propagation
    # ------------------------------------------------------------------

    def run_sam2_on_boxes(
        self,
        video_frames: List[np.ndarray],
        prompt_boxes_per_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Run SAM2 video propagation seeded with bounding box prompts.

        Args:
            video_frames: list of HxWx3 uint8 RGB numpy frames
            prompt_boxes_per_frame: dict mapping frame_idx ->
                list of (obj_id, box_xyxy_pixels) tuples

        Returns:
            masks_per_frame: dict mapping frame_idx ->
                dict mapping obj_id -> binary mask (H, W) numpy bool array
        """
        if not prompt_boxes_per_frame:
            return {}

        H, W = video_frames[0].shape[:2]

        # SAM2 init_state requires a directory of JPEG frames
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, frame in enumerate(video_frames):
                fpath = os.path.join(tmpdir, f"{i:05d}.jpg")
                Image.fromarray(frame).save(fpath, quality=95)

            # SAM2 only supports prompts from a single frame without triggering
            # bfloat16/float32 dtype conflicts in memory_attention on PyTorch 2.7+.
            # Use only the first available prompt frame.
            first_prompt_frame = min(prompt_boxes_per_frame.keys())
            single_frame_prompts = {first_prompt_frame: prompt_boxes_per_frame[first_prompt_frame]}

            with torch.inference_mode():
                inference_state = self.sam2_predictor.init_state(
                    video_path=tmpdir,
                    offload_video_to_cpu=True,
                )

                # Add box prompts from the first prompt frame only
                for frame_idx, obj_box_list in single_frame_prompts.items():
                    for obj_id, box in obj_box_list:
                        # box: [x1,y1,x2,y2] numpy array
                        box_tensor = torch.tensor(box, dtype=torch.float32)
                        self.sam2_predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            obj_id=obj_id,
                            box=box_tensor,
                        )

                # Propagate forward; autocast bfloat16 matches SAM2's intended precision
                masks_per_frame: Dict[int, Dict[int, np.ndarray]] = {}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    for frame_idx, obj_ids, video_res_masks in self.sam2_predictor.propagate_in_video(
                        inference_state
                    ):
                        # video_res_masks: (num_objs, 1, H, W) float logits on GPU
                        binary_masks = (video_res_masks.float() > 0).squeeze(1).cpu().numpy()  # (num_objs, H, W)
                        frame_masks = {}
                        for i, oid in enumerate(obj_ids):
                            frame_masks[oid] = binary_masks[i]
                        masks_per_frame[frame_idx] = frame_masks

                self.sam2_predictor.reset_state(inference_state)

        return masks_per_frame

    # ------------------------------------------------------------------
    # Full video processing
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path_or_frames: Union[str, List[np.ndarray]],
        known_categories: List[str],
        run_sam2: bool = True,
        sam2_prompt_every_n_frames: int = 10,
    ) -> Dict:
        """
        Run the full OW-VIS pipeline on a video.

        Args:
            video_path_or_frames: path to .mp4 file, or list of HxWx3 numpy frames
            known_categories: list of text labels for known classes
            run_sam2: whether to run SAM2 mask propagation after detection
            sam2_prompt_every_n_frames: re-prompt SAM2 from detections every N frames

        Returns:
            dict with:
              'per_frame_detections': list of per-frame detection dicts
              'masks_per_frame': dict frame_idx -> {obj_id: binary_mask} (if run_sam2)
              'unknown_instances': dict instance_id -> metadata
        """
        # Load frames
        if isinstance(video_path_or_frames, str):
            frames = self._load_video_frames(video_path_or_frames)
        else:
            frames = video_path_or_frames

        # Per-frame detection
        per_frame_detections = []
        prompt_boxes_per_frame: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        all_instance_ids = set()

        for frame_idx, frame in enumerate(frames):
            det = self.process_frame(frame, frame_idx, known_categories)
            per_frame_detections.append(det)

            # Collect SAM2 prompts from unknown detections at interval frames
            if run_sam2 and frame_idx % sam2_prompt_every_n_frames == 0:
                if len(det["unknown_boxes"]) > 0:
                    prompt_boxes_per_frame[frame_idx] = [
                        (inst_id, box)
                        for inst_id, box in zip(det["unknown_ids"], det["unknown_boxes"])
                    ]
                    all_instance_ids.update(det["unknown_ids"])

                # Also add known detections with negative IDs to avoid collision
                if len(det["known_boxes"]) > 0:
                    known_sam2_pairs = []
                    for k_idx, (box, label) in enumerate(
                        zip(det["known_boxes"], det["known_labels"])
                    ):
                        obj_id = -(k_idx + 1) * (frame_idx + 1)  # unique negative ID
                        known_sam2_pairs.append((obj_id, box))
                    if frame_idx in prompt_boxes_per_frame:
                        prompt_boxes_per_frame[frame_idx].extend(known_sam2_pairs)
                    else:
                        prompt_boxes_per_frame[frame_idx] = known_sam2_pairs

        # SAM2 propagation
        masks_per_frame = {}
        if run_sam2 and prompt_boxes_per_frame:
            masks_per_frame = self.run_sam2_on_boxes(frames, prompt_boxes_per_frame)

        # Collect instance metadata from memory bank
        unknown_instances = {}
        active_mask = self.memory_bank.is_active
        for slot_idx in active_mask.nonzero(as_tuple=True)[0].tolist():
            iid = self.memory_bank.instance_ids[slot_idx].item()
            obs = self.memory_bank.observation_counts[slot_idx].item()
            last = self.memory_bank.last_seen_frame[slot_idx].item()
            promoted = iid in [p for p in self.memory_bank.get_promotion_candidates()]
            unknown_instances[iid] = {
                "observation_count": obs,
                "last_seen_frame": last,
                "promoted": promoted,
            }

        return {
            "per_frame_detections": per_frame_detections,
            "masks_per_frame": masks_per_frame,
            "unknown_instances": unknown_instances,
        }

    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Load video frames from .mp4 file using cv2 or PIL fallback."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            return frames
        except ImportError:
            # Fallback: use imageio
            import imageio
            reader = imageio.get_reader(video_path)
            return [np.array(f) for f in reader]
