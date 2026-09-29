"""
Unknown Instance Memory Bank for OW-VIS.

Core novel contribution: stores embeddings of discovered unknown object instances
and matches them across video frames for consistent identity, even across occlusion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np


class UnknownInstanceMemoryBank(nn.Module):
    """
    Maintains a persistent store of unknown object instance embeddings.
    
    When OW-OVD discovers an unknown object in a frame, its crop is embedded
    and stored here. On subsequent frames, new unknown detections are matched
    against the bank via contrastive similarity to retrieve a consistent ID.
    
    Key design choices:
    - Embeddings updated via exponential moving average (EMA) for stability
    - Contrastive metric learning with triplet loss for discriminative embeddings
    - Instance pruning: entries removed if not seen for N frames
    - Promotion: after K observations, unknown promoted to pseudo-class
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        max_instances: int = 512,
        ema_momentum: float = 0.9,
        match_threshold: float = 0.7,
        prune_after_frames: int = 30,
        promote_after_obs: int = 10,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_instances = max_instances
        self.ema_momentum = ema_momentum
        self.match_threshold = match_threshold
        self.prune_after_frames = prune_after_frames
        self.promote_after_obs = promote_after_obs

        # Memory storage (not nn.Parameters — updated manually)
        self.register_buffer('embeddings', torch.zeros(max_instances, embedding_dim))
        self.register_buffer('instance_ids', torch.full((max_instances,), -1, dtype=torch.long))
        self.register_buffer('observation_counts', torch.zeros(max_instances, dtype=torch.long))
        self.register_buffer('last_seen_frame', torch.full((max_instances,), -1, dtype=torch.long))
        self.register_buffer('is_active', torch.zeros(max_instances, dtype=torch.bool))

        self._next_id = 0
        self._current_frame = 0

    def step_frame(self):
        """Call once per video frame to update internal frame counter."""
        self._current_frame += 1
        # Prune stale entries
        stale = self.is_active & (
            self._current_frame - self.last_seen_frame > self.prune_after_frames
        )
        self.is_active[stale] = False

    def match_or_register(
        self, query_embedding: torch.Tensor
    ) -> Tuple[int, bool]:
        """
        Given a query embedding from a newly detected unknown object, find the
        best matching existing instance or register a new one.

        Args:
            query_embedding: (D,) normalized embedding of detected unknown crop

        Returns:
            instance_id: consistent integer ID for this instance
            is_new: True if a new instance was registered
        """
        query_embedding = F.normalize(query_embedding, dim=-1)
        active_mask = self.is_active  # (max_instances,)

        if active_mask.any():
            active_embs = self.embeddings[active_mask]  # (N_active, D)
            sims = torch.mv(active_embs, query_embedding)  # (N_active,)
            best_sim, best_local_idx = sims.max(dim=0)

            if best_sim.item() >= self.match_threshold:
                # Match found — retrieve global index
                active_indices = active_mask.nonzero(as_tuple=True)[0]
                global_idx = active_indices[best_local_idx].item()
                # EMA update
                self.embeddings[global_idx] = (
                    self.ema_momentum * self.embeddings[global_idx]
                    + (1 - self.ema_momentum) * query_embedding
                )
                self.embeddings[global_idx] = F.normalize(self.embeddings[global_idx], dim=-1)
                self.observation_counts[global_idx] += 1
                self.last_seen_frame[global_idx] = self._current_frame
                return self.instance_ids[global_idx].item(), False

        # No match — register new instance
        slot = self._find_free_slot()
        if slot is None:
            # Bank full — evict least recently seen
            slot = self.last_seen_frame.argmin().item()

        self.embeddings[slot] = query_embedding
        self.instance_ids[slot] = self._next_id
        self.observation_counts[slot] = 1
        self.last_seen_frame[slot] = self._current_frame
        self.is_active[slot] = True

        new_id = self._next_id
        self._next_id += 1
        return new_id, True

    def get_promotion_candidates(self) -> List[int]:
        """Return instance IDs that have been observed enough to promote to pseudo-class."""
        eligible = self.is_active & (self.observation_counts >= self.promote_after_obs)
        return self.instance_ids[eligible].tolist()

    def _find_free_slot(self) -> Optional[int]:
        inactive = (~self.is_active).nonzero(as_tuple=True)[0]
        if len(inactive) == 0:
            return None
        return inactive[0].item()

    def __len__(self) -> int:
        return self.is_active.sum().item()


class CropEmbedder(nn.Module):
    """
    Embeds image crops (unknown object detections) into the memory bank space.
    
    Uses a lightweight projection head on top of frozen DINOv2 or CLIP features.
    The projector is trained with triplet loss during the video sequence.
    """

    def __init__(self, backbone_dim: int = 768, embedding_dim: int = 256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, crop_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            crop_features: (B, D_backbone) — pooled features from backbone
        Returns:
            embeddings: (B, D_embed) — L2-normalized embeddings
        """
        emb = self.projector(crop_features)
        return F.normalize(emb, dim=-1)


class TripletMemoryLoss(nn.Module):
    """
    Trains the CropEmbedder to be discriminative across unknown instances.
    
    Positive pairs: two crops of the same unknown instance (different frames)
    Negative pairs: crops from different unknown instances
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        d_pos = (anchor - positive).pow(2).sum(dim=-1)
        d_neg = (anchor - negative).pow(2).sum(dim=-1)
        loss = F.relu(d_pos - d_neg + self.margin).mean()
        return loss
