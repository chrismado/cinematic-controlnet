"""Multi-Shot Consistency Module

Cross-attention mechanism that enforces temporal and stylistic
consistency across multiple shots in a cinematic sequence.

Ensures that lighting, color grading, character appearance, and
scene geometry remain coherent when cutting between camera angles.
"""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """Multi-head cross-attention between shot features."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, N_q, D] features of the current shot
            context: [B, N_kv, D] features of reference shots
        Returns:
            out: [B, N_q, D] cross-attended features
        """
        query = self.norm_q(query)
        context = self.norm_kv(context)

        B, N_q, D = query.shape
        N_kv = context.shape[1]

        q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, D)
        return self.out_proj(out)


class MultiShotConsistency(nn.Module):
    """
    Enforces multi-shot consistency via cross-attention.

    Each shot attends to all other shots in the sequence,
    producing features that are aware of the global cinematic context.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        metadata_dim: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Metadata encoder: shot index, camera angle, time code, etc.
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # Cross-attention layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "cross_attn": CrossAttentionBlock(feature_dim, num_heads, dropout),
                        "ffn": nn.Sequential(
                            nn.LayerNorm(feature_dim),
                            nn.Linear(feature_dim, feature_dim * 4),
                            nn.GELU(),
                            nn.Linear(feature_dim * 4, feature_dim),
                            nn.Dropout(dropout),
                        ),
                    }
                )
            )

        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        shot_features: list[torch.Tensor],
        shot_metadata: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Process multiple shots for consistency.

        Args:
            shot_features: List of [B, N_tokens, feature_dim] per shot
            shot_metadata: List of [B, metadata_dim] per shot
        Returns:
            List of [B, N_tokens, feature_dim] consistency-enhanced features
        """
        num_shots = len(shot_features)

        # Encode metadata and add to shot features
        enriched = []
        for feat, meta in zip(shot_features, shot_metadata):
            meta_emb = self.metadata_encoder(meta).unsqueeze(1)  # [B, 1, D]
            enriched.append(feat + meta_emb)

        # Concatenate all shots as global context
        global_context = torch.cat(enriched, dim=1)  # [B, total_tokens, D]

        # Each shot cross-attends to the global context
        outputs = []
        for i in range(num_shots):
            x = enriched[i]
            for layer_module in self.layers:
                layer = cast(nn.ModuleDict, layer_module)
                cross_attn = cast(CrossAttentionBlock, layer["cross_attn"])
                ffn = cast(nn.Sequential, layer["ffn"])
                x = x + cross_attn(x, global_context)
                x = x + ffn(x)
            x = self.output_norm(x)
            outputs.append(x)

        return outputs

    def consistency_loss(
        self,
        shot_features: list[torch.Tensor],
        shot_metadata: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute consistency loss encouraging similar global statistics
        across shots.

        Returns:
            loss: Scalar consistency loss
        """
        outputs = self.forward(shot_features, shot_metadata)

        # Compute mean feature per shot
        means = [out.mean(dim=1) for out in outputs]  # List of [B, D]

        # Pairwise MSE between shot means
        loss = torch.tensor(0.0, device=outputs[0].device)
        count = 0
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                loss = loss + F.mse_loss(means[i], means[j])
                count += 1

        if count > 0:
            loss = loss / count

        return loss
