"""
Force Tokenizer

VQ-VAE based tokenizer that converts continuous force fields into
discrete token sequences for efficient processing and conditioning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForceTokenizer(nn.Module):
    """
    Vector-Quantized VAE for force field tokenization.

    Encodes continuous force fields into discrete codebook indices,
    enabling efficient storage, transmission, and conditioning.
    """

    def __init__(
        self,
        force_channels: int = 6,
        hidden_dim: int = 256,
        codebook_size: int = 512,
        codebook_dim: int = 64,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_weight = commitment_weight

        # Encoder: force field -> continuous latent
        self.encoder = nn.Sequential(
            nn.Conv2d(force_channels, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, codebook_dim, 3, stride=2, padding=1),
        )

        # VQ codebook
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        # Decoder: discrete tokens -> reconstructed force field
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(codebook_dim, hidden_dim, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, force_channels, 3, padding=1),
        )

    def _quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vector quantization with straight-through estimator.

        Args:
            z: [B, codebook_dim, h, w] encoder output
        Returns:
            z_q: quantized tensor (same shape as z)
            indices: [B, h, w] codebook indices
            commitment_loss: scalar commitment loss
        """
        B, C, h, w = z.shape

        # Reshape for distance computation: [B*h*w, C]
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)

        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z@e^T
        dists = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.codebook.weight.pow(2).sum(dim=1, keepdim=False)
            - 2.0 * z_flat @ self.codebook.weight.T
        )

        # Nearest codebook entry
        indices = dists.argmin(dim=1)  # [B*h*w]
        z_q = self.codebook(indices).view(B, h, w, C).permute(0, 3, 1, 2)

        # Commitment loss
        commitment_loss = F.mse_loss(z_q.detach(), z) + self.commitment_weight * F.mse_loss(z_q, z.detach())

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        indices = indices.view(B, h, w)
        return z_q, indices, commitment_loss

    def forward(
        self, force_field: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode force field to discrete tokens.

        Args:
            force_field: [B, force_channels, H, W] continuous force field
        Returns:
            tokens: [B, h, w] discrete codebook indices
            commitment_loss: scalar VQ commitment loss for training
        """
        z = self.encoder(force_field)
        z_q, tokens, commitment_loss = self._quantize(z)
        return tokens, commitment_loss

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete tokens back to a force field.

        Args:
            tokens: [B, h, w] codebook indices
        Returns:
            force_field: [B, force_channels, H', W'] reconstructed force field
        """
        B, h, w = tokens.shape
        z_q = self.codebook(tokens)  # [B, h, w, codebook_dim]
        z_q = z_q.permute(0, 3, 1, 2)  # [B, codebook_dim, h, w]
        return self.decoder(z_q)

    def encode_and_decode(self, force_field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full VQ-VAE forward pass with reconstruction.

        Args:
            force_field: [B, force_channels, H, W]
        Returns:
            reconstructed: [B, force_channels, H', W']
            commitment_loss: scalar
        """
        z = self.encoder(force_field)
        z_q, _, commitment_loss = self._quantize(z)
        reconstructed = self.decoder(z_q)
        return reconstructed, commitment_loss
