import torch
import torch.nn as nn
import torch.nn.functional as F


class DPTFusion(nn.Module):
    """
    Fuse features from multiple ViT layers into a single spatial feature map.

    Each layer output [N, C_in, H, W] is projected to dim_fuse via 1x1 conv,
    concatenated, then fused with 1x1 + residual 3x3 conv to produce a
    [N, dim_fuse, H, W] feature map.
    """

    def __init__(self, in_dim: int, fuse_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(in_dim, fuse_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fuse_dim * num_layers, fuse_dim, 1, bias=False),
            nn.BatchNorm2d(fuse_dim),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(fuse_dim, fuse_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(fuse_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        assert len(features) == self.num_layers
        projected = []
        for feat, proj in zip(features, self.proj_layers):
            projected.append(proj(feat))
        fused = self.fuse_conv(torch.cat(projected, dim=1))
        fused = fused + self.refine(fused)
        return fused
