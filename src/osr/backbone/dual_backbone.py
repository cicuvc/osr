import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet_8_2
from .dpt_fusion import DPTFusion
from .vit_wrapper import DINOv3Encoder, SARMAEEncoder


class DualBackbone(nn.Module):
    """
    Dual-encoder backbone for optical-SAR registration.

    Input [2N, 1, H, W]  (concatenated image0 + image1 along batch)
      ├─ Resize→VIT_IMG_SIZE ──── ViT (layers 2,3,4) ─ DPT → [D_fuse, G, G]
      └─ Resize→VIT_IMG_SIZE//2 ─ ResNet_8_2 (shared BN)
                                        │
                          feat_c [C3, G//2, G//2]  (G = VIT_IMG_SIZE / patch_size)
                          feat_m [C2, G,    G]
                          feat_f [C1, G*2,  G*2]
                                        │
                          concat(ViT_feat) → 1×1 fuse at each level
    """

    def __init__(self, config):
        super().__init__()
        block_dims = config['resnet']['block_dims']  # [128, 196, 256]
        self.block_dims = block_dims
        fuse_dim = config.get('vit_fuse_dim', 64)

        self.resnet = ResNet_8_2(config['resnet'])

        dinov3_path = config.get('dinov3_path')
        sarmae_path = config.get('sarmae_path')
        self.vit_img_size = config.get('vit_img_size', 384)
        self.resnet_img_size = config.get('resnet_img_size', self.vit_img_size // 2)
        dinov3_layers = config.get('dinov3_layers', 5)
        sarmae_blocks = config.get('sarmae_blocks', 4)

        self.dinov3 = DINOv3Encoder(dinov3_path, img_size=self.vit_img_size,
                                    num_layers=dinov3_layers)

        unfreeze_sar = config.get('unfreeze_sarmae', False)
        cross_attn_indices = config.get('sarmae_cross_attn_indices', None)
        self.use_cross_attn = unfreeze_sar and cross_attn_indices is not None
        self.cross_attn_indices = cross_attn_indices or []

        self.sarmae = SARMAEEncoder(sarmae_path, img_size=self.vit_img_size,
                                    num_blocks=sarmae_blocks,
                                    unfreeze=unfreeze_sar,
                                    cross_attn_indices=self.cross_attn_indices,
                                    cross_attn_kv_dim=384)

        self.dino_cross_attn_layers = config.get('dino_cross_attn_layers', None) or []

        # DPT fusion (last 3 layers each)
        self.dinov3_dpt = DPTFusion(in_dim=384, fuse_dim=fuse_dim, num_layers=3)
        self.sarmae_dpt = DPTFusion(in_dim=768, fuse_dim=fuse_dim, num_layers=3)

        # Level-wise fusion: [ResNet_C + opt_64 + sar_64] → ResNet_C
        self.fuse_c = nn.Conv2d(block_dims[2] + fuse_dim * 2, block_dims[2], 1)
        self.fuse_m = nn.Conv2d(block_dims[1] + fuse_dim * 2, block_dims[1], 1)
        self.fuse_f = nn.Conv2d(block_dims[0] + fuse_dim * 2, block_dims[0], 1)

    def _extract_vit_features(self, x_opt: torch.Tensor, x_sar: torch.Tensor):
        # DINOv3: always frozen, extract DPT layers + cross-attn layers
        dino_return = [2, 3, 4]
        if self.use_cross_attn:
            dino_return = sorted(set(dino_return + self.dino_cross_attn_layers))

        with torch.no_grad():
            dino_outputs = self.dinov3(x_opt, return_layers=dino_return)

        # Map DINOv3 layer indices to output list positions
        dino_return_sorted = sorted(dino_return)
        if isinstance(dino_outputs, list):
            dino_map = {dino_return_sorted[i]: dino_outputs[i] for i in range(len(dino_return_sorted))}
        else:
            dino_map = {dino_return_sorted[0]: dino_outputs}
            if len(dino_return_sorted) > 1:
                raise ValueError('multi-layer return expected list, got tensor')

        # DPT layers for optical
        opt_layers_dpt = [dino_map[2], dino_map[3], dino_map[4]]
        opt_spatial = [self.dinov3.tokens_to_spatial(t) for t in opt_layers_dpt]
        opt_fused = self.dinov3_dpt(opt_spatial)

        # SARMAE with optional cross-attention
        cross_kv = None
        if self.use_cross_attn:
            cross_kv = {k: dino_map[v] for k, v in zip(self.cross_attn_indices, self.dino_cross_attn_layers)}

        sar_layers = self.sarmae(x_sar, return_layers=[1, 2, 3], cross_attn_kv=cross_kv)

        sar_spatial = [self.sarmae.tokens_to_spatial(t) for t in sar_layers]
        sar_fused = self.sarmae_dpt(sar_spatial)

        return opt_fused, sar_fused

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [2N, 1, H, W]  concatenated image0 + image1 along batch dim

        Returns:
            feats_c: [2N, C3, Hc, Wc]
            feats_m: [2N, C2, Hm, Wm]
            feats_f: [2N, C1, Hf, Wf]
        """
        img0, img1 = x.chunk(2, dim=0)

        # ViT: DINOv3 frozen (no_grad in _extract), SARMAE may be unfrozen
        img0_3ch = img0.repeat(1, 3, 1, 1)
        img1_3ch = img1.repeat(1, 3, 1, 1)
        img0_vit = F.interpolate(img0_3ch, size=self.vit_img_size, mode='bilinear', align_corners=False)
        img1_vit = F.interpolate(img1_3ch, size=self.vit_img_size, mode='bilinear', align_corners=False)
        opt_enc, sar_enc = self._extract_vit_features(img0_vit, img1_vit)

        # ResNet at half the ViT resolution
        x_down = F.interpolate(x, size=self.resnet_img_size, mode='bilinear', align_corners=False)
        feats_c, feats_m, feats_f = self.resnet(x_down)

        feat_c0, feat_c1 = feats_c.chunk(2, dim=0)
        feat_m0, feat_m1 = feats_m.chunk(2, dim=0)
        feat_f0, feat_f1 = feats_f.chunk(2, dim=0)

        # Upsample ViT features to match medium and fine resolutions
        opt_m = F.interpolate(opt_enc, size=feat_m0.shape[-2:], mode='bilinear', align_corners=False)
        opt_f = F.interpolate(opt_enc, size=feat_f0.shape[-2:], mode='bilinear', align_corners=False)
        sar_m = F.interpolate(sar_enc, size=feat_m0.shape[-2:], mode='bilinear', align_corners=False)
        sar_f = F.interpolate(sar_enc, size=feat_f0.shape[-2:], mode='bilinear', align_corners=False)

        # Fuse at each level
        feat_c0 = self.fuse_c(torch.cat([feat_c0, opt_enc, sar_enc], dim=1))
        feat_c1 = self.fuse_c(torch.cat([feat_c1, opt_enc, sar_enc], dim=1))
        feat_m0 = self.fuse_m(torch.cat([feat_m0, opt_m, sar_m], dim=1))
        feat_m1 = self.fuse_m(torch.cat([feat_m1, opt_m, sar_m], dim=1))
        feat_f0 = self.fuse_f(torch.cat([feat_f0, opt_f, sar_f], dim=1))
        feat_f1 = self.fuse_f(torch.cat([feat_f1, opt_f, sar_f], dim=1))

        return (
            torch.cat([feat_c0, feat_c1], dim=0),
            torch.cat([feat_m0, feat_m1], dim=0),
            torch.cat([feat_f0, feat_f1], dim=0),
        )
