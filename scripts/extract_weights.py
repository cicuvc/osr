"""
Generate a pretrained DualBackbone checkpoint.

Combines:
  1. ResNet weights from an existing OSR checkpoint (optional)
  2. DINOv3 weights (safetensors) for optical ViT encoder layers 0-4
  3. SARMAE weights (.pth) for SAR ViT encoder blocks 0-3

The fusion layers (DPTFusion, fuse_c/m/f) are left randomly initialised.

Usage:
    python scripts/extract_weights.py \
        --dinov3 /path/to/dinov3s/model.safetensors \
        --sarmae /path/to/SARMAE_vit_Base.pth \
        --output dual_backbone_init.ckpt
"""

import argparse
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load


def extract_dinov3_weights(safetensors_path: str, num_layers: int = 5,
                           grid_size: int = 24) -> dict:
    """Extract DINOv3 patch_embed, CLS, register, and first N layers."""
    w = safetensors_load(safetensors_path)
    out = {}

    out['dinov3.patch_embed.weight'] = w['embeddings.patch_embeddings.weight']
    out['dinov3.patch_embed.bias']   = w['embeddings.patch_embeddings.bias']
    out['dinov3.cls_token']          = w['embeddings.cls_token']
    out['dinov3.register_tokens']    = w['embeddings.register_tokens']

    for i in range(num_layers):
        dst_p = f'dinov3.layers.{i}.'
        src_p = f'layer.{i}.'
        mapping = {
            'norm1.weight':     'norm1.weight',
            'norm1.bias':       'norm1.bias',
            'norm2.weight':     'norm2.weight',
            'norm2.bias':       'norm2.bias',
            'q_proj.weight':    'attention.q_proj.weight',
            'q_proj.bias':      'attention.q_proj.bias',
            'k_proj.weight':    'attention.k_proj.weight',
            'k_proj.bias':      'attention.k_proj.bias',
            'v_proj.weight':    'attention.v_proj.weight',
            'v_proj.bias':      'attention.v_proj.bias',
            'o_proj.weight':    'attention.o_proj.weight',
            'o_proj.bias':      'attention.o_proj.bias',
            'layer_scale1':     'layer_scale1.lambda1',
            'layer_scale2':     'layer_scale2.lambda1',
            'mlp.0.weight':     'mlp.up_proj.weight',
            'mlp.0.bias':       'mlp.up_proj.bias',
            'mlp.2.weight':     'mlp.down_proj.weight',
            'mlp.2.bias':       'mlp.down_proj.bias',
        }
        for dst_k, src_k in mapping.items():
            key = src_p + src_k
            if key in w:
                out[dst_p + dst_k] = w[key]

    print(f'DINOv3: {len(out)} keys')
    return out


def extract_sarmae_weights(pth_path: str, num_blocks: int = 4,
                           grid_size: int = 24) -> dict:
    """Extract SARMAE patch_embed, CLS, pos_embed, and first N blocks."""
    ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
    w = ckpt['model']
    out = {}

    out['sarmae.patch_embed.weight'] = w['patch_embed.proj.weight']
    out['sarmae.patch_embed.bias']   = w['patch_embed.proj.bias']
    out['sarmae.cls_token']          = w['cls_token']

    # Interpolate position embedding
    src_pe = w['pos_embed']  # [1, 197, 768]
    src_g = int((src_pe.shape[1] - 1) ** 0.5)
    if src_g != grid_size:
        cls_pe = src_pe[:, :1, :]
        patch_pe = src_pe[:, 1:, :].reshape(1, src_g, src_g, -1).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(patch_pe, size=(grid_size, grid_size),
                                 mode='bicubic', align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, grid_size**2, -1)
        out['sarmae.pos_embed'] = torch.cat([cls_pe, patch_pe], dim=1)
    else:
        out['sarmae.pos_embed'] = src_pe

    for i in range(num_blocks):
        dst_p = f'sarmae.blocks.{i}.'
        src_p = f'blocks.{i}.'
        mapping = {
            'norm1.weight':      'norm1.weight',
            'norm1.bias':        'norm1.bias',
            'norm2.weight':      'norm2.weight',
            'norm2.bias':        'norm2.bias',
            'attn_qkv.weight':   'attn.qkv.weight',
            'attn_qkv.bias':     'attn.qkv.bias',
            'attn_proj.weight':  'attn.proj.weight',
            'attn_proj.bias':    'attn.proj.bias',
            'mlp.0.weight':      'mlp.fc1.weight',
            'mlp.0.bias':        'mlp.fc1.bias',
            'mlp.2.weight':      'mlp.fc2.weight',
            'mlp.2.bias':        'mlp.fc2.bias',
        }
        for dst_k, src_k in mapping.items():
            key = src_p + src_k
            if key in w:
                out[dst_p + dst_k] = w[key]

    print(f'SARMAE: {len(out)} keys')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--osr_ckpt', type=str, default=None)
    parser.add_argument('--dinov3', type=str, required=True)
    parser.add_argument('--sarmae', type=str, required=True)
    parser.add_argument('--output', type=str, default='dual_backbone_init.ckpt')
    parser.add_argument('--dinov3_layers', type=int, default=5)
    parser.add_argument('--sarmae_blocks', type=int, default=4)
    parser.add_argument('--grid_size', type=int, default=24)
    args = parser.parse_args()

    state_dict = {}

    # ResNet weights from old OSR checkpoint
    if args.osr_ckpt:
        osr = torch.load(args.osr_ckpt, map_location='cpu', weights_only=False)
        osr_sd = osr.get('state_dict', osr)
        for k, v in osr_sd.items():
            if k.startswith('matcher.'):
                k = k[len('matcher.'):]
            if k.startswith('backbone.'):
                state_dict['backbone.resnet.' + k[len('backbone.'):]] = v
        print(f'ResNet: {len(state_dict)} keys')

    state_dict.update(extract_dinov3_weights(args.dinov3, args.dinov3_layers, args.grid_size))
    state_dict.update(extract_sarmae_weights(args.sarmae, args.sarmae_blocks, args.grid_size))

    out = {'state_dict': state_dict}
    torch.save(out, args.output)
    print(f'Saved {args.output}  ({len(state_dict)} keys total)')


if __name__ == '__main__':
    main()
