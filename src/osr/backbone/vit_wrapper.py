import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.modules.fused_norm_gate import FusedRMSNormSwishGate
from flash_attn import flash_attn_func
from safetensors.torch import load_file as safetensors_load


# ---------------------------------------------------------------------------
# RoPE (Rotation Position Embedding) utilities
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _compute_rope_cos_sin(dim: int, seq_len: int, theta: float = 100.0,
                          device=None, dtype=None):
    position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    args = position * freqs.unsqueeze(0)
    return args.cos().to(dtype), args.sin().to(dtype)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos[None, :, None, :]  # [1, seq_len, 1, head_dim//2]
    sin = sin[None, :, None, :]
    cos = torch.cat([cos, cos], dim=-1)  # [1, seq_len, 1, head_dim]
    sin = torch.cat([sin, sin], dim=-1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# DINOv3-S encoder  (layers 0-4 of 12, 384-dim, 16×16 patch, 4 register tokens)
# ---------------------------------------------------------------------------

class DINOv3Layer(nn.Module):
    def __init__(self, dim=384, mlp_ratio=4, num_heads=6, rope_theta=100.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope_theta = rope_theta

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)
        self.layer_scale1 = nn.Parameter(torch.ones(dim) * 1e-5)
        self.layer_scale2 = nn.Parameter(torch.ones(dim) * 1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio), bias=True),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim, bias=True),
        )

    def _compute_rope(self, seq_len, device, dtype):
        return _compute_rope_cos_sin(self.head_dim, seq_len, self.rope_theta, device, dtype)

    def forward(self, x, cos, sin):
        residual = x
        x_norm = self.norm1(x)
        B, N, C = x_norm.shape

        q = self.q_proj(x_norm).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x_norm).view(B, N, self.num_heads, self.head_dim)

        q, k = apply_rope(q, k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        x = residual + self.layer_scale1 * self.o_proj(attn_out)

        x = x + self.layer_scale2 * self.mlp(self.norm2(x))
        return x


class DINOv3Encoder(nn.Module):
    def __init__(self, ckpt_path: str, img_size: int = 384, num_layers: int = 5,
                 num_register: int = 4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = 16
        self.grid_size = img_size // self.patch_size
        self.dim = 384
        self.num_register = num_register

        self.patch_embed = nn.Conv2d(3, self.dim, kernel_size=16, stride=16, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register, self.dim))
        self.layers = nn.ModuleList([DINOv3Layer() for _ in range(num_layers)])

        self._init_weights()
        if ckpt_path:
            self._load_pretrained(ckpt_path)

        for param in self.parameters():
            param.requires_grad = False

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)

    def _load_pretrained(self, ckpt_path: str):
        weights = safetensors_load(ckpt_path)
        sd = self.state_dict()

        key_map = {
            'patch_embed.weight': 'embeddings.patch_embeddings.weight',
            'patch_embed.bias':   'embeddings.patch_embeddings.bias',
            'cls_token':          'embeddings.cls_token',
            'register_tokens':    'embeddings.register_tokens',
        }
        for dst_k, src_k in key_map.items():
            if dst_k in sd and src_k in weights:
                sd[dst_k] = weights[src_k]

        for i in range(len(self.layers)):
            dst_p = f'layers.{i}.'
            src_p = f'layer.{i}.'
            layer_map = {
                dst_p + 'norm1.weight':     src_p + 'norm1.weight',
                dst_p + 'norm1.bias':       src_p + 'norm1.bias',
                dst_p + 'norm2.weight':     src_p + 'norm2.weight',
                dst_p + 'norm2.bias':       src_p + 'norm2.bias',
                dst_p + 'q_proj.weight':    src_p + 'attention.q_proj.weight',
                dst_p + 'q_proj.bias':      src_p + 'attention.q_proj.bias',
                dst_p + 'k_proj.weight':    src_p + 'attention.k_proj.weight',
                dst_p + 'k_proj.bias':      src_p + 'attention.k_proj.bias',
                dst_p + 'v_proj.weight':    src_p + 'attention.v_proj.weight',
                dst_p + 'v_proj.bias':      src_p + 'attention.v_proj.bias',
                dst_p + 'o_proj.weight':    src_p + 'attention.o_proj.weight',
                dst_p + 'o_proj.bias':      src_p + 'attention.o_proj.bias',
                dst_p + 'layer_scale1':     src_p + 'layer_scale1.lambda1',
                dst_p + 'layer_scale2':     src_p + 'layer_scale2.lambda1',
                dst_p + 'mlp.0.weight':     src_p + 'mlp.up_proj.weight',
                dst_p + 'mlp.0.bias':       src_p + 'mlp.up_proj.bias',
                dst_p + 'mlp.2.weight':     src_p + 'mlp.down_proj.weight',
                dst_p + 'mlp.2.bias':       src_p + 'mlp.down_proj.bias',
            }
            for dst_k, src_k in layer_map.items():
                if dst_k in sd and src_k in weights:
                    sd[dst_k] = weights[src_k]

        self.load_state_dict(sd, strict=False)

    def forward(self, x: torch.Tensor, return_layers: list[int] | None = None):
        B, C, H, W = x.shape
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_t = self.cls_token.expand(B, -1, -1)
        reg_t = self.register_tokens.expand(B, -1, -1)
        x_seq = torch.cat([cls_t, reg_t, patches], dim=1)

        seq_len = x_seq.shape[1]
        cos, sin = _compute_rope_cos_sin(self.layers[0].head_dim, seq_len,
                                         self.layers[0].rope_theta,
                                         x_seq.device, x_seq.dtype)

        if return_layers is None:
            return_layers = [len(self.layers) - 1]

        layer_outputs = []
        for i, layer in enumerate(self.layers):
            x_seq = layer(x_seq, cos, sin)
            if i in return_layers:
                layer_outputs.append(x_seq)

        if len(layer_outputs) == 1:
            return layer_outputs[0]
        return layer_outputs

    def tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        N, _, C = tokens.shape
        patch_tokens = tokens[:, 1 + self.num_register:]
        return patch_tokens.transpose(1, 2).reshape(N, C, self.grid_size, self.grid_size)


# ---------------------------------------------------------------------------
# SARMAE ViT-Base encoder  (blocks 0-3 of 12, 768-dim, 16×16 patch)
# ---------------------------------------------------------------------------

class SARMAEBlock(nn.Module):
    def __init__(self, dim=768, mlp_ratio=4, num_heads=12):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_proj = nn.Linear(dim, dim, bias=True)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio), bias=True),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim, bias=True),
        )

    def forward(self, x):
        residual = x
        B, N, C = self.norm1(x).shape
        x_norm = self.norm1(x)
        qkv = self.attn_qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        x = residual + self.attn_proj(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossModalBlock(nn.Module):
    """Cross-attention from SAR modality to optical modality.

    Q: SARMAE features [N, Lq, dim_q]
    KV: DINOv3 features [N, Lkv, dim_kv] projected to dim_q
    Output: SARMAE features + cross_attn(SARMAE, projected_DINOv3)
    """

    def __init__(self, q_dim: int, kv_dim: int, num_heads: int = 12):
        super().__init__()
        self.norm = nn.LayerNorm(q_dim)
        self.k_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, q_dim, bias=False)
        
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.q_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.gate_down = nn.Linear(q_dim, q_dim // 8)
        self.gate_up = nn.Linear(q_dim // 8, q_dim)
        self.o_gate = FusedRMSNormSwishGate(q_dim)

        self.q_norm = nn.RMSNorm((self.head_dim, ))
        self.k_norm = nn.RMSNorm((self.head_dim, ))

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        residual = x
        B, Lq, Cq = x.shape

        q = self.q_norm(self.q_proj(self.norm(x)).view(B, Lq, self.num_heads, self.head_dim)).to(torch.bfloat16)
        k = self.k_norm(self.k_proj(kv).view(B, -1, self.num_heads, self.head_dim)).to(torch.bfloat16)
        v = self.v_proj(kv).view(B, -1, self.num_heads, self.head_dim)

        g = self.gate_up(self.gate_down(x))
        attn_out = flash_attn_func(q, k, v)
        attn_out = self.o_gate(attn_out.reshape(B, Lq, Cq), g)
        return residual + self.o_proj(attn_out)


class SARMAEEncoder(nn.Module):
    def __init__(self, ckpt_path: str, img_size: int = 384, num_blocks: int = 4,
                 unfreeze: bool = False, cross_attn_indices: list[int] | None = None,
                 cross_attn_kv_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = 16
        self.grid_size = img_size // self.patch_size
        self.dim = 768

        self.patch_embed = nn.Conv2d(3, self.dim, kernel_size=16, stride=16, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size**2 + 1, self.dim))
        self.blocks = nn.ModuleList([SARMAEBlock() for _ in range(num_blocks)])

        # Optional cross-attention after specific blocks (KV from another encoder)
        self.cross_attn_indices = cross_attn_indices or []
        self.cross_attn = nn.ModuleDict()
        for idx in self.cross_attn_indices:
            self.cross_attn[str(idx)] = CrossModalBlock(
                q_dim=self.dim, kv_dim=cross_attn_kv_dim)

        self._init_weights()
        if ckpt_path:
            self._load_pretrained(ckpt_path)

        for param in self.parameters():
            param.requires_grad = unfreeze

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _load_pretrained(self, ckpt_path: str):
        mae_state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        src_sd = mae_state['model']
        sd = self.state_dict()

        key_map = {
            'patch_embed.weight': 'patch_embed.proj.weight',
            'patch_embed.bias':   'patch_embed.proj.bias',
            'cls_token':          'cls_token',
        }
        for dst_k, src_k in key_map.items():
            if dst_k in sd and src_k in src_sd:
                sd[dst_k] = src_sd[src_k]

        # Interpolate position embedding
        src_pe = src_sd['pos_embed']
        src_grid = int((src_pe.shape[1] - 1) ** 0.5)
        dst_grid = self.grid_size
        if src_grid != dst_grid:
            cls_pe = src_pe[:, :1, :]
            patch_pe = src_pe[:, 1:, :]
            patch_pe = patch_pe.reshape(1, src_grid, src_grid, self.dim).permute(0, 3, 1, 2)
            patch_pe = F.interpolate(patch_pe, size=(dst_grid, dst_grid), mode='bicubic', align_corners=False)
            patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, dst_grid * dst_grid, self.dim)
            sd['pos_embed'] = torch.cat([cls_pe, patch_pe], dim=1)
        else:
            sd['pos_embed'] = src_sd['pos_embed']

        for i in range(len(self.blocks)):
            dst_p = f'blocks.{i}.'
            src_p = f'blocks.{i}.'
            block_map = {
                dst_p + 'norm1.weight':      src_p + 'norm1.weight',
                dst_p + 'norm1.bias':        src_p + 'norm1.bias',
                dst_p + 'norm2.weight':      src_p + 'norm2.weight',
                dst_p + 'norm2.bias':        src_p + 'norm2.bias',
                dst_p + 'attn_qkv.weight':   src_p + 'attn.qkv.weight',
                dst_p + 'attn_qkv.bias':     src_p + 'attn.qkv.bias',
                dst_p + 'attn_proj.weight':  src_p + 'attn.proj.weight',
                dst_p + 'attn_proj.bias':    src_p + 'attn.proj.bias',
                dst_p + 'mlp.0.weight':      src_p + 'mlp.fc1.weight',
                dst_p + 'mlp.0.bias':        src_p + 'mlp.fc1.bias',
                dst_p + 'mlp.2.weight':      src_p + 'mlp.fc2.weight',
                dst_p + 'mlp.2.bias':        src_p + 'mlp.fc2.bias',
            }
            for dst_k, src_k in block_map.items():
                if dst_k in sd and src_k in src_sd:
                    sd[dst_k] = src_sd[src_k]

        self.load_state_dict(sd, strict=False)

    def forward(self, x: torch.Tensor, return_layers: list[int] | None = None,
                cross_attn_kv: dict[int, torch.Tensor] | None = None):
        B, C, H, W = x.shape
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_t = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat([cls_t, patches], dim=1)
        x_seq = x_seq + self.pos_embed

        if return_layers is None:
            return_layers = [len(self.blocks) - 1]

        layer_outputs = []
        for i, block in enumerate(self.blocks):
            x_seq = block(x_seq)
            if cross_attn_kv and i in cross_attn_kv:
                x_seq = self.cross_attn[str(i)](x_seq, cross_attn_kv[i])
            if i in return_layers:
                layer_outputs.append(x_seq)

        if len(layer_outputs) == 1:
            return layer_outputs[0]
        return layer_outputs

    def tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        N, _, C = tokens.shape
        patch_tokens = tokens[:, 1:]
        return patch_tokens.transpose(1, 2).reshape(N, C, self.grid_size, self.grid_size)
