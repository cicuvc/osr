import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
import torch.utils.checkpoint as cp
from flash_attn import flash_attn_func


def checkpointed(module, *args):
    return cp.checkpoint(lambda *x: module(*x), *args)


class Mlp(nn.Module):

    def __init__(self,
                 in_dim,
                 hidden_dim=None,
                 out_dim=None,
                 act_layer=nn.GELU):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1, x_size[-1])
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.view(*x_size[:-1], self.out_dim)
        return x


class VanillaAttention(nn.Module):
    def __init__(self, dim, num_heads=8, proj_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.softmax_temp = self.head_dim ** -0.5
        self.kv_proj = nn.Linear(dim, dim * 2, bias=proj_bias)
        self.q_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.merge = nn.Linear(dim, dim)

    def forward(self, x_q, x_kv=None):
        if x_kv is None:
            x_kv = x_q
        bs, _, dim = x_q.shape

        kv = self.kv_proj(x_kv).reshape(bs, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = self.q_proj(x_q).reshape(bs, -1, self.num_heads, self.head_dim)

        k = kv[0].permute(0, 2, 1, 3).contiguous()
        v = kv[1].permute(0, 2, 1, 3).contiguous()

        x_q = flash_attn_func(q, k, v, softmax_scale=self.softmax_temp)
        x_q = x_q.reshape(bs, -1, dim)
        x_q = self.merge(x_q)
        return x_q


class CrossBidirectionalAttention(nn.Module):
    def __init__(self, dim, num_heads, proj_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.softmax_temp = self.head_dim ** -0.5
        self.qk_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.v_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.merge = nn.Linear(dim, dim, bias=proj_bias)
        self.temperature = nn.Parameter(torch.tensor([0.0]), requires_grad=True)

    def forward(self, x0, x1):
        bs = x0.size(0)

        qk0 = self.qk_proj(x0).reshape(bs, -1, self.num_heads, self.head_dim)
        qk1 = self.qk_proj(x1).reshape(bs, -1, self.num_heads, self.head_dim)
        v0 = self.v_proj(x0).reshape(bs, -1, self.num_heads, self.head_dim)
        v1 = self.v_proj(x1).reshape(bs, -1, self.num_heads, self.head_dim)

        x0 = flash_attn_func(qk0, qk1, v1, softmax_scale=self.softmax_temp)
        x1 = flash_attn_func(qk1, qk0, v0, softmax_scale=self.softmax_temp)

        x0 = self.merge(x0.reshape(bs, -1, self.num_heads * self.head_dim))
        x1 = self.merge(x1.reshape(bs, -1, self.num_heads * self.head_dim))

        return x0, x1


class SwinPosEmbMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pos_embed = None
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, dim, bias=False))

    def forward(self, H, W, ref_H, ref_W, device, dtype):
        if self.pos_embed is None or self.training:
            coords_y = torch.arange(H, device=device, dtype=torch.float32)
            coords_x = torch.arange(W, device=device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing='ij')
            scale_y = ref_H / H
            scale_x = ref_W / W
            grid_y = grid_y * scale_y
            grid_x = grid_x * scale_x
            grid_y = (grid_y - ref_H / 2) / (ref_H / 2)
            grid_x = (grid_x - ref_W / 2) / (ref_W / 2)
            grid = torch.stack([grid_x, grid_y], dim=-1)
            self.pos_embed = self.pos_mlp(grid.reshape(H * W, 2)).unsqueeze(0)
        return self.pos_embed.to(dtype)


class WindowSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_coef, use_pre_pos_embed=False):
        super().__init__()
        self.mlp = Mlp(in_dim=dim*2, hidden_dim=dim*mlp_hidden_coef, out_dim=dim, act_layer=nn.GELU)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = VanillaAttention(dim, num_heads=num_heads)
        self.pos_embed = SwinPosEmbMLP(dim) if use_pre_pos_embed else None

    def forward(self, x, x_pre, H, W, H_pre, W_pre, ref_H, ref_W):
        device, dtype = x.device, x.dtype
        ww = x.shape[1]
        ww_pre = x_pre.shape[1]

        if self.pos_embed is not None:
            x = x + self.pos_embed(H, W, ref_H, ref_W, device, dtype)
        x = torch.cat((x, x_pre), dim=1)
        x = x + self.gamma * self.norm1(self.mlp(torch.cat([x, self.attn(self.norm2(x))], dim=-1)))
        x, x_pre = x.split([ww, ww_pre], dim=1)

        return x, x_pre


class WindowCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_coef):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_dim=dim*2, hidden_dim=dim*mlp_hidden_coef, out_dim=dim, act_layer=nn.GELU)
        self.cross_attn = CrossBidirectionalAttention(dim, num_heads=num_heads, proj_bias=False)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x0, x1):
        m_x0, m_x1 = self.cross_attn(self.norm1(x0), self.norm1(x1))
        x0 = x0 + self.gamma * self.norm2(self.mlp(torch.cat([x0, m_x0], dim=-1)))
        x1 = x1 + self.gamma * self.norm2(self.mlp(torch.cat([x1, m_x1], dim=-1)))
        return x0, x1


class FineProcess(nn.Module):
    def __init__(self, config):
        super().__init__()
        block_dims = config['resnet']['block_dims']
        self.block_dims = block_dims
        self.W_f = config['fine_window_size']
        self.W_m = config['medium_window_size']
        nhead_f = config["fine"]['nhead_fine_level']
        nhead_m = config["fine"]['nhead_medium_level']
        mlp_hidden_coef = config["fine"]['mlp_hidden_dim_coef']

        self.conv_merge = nn.Sequential(
            nn.Conv2d(block_dims[2]*2, block_dims[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(block_dims[1], block_dims[1], kernel_size=3, stride=1, padding=1, groups=block_dims[1], bias=False),
            nn.BatchNorm2d(block_dims[1]))
        self.out_conv_m = nn.Conv2d(block_dims[1], block_dims[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.out_conv_f = nn.Conv2d(block_dims[0], block_dims[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.self_attn_m = WindowSelfAttention(block_dims[1], num_heads=nhead_m,
                                                mlp_hidden_coef=mlp_hidden_coef, use_pre_pos_embed=True)
        self.cross_attn_m = WindowCrossAttention(block_dims[1], num_heads=nhead_m,
                                                  mlp_hidden_coef=mlp_hidden_coef)
        self.self_attn_f = WindowSelfAttention(block_dims[0], num_heads=nhead_f,
                                                mlp_hidden_coef=mlp_hidden_coef, use_pre_pos_embed=True)
        self.cross_attn_f = WindowCrossAttention(block_dims[0], num_heads=nhead_f,
                                                  mlp_hidden_coef=mlp_hidden_coef)
        self.down_proj_m_f = nn.Linear(block_dims[1], block_dims[0], bias=False)

        self.pos_embed_c = SwinPosEmbMLP(block_dims[1])
        self.pos_embed_m = SwinPosEmbMLP(block_dims[1])
        self.pos_embed_m_fine = SwinPosEmbMLP(block_dims[0])
        self.pos_embed_f = SwinPosEmbMLP(block_dims[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _flatten_feat(self, feat):
        return feat.flatten(2).transpose(1, 2)

    def _gather_windows(self, x0_flat, x1_flat, data):
        N = x0_flat.shape[0]
        Hf, Wf = data['hw0_f']
        C = x0_flat.shape[-1]

        x0_map = x0_flat.reshape(N, Hf, Wf, C).permute(0, 3, 1, 2).contiguous()
        x1_map = x1_flat.reshape(N, Hf, Wf, C).permute(0, 3, 1, 2).contiguous()

        stride_f = max(1, Hf // data['hw0_c'][0])
        W_f = self.W_f

        data['stride_f'] = stride_f  # needed by FineMatching for flow scaling

        if W_f == 1:
            feat_f0_unfold = rearrange(x0_map, 'n c h w -> n (h w) 1 c')
            feat_f1_unfold = rearrange(x1_map, 'n c h w -> n (h w) 1 c')
        else:
            u0 = F.unfold(x0_map, kernel_size=(W_f, W_f), stride=stride_f, padding=W_f // 2)
            feat_f0_unfold = rearrange(u0, 'n (c ww) l -> n l ww c', ww=W_f**2).contiguous()
            u1 = F.unfold(x1_map, kernel_size=(W_f, W_f), stride=stride_f, padding=W_f // 2)
            feat_f1_unfold = rearrange(u1, 'n (c ww) l -> n l ww c', ww=W_f**2).contiguous()

        return feat_f0_unfold, feat_f1_unfold

    def pre_process(self, feat_f0, feat_f1, feat_m0, feat_m1, feat_c0, feat_c1, feat_c0_pre, feat_c1_pre, data):
        W_f = self.W_f
        W_m = self.W_m
        data.update({'W_f': W_f, 'W_m': W_m})

        Hc, Wc = data['hw0_c']
        Hm, Wm = data['hw0_m']

        feat_c0 = rearrange(feat_c0, 'n (h w) c -> n c h w', h=Hc, w=Wc)
        feat_c1 = rearrange(feat_c1, 'n (h w) c -> n c h w', h=Hc, w=Wc)
        feat_c0 = self.conv_merge(torch.cat([feat_c0, feat_c0_pre], dim=1))
        feat_c1 = self.conv_merge(torch.cat([feat_c1, feat_c1_pre], dim=1))

        if feat_m0.shape[2] == feat_m1.shape[2] and feat_m0.shape[3] == feat_m1.shape[3]:
            feat_m = self.out_conv_m(torch.cat([feat_m0, feat_m1], dim=0))
            feat_m0, feat_m1 = torch.chunk(feat_m, 2, dim=0)
            feat_f = self.out_conv_f(torch.cat([feat_f0, feat_f1], dim=0))
            feat_f0, feat_f1 = torch.chunk(feat_f, 2, dim=0)
        else:
            feat_m0 = self.out_conv_m(feat_m0)
            feat_m1 = self.out_conv_m(feat_m1)
            feat_f0 = self.out_conv_f(feat_f0)
            feat_f1 = self.out_conv_f(feat_f1)

        feat_c0 = self._flatten_feat(feat_c0)
        feat_c1 = self._flatten_feat(feat_c1)
        feat_m0 = self._flatten_feat(feat_m0)
        feat_m1 = self._flatten_feat(feat_m1)
        feat_f0 = self._flatten_feat(feat_f0)
        feat_f1 = self._flatten_feat(feat_f1)

        return feat_c0, feat_c1, feat_m0, feat_m1, feat_f0, feat_f1

    def forward(self, feat_f0, feat_f1, feat_m0, feat_m1, feat_c0, feat_c1, feat_c0_pre, feat_c1_pre, data):
        Hc, Wc = data['hw0_c']
        Hm, Wm = data['hw0_m']
        Hf, Wf = data['hw0_f']

        feat_c0, feat_c1, feat_m0, feat_m1, feat_f0, feat_f1 = self.pre_process(
            feat_f0, feat_f1, feat_m0, feat_m1, feat_c0, feat_c1, feat_c0_pre, feat_c1_pre, data)

        device = feat_f0.device
        dtype = feat_f0.dtype

        feat_m0 = feat_m0 + self.pos_embed_m(Hm, Wm, Hf, Wf, device, dtype)
        feat_m1 = feat_m1 + self.pos_embed_m(Hm, Wm, Hf, Wf, device, dtype)
        feat_c0 = feat_c0 + self.pos_embed_c(Hc, Wc, Hf, Wf, device, dtype)
        feat_c1 = feat_c1 + self.pos_embed_c(Hc, Wc, Hf, Wf, device, dtype)

        # 1. Self attention (c + m)
        feat_m, _ = self.self_attn_m(
            torch.cat([feat_m0, feat_m1], dim=0),
            torch.cat([feat_c0, feat_c1], dim=0),
            Hm, Wm, Hc, Wc, Hf, Wf)
        feat_m0, feat_m1 = torch.chunk(feat_m, 2, dim=0)

        # 2. Cross attention
        feat_m0, feat_m1 = self.cross_attn_m(feat_m0, feat_m1)

        # 3. medium-fine
        feat_m = self.down_proj_m_f(torch.cat([feat_m0, feat_m1], dim=0))
        feat_m0, feat_m1 = torch.chunk(feat_m, 2, dim=0)

        feat_f0 = feat_f0 + self.pos_embed_f(Hf, Wf, Hf, Wf, device, dtype)
        feat_f1 = feat_f1 + self.pos_embed_f(Hf, Wf, Hf, Wf, device, dtype)
        feat_m0 = feat_m0 + self.pos_embed_m_fine(Hm, Wm, Hf, Wf, device, dtype)
        feat_m1 = feat_m1 + self.pos_embed_m_fine(Hm, Wm, Hf, Wf, device, dtype)

        # 4. Self attention (m + f)
        feat_f, _ = self.self_attn_f(
            torch.cat([feat_f0, feat_f1], dim=0),
            torch.cat([feat_m0, feat_m1], dim=0),
            Hf, Wf, Hm, Wm, Hf, Wf)
        feat_f0, feat_f1 = torch.chunk(feat_f, 2, dim=0)

        # 5. Cross attention
        feat_f0, feat_f1 = self.cross_attn_f(feat_f0, feat_f1)

        feat_f0_unfold, feat_f1_unfold = self._gather_windows(feat_f0, feat_f1, data)

        return feat_f0_unfold, feat_f1_unfold
