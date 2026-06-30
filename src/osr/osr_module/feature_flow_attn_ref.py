"""
Reference implementation of windowed Feature-Flow Attention without materializing
the full [B*H*W, C, W^2] key tensor.

Algorithm (per pixel p):
  Given feature F [C, H, W] and flow U [2, H, W]:
    Q(p)    = q_proj(F[:, p])                          [1, C]
    K_patch = k_proj(F[:, p + delta]) for |delta| <= R  [W^2, C]
    V_patch = U[:, p + delta]                            [W^2, 2]
    scores  = softmax(Q(p) @ K_patch^T / sqrt(C))       [1, W^2]
    out(p)  = scores @ V_patch                           [1, 2]

Three implementations provided:
  1. naive      – per-pixel loop, purely for understanding
  2. fused      – unfold + single matmul (peak memory ~ BW*HW*C*WW)
  3. chunked    – unfold + chunked bmm (reduces peak memory)
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Pure naive reference (per-pixel loop — for understanding only)
# ---------------------------------------------------------------------------
def feature_flow_attn_local_naive(
    feature: torch.Tensor,          # [1, C, H, W]
    flow: torch.Tensor,             # [1, 2, H, W]
    q_proj: torch.nn.Module,
    k_proj: torch.nn.Module,
    radius: int = 2,
) -> torch.Tensor:
    """One pixel at a time.  Readable, slow, zero extra memory."""
    B, C, H, W = feature.shape
    ks = 2 * radius + 1                     # kernel_size = 5
    R = radius
    scale = C ** -0.5

    pad = F.pad(feature, (R, R, R, R))      # [1, C, H+2R, W+2R]
    fpad = F.pad(flow,    (R, R, R, R))

    out = feature.new_zeros(B, 2, H, W)

    for y in range(H):
        for x in range(W):
            q = q_proj(feature[:, :, y, x].view(1, C))               # [1, C]

            patch = pad[:, :, y:y+ks, x:x+ks]                        # [1, C, 5, 5]
            k = k_proj(patch.permute(0, 2, 3, 1).reshape(1, ks*ks, C))  # [1, 25, C]

            v = fpad[:, :, y:y+ks, x:x+ks]                           # [1, 2, 5, 5]
            v = v.permute(0, 2, 3, 1).reshape(1, ks*ks, 2)          # [1, 25, 2]

            score = torch.softmax((q @ k.squeeze(0).T) * scale, dim=-1)  # [1, 25]
            out[:, :, y, x] = (score @ v.squeeze(0)).squeeze(0)      # [2]

    return out


# ---------------------------------------------------------------------------
# 2. Fused: unfold + single bmm (materializes k/v views — peak memory high)
# ---------------------------------------------------------------------------
def feature_flow_attn_local_fused(
    feature: torch.Tensor,          # [B, C, H, W]
    flow: torch.Tensor,             # [B, 2, H, W]
    q_proj: torch.nn.Module,
    k_proj: torch.nn.Module,
    radius: int = 2,
) -> torch.Tensor:
    """Correctness reference.  Equivalent to current forward_local_window_attn."""
    B, C, H, W = feature.shape
    ks = 2 * radius + 1
    WW = ks * ks
    scale = C ** -0.5

    # ---- project queries (all pixels at once) ----
    feat_flat = feature.view(B, C, -1).permute(0, 2, 1)            # [B, HW, C]
    q = q_proj(feat_flat).view(B, H * W, 1, C)                     # [B, HW, 1, C]

    # ---- project keys and unfold ----
    feat_k = k_proj(feat_flat).permute(0, 2, 1).view(B, C, H, W)   # [B, C, H, W]
    k_folded = F.unfold(feat_k, kernel_size=ks, padding=radius)     # [B, C*WW, HW]
    k_folded = k_folded.view(B, C, WW, H*W).permute(0, 3, 1, 2).contiguous()  # [B, HW, C, WW]

    # ---- unfold flow (no projection) ----
    v_folded = F.unfold(flow, kernel_size=ks, padding=radius)       # [B, 2*WW, HW]
    v_folded = v_folded.view(B, 2, WW, H*W).permute(0, 3, 1, 2).contiguous()  # [B, HW, 2, WW]
    v_folded = v_folded.permute(0, 1, 3, 2).contiguous()                      # [B, HW, WW, 2]

    # ---- attention ----
    q_flat = q.reshape(B * H * W, 1, C)
    k_flat = k_folded.reshape(B * H * W, C, WW)
    scores = torch.bmm(q_flat, k_flat) * scale
    scores = scores.view(B, H * W, 1, WW)                              # [B, HW, 1, WW]
    prob = torch.softmax(scores, dim=-1)

    v_flat = v_folded.reshape(B * H * W, WW, 2)
    out = torch.bmm(prob.view(B * H * W, 1, WW), v_flat)               # [BHW, 1, 2]
    return out.view(B, H, W, 2).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# 3. Chunked: unfold-like gather, process pixels in sub-batches
#    Avoids the [B*H*W, C, WW] intermediate — only stores one chunk at a time.
# ---------------------------------------------------------------------------
def feature_flow_attn_local_chunked(
    feature: torch.Tensor,          # [B, C, H, W]
    flow: torch.Tensor,             # [B, 2, H, W]
    q_proj: torch.nn.Module,
    k_proj: torch.nn.Module,
    radius: int = 2,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    The unfolded key/flow tensors [B, C*WW, H*W] are kept in their natural
    folded layout.  Only a chunk of pixels is extracted and reshaped into
    [chunk, C, WW] at a time for the dot product.

    Peak extra memory ≈ chunk_size × C × WW  (vs  BHW × C × WW for fused).
    """
    B, C, H, W = feature.shape
    ks = 2 * radius + 1
    WW = ks * ks
    HW = H * W
    scale = C ** -0.5

    # ---- project queries and keys ----
    feat_flat = feature.view(B, C, HW).permute(0, 2, 1)             # [B, HW, C]
    q_all = q_proj(feat_flat).view(B * HW, 1, C)                    # [BHW, 1, C]

    feat_k = k_proj(feat_flat).permute(0, 2, 1).view(B, C, H, W)    # [B, C, H, W]

    # ---- fold layout: [B, C*WW, HW] for keys, [B, 2*WW, HW] for values ----
    k_fold = F.unfold(feat_k, kernel_size=ks, padding=radius)       # [B, C*WW, HW]
    v_fold = F.unfold(flow,   kernel_size=ks, padding=radius)       # [B, 2*WW, HW]

    out = feature.new_zeros(B, 2, H, W)

    for start in range(0, B * HW, chunk_size):
        end = min(start + chunk_size, B * HW)
        idxs = torch.arange(start, end, device=feature.device)

        b_idx = idxs // HW                                             # [chunk]
        sp_idx = idxs % HW                                             # [chunk]
        h_idx = sp_idx // W
        w_idx = sp_idx % W

        q = q_all[idxs]                                                # [chunk, 1, C]

        # Gather one (C*WW)-element column per pixel from the folded tensor.
        # k_fold: [B, C*WW, HW] -> index along dims 0 and 2 -> [chunk, C*WW]
        k_chunk = k_fold[b_idx, :, sp_idx].view(-1, C, WW)            # [chunk, C, WW]
        v_chunk = v_fold[b_idx, :, sp_idx].view(-1, 2, WW)            # [chunk, 2, WW]

        scores = torch.bmm(q, k_chunk) * scale                       # [chunk, 1, WW]
        prob = torch.softmax(scores, dim=-1)

        flow_out = torch.bmm(prob, v_chunk.permute(0, 2, 1))          # [chunk, 1, 2]

        out[b_idx, 0, h_idx, w_idx] = flow_out[:, 0, 0]
        out[b_idx, 1, h_idx, w_idx] = flow_out[:, 0, 1]

    return out


# ---------------------------------------------------------------------------
# Correctness test
# ---------------------------------------------------------------------------
def test():
    B, C, H, W = 1, 32, 16, 16
    R = 2
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(0)
    feat = torch.randn(B, C, H, W, device=dev)
    flow = torch.randn(B, 2, H, W, device=dev)
    q_p = torch.nn.Linear(C, C, bias=False).to(dev)
    k_p = torch.nn.Linear(C, C, bias=False).to(dev)

    naive = feature_flow_attn_local_naive(feat, flow, q_p, k_p, R)
    fused = feature_flow_attn_local_fused(feat, flow, q_p, k_p, R)
    chunk = feature_flow_attn_local_chunked(feat, flow, q_p, k_p, R, chunk_size=32)

    print(f'naive vs fused  max diff: {(naive - fused).abs().max().item():.6e}')
    print(f'fused vs chunk  max diff: {(fused - chunk).abs().max().item():.6e}')

    assert (naive - fused).abs().max().item() < 1e-5, 'fused mismatch'
    assert (fused - chunk).abs().max().item() < 1e-5, 'chunked mismatch'
    print('OK — all three match')


if __name__ == '__main__':
    test()
