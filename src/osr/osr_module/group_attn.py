import torch
import triton
import triton.language as tl


def group_attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float) -> torch.Tensor:
    """
    q, k: [B, N, C]
    v: [B, N, 2]
    where B is large, N < 32 and C = 128
    """
    sim = q @ k.transpose(-1, -2)
    probs = torch.softmax(sim * sm_scale, -1)

    return probs @ v


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BATCH_B': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 32, 'BATCH_B': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 32, 'BATCH_B': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_C': 64, 'BATCH_B': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BATCH_B': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 32, 'BATCH_B': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BATCH_B': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BATCH_B': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BATCH_B': 2}, num_warps=4, num_stages=1),
    ],
    key=['N', 'C'],
)
@triton.jit
def _group_attention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    sm_scale,
    stride_qb, stride_qn, stride_qc,
    stride_kb, stride_kn, stride_kc,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    B, N: tl.constexpr, C: tl.constexpr, MAX_N: tl.constexpr,
    BLOCK_C: tl.constexpr, BATCH_B: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    b_start = pid * BATCH_B
    while b_start < B:
        b_offs = b_start + tl.arange(0, BATCH_B)
        b_mask = b_offs < B

        n_range = tl.arange(0, MAX_N)
        d_range = tl.arange(0, 2)
        n_mask = n_range < N

        q_base = q_ptr + b_offs[:, None, None] * stride_qb
        k_base = k_ptr + b_offs[:, None, None] * stride_kb
        v_base = v_ptr + b_offs[:, None, None] * stride_vb
        o_base = out_ptr + b_offs[:, None, None] * stride_ob

        S = tl.zeros((BATCH_B, MAX_N, MAX_N), dtype=tl.float32)

        for c_start in range(0, C, BLOCK_C):
            c_offs = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offs < C

            q_block = tl.load(
                q_base + n_range[None, :, None] * stride_qn + c_offs[None, None, :] * stride_qc,
                mask=b_mask[:, None, None] & n_mask[None, :, None] & c_mask[None, None, :],
                other=0.0,
            )

            k_block = tl.load(
                k_base + n_range[None, :, None] * stride_kn + c_offs[None, None, :] * stride_kc,
                mask=b_mask[:, None, None] & n_mask[None, :, None] & c_mask[None, None, :],
                other=0.0,
            )

            S += tl.dot(q_block, tl.trans(k_block))

        S = S * sm_scale

        row_mask = n_mask[None, :, None] & n_mask[None, None, :]
        S = tl.where(b_mask[:, None, None] & row_mask, S, float('-inf'))
        S_max = tl.max(S, axis=2, keep_dims=True)
        S = S - S_max
        S = tl.exp(S)
        S = tl.where(b_mask[:, None, None] & row_mask, S, 0.0)
        S_sum = tl.sum(S, axis=2, keep_dims=True) + 1e-9
        S = S / S_sum

        v_vals = tl.load(
            v_base + n_range[None, :, None] * stride_vn + d_range[None, None, :] * stride_vd,
            mask=b_mask[:, None, None] & n_mask[None, :, None], other=0.0,
        )
        out_vals = tl.dot(S.to(v_vals.dtype), v_vals)

        tl.store(
            o_base + n_range[None, :, None] * stride_on + d_range[None, None, :] * stride_od,
            out_vals, mask=b_mask[:, None, None] & n_mask[None, :, None],
        )

        b_start += BATCH_B * num_pids


def group_attention_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float) -> torch.Tensor:
    B, N, C = q.shape
    assert k.shape == (B, N, C)
    assert v.shape == (B, N, 2)

    out = torch.empty(B, N, 2, device=q.device, dtype=q.dtype)

    MAX_N = max(triton.next_power_of_2(N), 16)
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid = (min(num_sms * 2, B),)

    _group_attention_fwd_kernel[grid](
        q, k, v, out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        B, N, C, MAX_N,
    )

    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BATCH_B': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 32, 'BATCH_B': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BATCH_B': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BATCH_B': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BATCH_B': 2}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_C': 32, 'BATCH_B': 1}, num_warps=4, num_stages=2),
    ],
    key=['N', 'C'],
)
@triton.jit
def _group_attention_bwd_kernel(
    q_ptr, k_ptr, v_ptr, dO_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    sm_scale,
    stride_qb, stride_qn, stride_qc,
    stride_kb, stride_kn, stride_kc,
    stride_vb, stride_vn, stride_vd,
    stride_dob, stride_don, stride_dod,
    stride_dqb, stride_dqn, stride_dqc,
    stride_dkb, stride_dkn, stride_dkc,
    stride_dvb, stride_dvn, stride_dvd,
    B, N: tl.constexpr, C: tl.constexpr, MAX_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr, BATCH_B: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    b_start = pid * BATCH_B
    while b_start < B:
        b_offs = b_start + tl.arange(0, BATCH_B)
        b_mask = b_offs < B

        n_range = tl.arange(0, MAX_N)
        d_range = tl.arange(0, 2)
        n_mask = n_range < N

        q_base = q_ptr + b_offs[:, None, None] * stride_qb
        k_base = k_ptr + b_offs[:, None, None] * stride_kb
        v_base = v_ptr + b_offs[:, None, None] * stride_vb
        dO_base = dO_ptr + b_offs[:, None, None] * stride_dob
        dq_base = dq_ptr + b_offs[:, None, None] * stride_dqb
        dk_base = dk_ptr + b_offs[:, None, None] * stride_dkb
        dv_base = dv_ptr + b_offs[:, None, None] * stride_dvb

        # --- Phase 1: forward pass to compute P = softmax(q @ k^T * sm_scale) ---
        S = tl.zeros((BATCH_B, MAX_N, MAX_N), dtype=tl.float32)

        for c_start in range(0, C, BLOCK_C):
            c_offs = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offs < C

            q_block = tl.load(
                q_base + n_range[None, :, None] * stride_qn + c_offs[None, None, :] * stride_qc,
                mask=b_mask[:, None, None] & n_mask[None, :, None] & c_mask[None, None, :],
                other=0.0,
            )
            k_block = tl.load(
                k_base + n_range[None, :, None] * stride_kn + c_offs[None, None, :] * stride_kc,
                mask=b_mask[:, None, None] & n_mask[None, :, None] & c_mask[None, None, :],
                other=0.0,
            )
            S += tl.dot(q_block, tl.trans(k_block))

        S = S * sm_scale

        row_mask = n_mask[None, :, None] & n_mask[None, None, :]
        S = tl.where(b_mask[:, None, None] & row_mask, S, float('-inf'))
        S_max = tl.max(S, axis=2, keep_dims=True)
        S = S - S_max
        S = tl.exp(S)
        S = tl.where(b_mask[:, None, None] & row_mask, S, 0.0)
        S_sum = tl.sum(S, axis=2, keep_dims=True) + 1e-9
        S = S / S_sum  # S is now P

        # --- Phase 2: dv, dP, dS ---
        v_vals = tl.load(
            v_base + n_range[None, :, None] * stride_vn + d_range[None, None, :] * stride_vd,
            mask=b_mask[:, None, None] & n_mask[None, :, None], other=0.0,
        )
        dO_vals = tl.load(
            dO_base + n_range[None, :, None] * stride_don + d_range[None, None, :] * stride_dod,
            mask=b_mask[:, None, None] & n_mask[None, :, None], other=0.0,
        )

        dv_vals = tl.dot(tl.trans(S), dO_vals)
        tl.store(
            dv_base + n_range[None, :, None] * stride_dvn + d_range[None, None, :] * stride_dvd,
            dv_vals, mask=b_mask[:, None, None] & n_mask[None, :, None],
        )

        O_vals = tl.dot(S.to(v_vals.dtype), v_vals)
        dP_sum = tl.sum(dO_vals * O_vals, axis=2, keep_dims=True)

        d_range_padded = tl.arange(0, BLOCK_D)
        dO_padded = tl.load(
            dO_base + n_range[None, :, None] * stride_don + d_range_padded[None, None, :] * stride_dod,
            mask=b_mask[:, None, None] & n_mask[None, :, None] & (d_range_padded < 2)[None, None, :],
            other=0.0,
        )
        v_padded = tl.load(
            v_base + n_range[None, :, None] * stride_vn + d_range_padded[None, None, :] * stride_vd,
            mask=b_mask[:, None, None] & n_mask[None, :, None] & (d_range_padded < 2)[None, None, :],
            other=0.0,
        )
        dP = tl.dot(dO_padded, tl.trans(v_padded))

        S = tl.where(b_mask[:, None, None] & row_mask, S * (dP - dP_sum), 0.0)
        S = S * sm_scale  # S is now dS

        # --- Phase 3: dq = dS @ k, dk = dS^T @ q ---
        for c_start in range(0, C, BLOCK_C):
            c_offs = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offs < C

            q_tile = tl.load(
                q_base + n_range[None, :, None] * stride_qn + c_offs[None, None, :] * stride_qc,
                mask=b_mask[:, None, None] & n_mask[None, :, None] & c_mask[None, None, :],
                other=0.0,
            )
            k_tile = tl.load(
                k_base + n_range[None, :, None] * stride_kn + c_offs[None, None, :] * stride_kc,
                mask=b_mask[:, None, None] & n_mask[None, :, None] & c_mask[None, None, :],
                other=0.0,
            )

            dq_tile = tl.dot(S, k_tile)
            dk_tile = tl.dot(tl.trans(S), q_tile)

            tl.store(
                dq_base + n_range[None, :, None] * stride_dqn + c_offs[None, None, :] * stride_dqc,
                dq_tile, mask=b_mask[:, None, None] & n_mask[None, :, None] & c_mask[None, None, :],
            )
            tl.store(
                dk_base + n_range[None, :, None] * stride_dkn + c_offs[None, None, :] * stride_dkc,
                dk_tile, mask=b_mask[:, None, None] & n_mask[None, :, None] & c_mask[None, None, :],
            )

        b_start += BATCH_B * num_pids


def group_attention_triton_bwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    dO: torch.Tensor, sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, C = q.shape
    assert k.shape == (B, N, C)
    assert v.shape == (B, N, 2)
    assert dO.shape == (B, N, 2)

    dq = torch.empty(B, N, C, device=q.device, dtype=torch.float32)
    dk = torch.empty(B, N, C, device=q.device, dtype=torch.float32)
    dv = torch.empty(B, N, 2, device=q.device, dtype=torch.float32)

    BLOCK_D = 16
    MAX_N = max(triton.next_power_of_2(N), 16)
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid = (min(num_sms * 2, B),)

    _group_attention_bwd_kernel[grid](
        q, k, v, dO,
        dq, dk, dv,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        B, N, C, MAX_N, BLOCK_D,
    )

    return dq, dk, dv


class GroupAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        ctx.sm_scale = sm_scale
        ctx.save_for_backward(q, k, v)
        return group_attention_triton(q, k, v, sm_scale)

    @staticmethod
    def backward(ctx, dO):
        q, k, v = ctx.saved_tensors
        dq, dk, dv = group_attention_triton_bwd(q, k, v, dO.contiguous(), ctx.sm_scale)
        return dq, dk, dv, None


if __name__ == "__main__":
    torch.manual_seed(42)

    def _ref_bwd(q, k, v, dO, sm_scale):
        sim = q @ k.transpose(-1, -2)
        P = torch.softmax(sim * sm_scale, -1)
        dv = P.transpose(-1, -2) @ dO
        dP = dO @ v.transpose(-1, -2)
        dS = P * (dP - (P * dP).sum(-1, keepdim=True)) * sm_scale
        dq = dS @ k
        dk = dS.transpose(-1, -2) @ q
        return dq, dk, dv

    for N_val in [16, 24, 32]:
        B, C = 256, 128
        q = torch.randn(B, N_val, C, device='cuda')
        k = torch.randn(B, N_val, C, device='cuda')
        v = torch.randn(B, N_val, 2, device='cuda')
        sm_scale = 0.125
        dO = torch.randn(B, N_val, 2, device='cuda')

        ref_out = group_attention_ref(q, k, v, sm_scale)
        tri_out = group_attention_triton(q, k, v, sm_scale)
        fwd_err = (ref_out - tri_out).abs().max().item()

        ref_dq, ref_dk, ref_dv = _ref_bwd(q, k, v, dO, sm_scale)
        tri_dq, tri_dk, tri_dv = group_attention_triton_bwd(q, k, v, dO, sm_scale)
        bwd_err = max(
            (ref_dq - tri_dq).abs().max().item(),
            (ref_dk - tri_dk).abs().max().item(),
            (ref_dv - tri_dv).abs().max().item(),
        )

        print(f"N={N_val:2d}  fwd_max_diff={fwd_err:.2e}  bwd_max_diff={bwd_err:.2e}")
        torch.testing.assert_close(ref_out, tri_out, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ref_dq, tri_dq, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ref_dk, tri_dk, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ref_dv, tri_dv, rtol=1e-3, atol=1e-3)

    print("PASSED")
